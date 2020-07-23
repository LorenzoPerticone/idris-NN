module IdxReader

import Data.Buffer
import Data.Bits

import Data.Vect

import Tensor

%access public export
%default total

data IdxType : Type where
  IdxUByte  : IdxType -- unsigned byte
  IdxSByte  : IdxType -- signed byte
  IdxSInt   : IdxType -- short int
  IdxInt    : IdxType -- int
  IdxFloat  : IdxType -- float
  IdxDouble : IdxType -- double

partial
contentIdxType : Bits8 -> IdxType
contentIdxType 8  = IdxUByte
contentIdxType 9  = IdxSByte
contentIdxType 11 = IdxSInt
contentIdxType 12 = IdxInt
contentIdxType 13 = IdxFloat
contentIdxType 14 = IdxDouble

idxType : IdxType -> Type
idxType IdxUByte  = Integer
idxType IdxSByte  = Integer
idxType IdxSInt   = Integer
idxType IdxInt    = Integer
idxType IdxFloat  = Double
idxType IdxDouble = Double

idxSize : IdxType -> Int
idxSize IdxUByte  = 1
idxSize IdxSByte  = 1
idxSize IdxSInt   = 2
idxSize IdxInt    = 4
idxSize IdxFloat  = 4
idxSize IdxDouble = 8

partial
dataReader : (idx : IdxType) -> Buffer -> Int -> IO (idxType idx)
dataReader IdxUByte  buf loc = 
  map (bitsToInt' {n = 8}) $ getByte buf loc            --should be correct
dataReader IdxSByte  buf loc = do
  px <- map (bitsToInt' {n = 8}) $ getByte buf loc
  mx <- map (bitsToInt . complement . MkBits {n = 8}) $ getByte buf loc
  pure $ if px > 128
     then 128 - mx                                      -- is this correct?
     else px
dataReader IdxSInt   buf loc = do
  x1 <- map (bitsToInt' {n = 8}) $ getByte buf loc
  x2 <- map (bitsToInt' {n = 8}) $ getByte buf (loc + 1)
  pure $ 256*x1 + x2                                     -- should be correct
dataReader IdxInt    buf loc = map cast $ getInt buf loc -- correct
--dataReader IdxFloat  buf loc = ?rhs_float
dataReader IdxDouble buf loc = getDouble buf loc         -- correct

-- First buffer, only 4 bytes read:

-- Content type
partial
getIdxType : Buffer -> IO (IdxType)
getIdxType buf =
  let io_idx = getByte buf 2
  in do
    idx <- io_idx
    pure . contentIdxType $ idx

-- Tensor dimensions
getRank : Buffer -> IO (Nat)
getRank buf = 
  let io_r = getByte buf 3
  in do
    r <- io_r
    pure . fromIntegerNat . bitsToInt {n = 8} . MkBits $ r


-- Second buffer, `rank`*4 bytes read:

-- length over a single dimension (specifically, the `loc`-th dimension)
getLen : Buffer -> (loc : Int) -> IO (Nat)
getLen buf loc =
  let io_l = getInt buf loc
  in do
    l <- io_l
    pure . fromIntegerNat . cast $ l

-- Assembly of all dimensions in the shape vector
getShape : Buffer -> (rank : Nat) -> IO (Vect rank Nat)
getShape buf rank = sequence . map (getLen buf) $ range rank
  where range : (rank : Nat) -> Vect rank Int
        range Z = []
        range (S r) = 4 * (toIntNat r) :: range r

-- Hardly readable because idris lacks a usable MTL library
-- (that's the cause of all these `map`, `<*>`, 
--  `sequence` and nested `do` blocks).
-- Will probaly supply it myself soon enough...
partial
readHead : File -> IO (Maybe ((Buffer, IdxType), (rank : Nat ** Vect rank Nat)))
readHead file = do
  let chunk = 4
  -- Create Buffer, 4 bytes
  m_buf <- newBuffer chunk
  -- Read first "chunk"
  m_buf <- sequence $ do
    buf <- m_buf
    pure $ readBufferFromFile file buf chunk
  -- Extract type and rank
  m_idx <- sequence $ map getIdxType m_buf
  m_rank <- sequence $ map getRank m_buf
  -- New chunk size: `rank` integers = `4*rank` bytes
  let m_size = map ((4 *) . cast) m_rank
  -- Buffer must be bigger, but needs to remember its position
  m_buf <- map join . sequence $ map resizeBuffer m_buf <*> m_size
  -- Read second "chunk" containing the shape
  m_buf <- sequence $ map (readBufferFromFile file) m_buf <*> m_size
  -- Storage of rank and shape in a dependent pair `(rank : Nat ** Vect rank Nat)`
  m_dcouple <- sequence $ do
    rank <- m_rank
    buf <- m_buf
    pure . map (MkDPair rank) $ getShape buf rank
  -- Storage of inner type with rank and shape in an ordinary pair `(IdxType, [...])`
  let m_stat = map MkPair m_buf <*> m_idx
  pure $ map MkPair m_stat <*> m_dcouple

partial
readRest : File -> Buffer -> (idx : IdxType) ->
           (rank : Nat) -> (shape : Vect rank Nat) ->
           IO (Maybe (Tensor rank shape (idxType idx)))
-- Should just be a base case for recursion, hope no-one makes a 0-rank idx file
-- just for storing a scalar (and in that case, we'd better hope it *is* a scalar)!
readRest file buf idx  Z        []          = do
  let size = idxSize idx
  m_buf <- resizeBuffer buf size
  m_buf <- sequence $ map (readBufferFromFile file) m_buf <*> (pure size)
  sequence . pure . map Scalar $ dataReader idx buf 0
readRest file buf idx (S rank) (Z :: shape) = 
  pure . pure $ Vector []
readRest file buf idx (S rank) (S l :: shape) = do
  m_hs <- readRest file buf idx rank shape -- "the index in the last dimension changes the fastest"
  m_ts <- readRest file buf idx (S rank) (l :: shape)
  pure $ do
    hs <- m_hs
    ts <- m_ts
    pure . Vector $ hs :: (unVector ts)

partial
readFile : File -> IO (Maybe (idx : IdxType **
                              rank : Nat **
                              shape : Vect rank Nat **
                              Tensor rank shape (idxType idx)))
readFile file = do stuff <- readHead file
                   let res = do
                     ((buf, idx), (rank ** shape)) <- stuff
                     let t = readRest file buf idx rank shape
                     pure $ map (map $ helper idx rank shape) t
                   map join . sequence $ res
  where helper : (idx : IdxType) ->
                 (rank : Nat) ->
                 (shape : Vect rank Nat) ->
                 Tensor rank shape (idxType idx) ->
                 (idx : IdxType **
                  rank : Nat **
                  shape : Vect rank Nat **
                  Tensor rank shape (idxType idx))
        helper idx rank shape t = (idx ** rank ** shape ** t)
