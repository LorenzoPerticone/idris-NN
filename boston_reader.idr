

import Data.Buffer
import Data.String

import Data.Fin
import Data.Vect
import Data.HVect

import Tensor
import Layer
import NeuralNet
import Floating

%access public export
%default total

runMaybe : b -> (a -> b) -> Maybe a -> b
runMaybe y _ Nothing  = y
runMaybe _ f (Just x) = f x

splitVect : Vect (l1 + l2) a -> (Vect l1 a, Vect l2 a)
splitVect {l1 = Z} xs = ([], xs)
splitVect {l1 = S l1} (x :: xs) = let (ys, zs) = splitVect xs
                                  in (x :: ys, zs)
                                  
infixl 0 <||>
(<||>) : (a1 -> a2) -> (b1 -> b2) -> (a1, b1) -> (a2, b2)
(<||>) f g (x, y) = (f x, g y)

dataset : String
dataset = "boston_dataset.txt"
data_len : Nat
data_len = 506
features : Nat
features = 13

-- Either's arent `Traversable`, so `sequence` won't work
fromIOEither : IO (Either a b) -> IO (Maybe b)
fromIOEither = map eitherToMaybe

-- Read k lines from a supplied file as a single `IO` action that
-- `Maybe` returns a `Vect` of strings
getLines : (k : Nat) -> File -> IO (Maybe (Vect k String))
-- `sequence` swaps a `Traversable` with an `Applicative`;
-- swaps `Vect` with `IO` and then with `Maybe`:
-- `Vect k (IO (Maybe a))` => `IO (Vect k (Maybe a))` => `IO (Maybe (Vect k a))`
getLines k file = map sequence . sequence $ helper k file
-- Read k lines from a supplied file as a `Vect` of IO actions
-- that *might* return a string
  where helper : (k : Nat) -> File -> Vect k (IO (Maybe String))
        helper Z     _ = []
        -- `fGetLine` returns an `IO (Either _ String)`,
        -- so we convert it to `IO (Maybe String)`
        helper (S k) f = (fromIOEither $ fGetLine f) :: (helper k f)

-- Open a `File` handler, read l lines form it and return them as a single
-- `IO` action that can `Maybe` supply a `Vect` of `String`s
readFile : String -> (l : Nat) -> IO (Maybe (Vect l String))
readFile fileName l = do
  -- `openFile` returns an `IO (Either _ File)`, so we convert it to `IO (Maybe File)`
  -- for ease of use (`Maybe` is `Traversable`, while `Either` isn't)
  file <- fromIOEither $ openFile fileName Read
  map join . sequence $ map (getLines l) file

-- Parse a line from a file; it's expected to supply *exactly* k space (" ")
-- separated `Double`s. If either more or less are present, or anything fails
-- the conversion, everything explodes.
parseLine : (k : Nat) -> String -> Vect k Double
parseLine k s =
--    splits by " " and removes empty strings, parses every `String` into
--    a `Maybe Double`, then converts `Maybe`s to `List`s, then concatenates
--    all of these.
  let l = join . map (toList . parseDouble) . filter non_empty $ split isSpace s
--    *assumes* length of result from previous line is *exactly* `k`.
--    VERY BAD.
      lemma = the (k = length l) $ believe_me ()
  in rewrite lemma in fromList l
  where non_empty : String -> Bool
        non_empty "" = False
        non_empty _ = True

parseDataset : IO (Maybe (Tensor 1 [506] (Tensor 1 [14] Double)))
parseDataset =
  let parseLines = map $ parseLine 14
      from = map $ fromVect 1 [14]
  in map (map $ fromVect 1 [506] . from . parseLines) $ readFile dataset data_len
  
parse_train_label : IO (Maybe (Tensor 1 [506] (Tensor 1 [13] Double),
                               Tensor 1 [506] (Tensor 1 [1] Double)))
parse_train_label =
  let parse1 = parseLine 14
      from1 = (fromVect 1 [13]) <||> (fromVect 1 [1])
      parse2 = map $ from1 . splitVect . parse1
      from2 = fromVect 1 [506]
      parse = map (map $ unzip . from2 . parse2)
  in parse $ readFile dataset data_len
  
parseChunk : (k : Nat) -> IO (Maybe (Tensor 1 [k] (Tensor 1 [14] Double)))
parseChunk k =
  let parseLines = map $ parseLine 14
      from = map $ fromVect 1 [14]
  in map (map $ fromVect 1 [k] . from . parseLines) $ readFile dataset k
  
parseChunk_train_label : (k : Nat) -> 
                         (IO (Maybe (Tensor 1 [k] (Tensor 1 [13] Double),
                                     Tensor 1 [k] (Tensor 1 [1] Double))))
parseChunk_train_label k =
  let parse1 = parseLine 14
      from1 = (fromVect 1 [13]) <||> (fromVect 1 [1])
      parse2 = map $ from1 . splitVect . parse1
      from2 = fromVect 1 [k]
      parse = map (map $ unzip . from2 . parse2)
  in parse $ readFile dataset k

-- Neural Network

constant : Double
constant = 0.0001

RawLabelType : Type
RawLabelType = Tensor 1 [1] Double

{-
layer1 : Layer 1 1 [13] [2] Double
layer1 = let w = pure 0
             b = pure 0
         in MkLayer w b softmax_activ softmax_activ_g

NetType : Type
NetType = NeuralNet 0 [1, 1] [[13], [2]] 1 [2] Double

NetInput : Type
NetInput = Tensor 1 [13] Double

NetOutput : Type
NetOutput = Tensor 1 [2] Double

net : NetType
net = FstLayer layer1
-}

--{-
layer1 : Layer 1 1 [13] [20] Double
layer1 = let w = pure constant
             b = pure constant
         in MkLayer w b relu relu_g

layer2 : Layer 1 1 [20] [10] Double
layer2 = let w = pure constant
             b = pure constant
         in MkLayer w b softmax softmax_g

layer3 : Layer 1 1 [10] [2] Double
layer3 = let w = pure constant
             b = pure constant
         in MkLayer w b softmax softmax_g

NetType : Type
NetType = NeuralNet 3 [1, 1, 1, 1] [[13], [20], [10], [2]] Double

NetInput : Type
NetInput = Tensor 1 [13] Double

NetOutput : Type
NetOutput = Tensor 1 [2] Double

net : NetType
net = fromLayers [layer1, layer2, layer3, [crossE, crossE_g]]
--net = AddLayer (AddLayer (FstLayer layer3) layer2) layer1
--}

-- The network will decide if the last column should be > 20:
-- "[1, 0]" means "< 20", and "[0, 1]" means "> 20".
--convertLabel : Tensor 1 [1] Double -> Tensor 1 [2] Double
convertLabel : RawLabelType -> NetOutput
convertLabel (Vector ((Scalar x) :: [])) = if x < 25
                                           then fromVect 1 [2] [1, 0]
                                           else fromVect 1 [2] [0, 1]
                                           
trainB : Double -> (batch_len : Nat) -> NetType ->
         Maybe (Tensor 1 [batch_len] NetInput, Tensor 1 [batch_len] NetOutput) ->
         Maybe (NetType)
trainB c batch_len net dat = do
  (dats, labs) <- dat
  let (scores, net, _) = runNetBatch_bp net dats labs
  pure net
  
accuracy : Double
accuracy = 1.0 / 10

round : Double -> Double
round x = if abs x < accuracy
      then 0
      else if abs (x - 1) < accuracy
           then 1
           else x
      
printer : (a -> String) -> Maybe a -> IO ()
printer f s =
  putStrLn $ runMaybe "Reading Error." f s

batch_size : Nat
batch_size = 506 --253

learning_coeff : Double
learning_coeff = 7

main : IO ()
main = do
  putStrLn "Training Data:"
  dats <- map (map (id <||> map convertLabel)) $ parseChunk_train_label batch_size
--  dats <- map (map (id <||> map convertLabel)) parse_train_label
  let output1 = map (flatten <||> flatten) dats
  printer show output1

  putStrLn "Net at the beginning:"
  putStrLn . show $ net
  
  putStrLn "Net results at the beginning:"
  let tmpRes = map (runNetBatch net) $ map fst dats
  printer show tmpRes

  putStrLn "Training..."
  let nnet = trainB learning_coeff batch_size net dats
  
  putStrLn "Net after training:"
  printer show nnet
  
  putStrLn "Results:"
  let results = (map runNetBatch nnet) <*> (map fst dats)
  printer show results
  
  putStrLn "Accuracy:"
  let labels = map snd dats
  let mm = Just . the Double $ fromInteger (toIntegerNat batch_size)
  let m = map fromInteger $ do
    r <- map (map (map round)) results
    l <- labels
    let v = unVector $ (map (==) r) <*> l
    let t = toList $ map unScalar v
    pure . toIntegerNat . length $ filter id t


  putStr "Correct guesses: "
  printer show m
  putStr "Total guesses: "
  printer show mm
  putStr "Correct / Total: "
  printer show $ (map (/) m) <*> mm

--  putStrLn ?output
