module Tensor

import Data.Fin
import Data.Vect
import Data.HVect

%access public export
%default total

-- Utilities
infixl 0 |*>
(|*>) : Num value => Functor f =>
        value -> f value -> f value
x |*> y = map (x *) y

infixl 0 <*|
(<*|) : Num value => Functor f =>
        f value -> value -> f value
y <*| x = map (* x) y

{-
Of course the equality
  x |*> y == y <*| x
will hold as long as (*) for `value` is commutative
-}

-- Pointwise `Num` implementation for `Vect`
implementation Num value => Num (Vect n value) where
  (+) = liftA2 (+)
  (*) = liftA2 (*)
  fromInteger = pure . fromInteger
  
implementation Neg value => Neg (Vect n value) where
  negate = map negate
  (-) = liftA2 (-)
  
-- Actual tensors
--  Important disclaimer: the notion of "tensor" in this library is
--  *more general* than the usual mathematical notion as an element
--  of the tensor algebra for a vector space: in that case, a tensor
--  would be an "hyper-cube" of scalars (once you choose a basis).
--  For the purposes of this library, it just is an "hyper-rectangle",
--  whose sides aren't required to have the same length.
data Tensor : (rank : Nat) ->
              (shape : Vect rank Nat) ->
              (value : Type) ->
              Type where
  Scalar : value -> Tensor Z [] value
  Vector : Vect n (Tensor rank shape value) ->
           Tensor (S rank) (n :: shape) value

-- Utility: deconstructors
unScalar : Tensor Z [] value -> value
unScalar (Scalar x) = x

unVector : Tensor (S k) (l :: ls) value -> Vect l (Tensor k ls value)
unVector (Vector v) = v

-- Implementations: bookkeeping, mostly

-- map: lifting operations from "scalars" to tensors
implementation Functor (Tensor rank shape) where
  map f (Scalar x) = Scalar $ f x
  map f (Vector v) = Vector $ map (map f) v
  
-- pure: repeats a value all over the tensor
-- <*>: applies a tensor of functions to a tensor of arguments
--      (producing a tensor of results)
implementation Applicative (Tensor rank shape) where
  pure {shape = []} x = Scalar x
  pure {shape = l :: shape} x = Vector $ pure (pure x)

  (Scalar f)  <*> (Scalar x) = Scalar $ f x
  (Vector fs) <*> (Vector v) = Vector $ (map (<*>) fs) <*> v

-- if we can check for scalar equality, we can check for tensor equality
implementation Eq value => Eq (Tensor rank shape value) where
  (Scalar x) == (Scalar y) = x == y
  (Vector v) == (Vector w) = v == w

-- if the scalars are ordered we can order tensors lexicographically
implementation Ord value => Ord (Tensor rank shape value) where
  compare (Scalar x) (Scalar y) = compare x y
  compare (Vector v) (Vector w) = compare v w

-- Easy print (quite involved, but we want newlines!)
implementation Show value => Show (Tensor rank shape value) where
  show {rank = Z}   (Scalar x)                    = show x
  show {rank = S Z} (Vector v)                    = show v
  show {rank = S (S rank)} (Vector [])            = " "
  show {rank = S (S rank)} (Vector [x])           = "[" ++ show x ++ "]"
  show {rank = S (S rank)} (Vector (x :: y :: v)) = 
    let shows = map show v
    in "[" ++ show x ++ "\n" ++ (helper $ (show y) :: shows)
    where 
          helper : Vect (S k) String -> String
          helper [l]      = " " ++ l ++ "]"
          helper (l1 :: l2 :: ls) =
            (" " ++ l1 ++ ",\n") ++ (helper$ l2 :: ls)

-- ring structure on scalars induces "pointwise" ring structure on tensors
-- (actually, the underlying abelian group is just a monoid here)
implementation Num value => Num (Tensor rank shape value) where
  (+) = liftA2 (+)
  (*) = liftA2 (*)
  fromInteger = pure . fromInteger

-- ring structure on scalars induces "pointwise" ring structure on tensors
-- (here the underlying abelian group is required to be a group, at least)
implementation Neg value => Neg (Tensor rank shape value) where
  negate = map negate
  (-) = liftA2 (-)

-- zip/unzip, useful for batch-training:
zip : Tensor rank shape a ->
      Tensor rank shape b ->
      Tensor rank shape (a, b)
zip (Scalar x) (Scalar y) = Scalar (x, y)
zip (Vector xs) (Vector ys) = Vector $ zipWith zip xs ys

unzip : Tensor rank shape (a, b) ->
        (Tensor rank shape a,
         Tensor rank shape b)
unzip (Scalar (x, y)) = (Scalar x, Scalar y)
unzip (Vector vw)     = let (v, w) = unzip $ map unzip vw
                        in (Vector v, Vector w)

zip3 : Tensor rank shape a ->
       Tensor rank shape b ->
       Tensor rank shape c ->
       Tensor rank shape (a, b, c)
zip3 (Scalar x) (Scalar y) (Scalar z) = Scalar (x, y, z)
zip3 (Vector u) (Vector v) (Vector w) = Vector $ map zip3 u <*> v <*> w

unzip3 : Tensor rank shape (a, b, c) ->
         (Tensor rank shape a,
          Tensor rank shape b,
          Tensor rank shape c)
unzip3 (Scalar (x, y, z)) = (Scalar x, Scalar y, Scalar z)
unzip3 (Vector uvw)       = let (u, v, w) = unzip3 $ map unzip3 uvw
                            in (Vector u, Vector v, Vector w)

-- Conversion utilities: dependent typing can be key here!

-- foldr/foldl surrogates (a full Foldable implementation is impossible!)
foldrT : (value -> value -> value) -> value ->
         Tensor rank shape value -> value
foldrT f x0 (Scalar x) = f x x0
foldrT f x0 (Vector v) = foldr f x0 $ map (foldrT f x0) v

foldlT : (value -> value -> value) -> value ->
         Tensor rank shape value -> value
foldlT f x0 (Scalar x) = f x0 x
foldlT f x0 (Vector v) = foldl f x0 $ map (foldlT f x0) v

-- Single index accessors

-- Out-of-Bounds is checked at the type-level (Fin l)
infixl 0 !!!!
(!!!!) : Tensor (S rank) (l :: shape) value ->
         Fin l -> Tensor rank shape value
(!!!!) m _ {l = Z}   impossible
(!!!!) (Vector v) i = indexV i v
  where indexV : Fin len -> Vect len elem -> elem
        indexV = index

-- "Plumbing" utilities, very useful
-- Generalizing the idea that "a vector of vectors is a matrix"
flatten : Tensor rank1 shape1 (Tensor rank2 shape2 value) ->
          Tensor (rank1 + rank2) (shape1 ++ shape2) value
flatten (Scalar m) = m
flatten (Vector v) = Vector $ map flatten v

-- Generalizing the idea that "a matrix is a vector of vectors"
bend : Tensor (rank1 + rank2) (shape1 ++ shape2) value ->
       Tensor rank1 shape1 (Tensor rank2 shape2 value)
bend {shape1 = []}          m          = Scalar m
bend {shape1 = l :: shape1} (Vector v) = Vector $ map bend v

-- User-level Vect <-> Tensor conversion:
-- The input type for the converter
fromVectType : (rank : Nat) -> Vect rank Nat -> Type -> Type
fromVectType  Z             []     value = value
fromVectType (S rank) (l :: shape) value = Vect l $ fromVectType rank shape value

-- The type of the third argument depends on the value of the first two arguments!
-- (this is what the end user will be using to construct tensors)
fromVect : (rank : Nat) -> (shape : Vect rank Nat) ->
           fromVectType rank shape value -> Tensor rank shape value
fromVect  Z             []     x = Scalar x
fromVect (S rank) (l :: shape) v = Vector $ map (fromVect rank shape) v

-- Function type conversion, useful for transposition:
-- Generates the function type, taking as many parameters as the rank of the tensor
-- (out-of-bounds is checked at the type level with Fin)
functionType : (rank : Nat) -> Vect rank Nat -> Type -> Type
functionType Z        []           value =  value
functionType (S rank) (l :: shape) value =
  Fin l -> functionType rank shape value

-- Tensor -> function
toFunction : Tensor rank shape value ->
             functionType rank shape value
toFunction {shape = []}         (Scalar x) = x
toFunction {shape = l :: shape} m          = \i => toFunction (m !!!! i)

-- Tensor -> function returning (smaller) tensors
toFunctionPart : Tensor (rank1 + rank2) (shape1 ++ shape2) value ->
                 functionType rank1 shape1 (Tensor rank2 shape2 value)
toFunctionPart {shape1} {shape2} m =
  toFunction {shape = shape1} . bend {shape1} {shape2} $ m

-- functon -> Tensor
fromFunction : functionType rank shape value -> Tensor rank shape value
fromFunction {shape = []}         x = Scalar x
fromFunction {shape = l :: shape} f = Vector . map (fromFunction {shape} . f) $ range l
  where range : (n : Nat) -> Vect n (Fin n)
        range Z     = []
        range (S n) = FZ :: (map FS $ range n)

-- function of separated arguments -> Tensor
fromFunctionPart : functionType (rank1 + rank2) (shape1 ++ shape2) value ->
                   functionType rank1 shape1 (Tensor rank2 shape2 value)
fromFunctionPart {shape1 = []}          f = fromFunction f
fromFunctionPart {shape1 = l :: shape1} f = \i => fromFunctionPart {shape1} (f i)

-- Function type conversion, but with single (HVect) argument (type-level out-of-bounds=
-- Type generation for the HVect containing the arguments
indexType : Vect rank Nat -> Vect rank Type
indexType [] = []
indexType (n :: shape) = Fin n :: indexType shape

-- Index accessors, "full indices"
infixl 0 #!
(#!) : Tensor rank shape value ->
       HVect (indexType shape) ->
       value
(#!) {shape = []}           (Scalar x)       []       = x
(#!) {shape = Z :: shape}           t  (i :: indices) impossible 
(#!) {shape = S l :: shape}         t  (i :: indices) = (#!) {shape} (t !!!! i) indices

fromVectTens : (HVect (indexType shape) -> v) ->
               Tensor rank shape v
fromVectTens {shape} f =
  fromFunction {shape} $ fromVectFun {s = shape} f
  where fromVectFun : (HVect (indexType s) -> v) -> functionType r s v
        fromVectFun {s = []}     f = f []
        fromVectFun {s = l :: s} f = \i => fromVectFun {s} $ f . (i ::)

-- Index equality, useful for the Kronecker delta
eq_index : HVect (indexType shape) -> HVect (indexType shape) -> Bool
eq_index {shape = []}         []        []        = True
eq_index {shape = l :: shape} (i :: is) (j :: js) =
  (i == j) && (eq_index is js)

-- Various products, the essence of this library
-- Tensor product: at position `i` we find the tensor `t[i] * s`
-- (where `*` denotes scalar product)
tensorProduct : Num value =>
                Tensor rank1 shape1 value ->
                Tensor rank2 shape2 value ->
                Tensor rank1 shape1 (Tensor rank2 shape2 value)
tensorProduct t s = map (flip (|*>) s) t

-- Generalization of matrix multiplications (rows-by-columns) to tensors: index contraction.
contractProduct : Num value =>
                  Tensor (rank1 + rank2) (shape1 ++ shape2) value ->
                  Tensor (rank2 + rank3) (shape2 ++ shape3) value ->
                  Tensor (rank1 + rank3) (shape1 ++ shape3) value
contractProduct {shape1 = []} {shape2} t          s =
  foldrT (+) 0 . liftA2 (|*>) t $ bend {shape1 = shape2} s
contractProduct {shape1 = l :: shape1} (Vector v) s =
  Vector $ map (flip (contractProduct {shape1}) s) v

-- Specialized versions of contractProduct
infixl 0 |#|
(|#|) : Num value =>
        Tensor (rank1 + rank2) (shape1 ++ shape2) value ->
        Tensor (rank2 + rank3) (shape2 ++ shape3) value ->
        Tensor (rank1 + rank3) (shape1 ++ shape3) value
(|#|) = contractProduct

infixl 0 |#>
(|#>) : Num value =>
        Tensor (rank1 + rank2) (shape1 ++ shape2) value ->
        Tensor rank2 shape2 value ->
        Tensor rank1 shape1 value
(|#>) {shape1} {shape2} t s =
  let s' = flatten . map Scalar $ s
      r = (|#|) {shape1} {shape2} {shape3 = []} t s'
  in map unScalar . bend {shape1} {shape2 = []} $ r

infixl 0 <#|
(<#|) : Num value =>
        Tensor rank2 shape2 value ->
        Tensor (rank2 + rank3) (shape2 ++ shape3) value ->
        Tensor rank3 shape3 value
(<#|) {shape2} {shape3} t s =
  let t' = flatten . Scalar $ t
      r = (|#|) {shape1 = []} {shape2} {shape3} t' s
  in unScalar . bend {shape1 = []} {shape2 = shape3} $ r

infixl 0 <#>
(<#>) : Num value =>
        Tensor rank2 shape2 value ->
        Tensor rank2 shape2 value ->
        value
(<#>) {shape2} t s =
  let t' = flatten . Scalar $ t
      s' = flatten . map Scalar $ s
      r = (|#|) {shape1 = []} {shape2} {shape3 = []} t' s'
  in unScalar r

-- To be read "bottom up", since it's a long composition
transposeT : Tensor (rank1 + rank2) (shape1 ++ shape2) value ->
             Tensor (rank2 + rank1) (shape2 ++ shape1) value
transposeT {shape1} {shape2} t =
  flatten {shape1 = shape2} {shape2 = shape1} -- "join" the two ranks
  . map (fromVectTens {shape = shape1})       -- inner "toTensor"
  . fromVectTens {shape = shape2}             -- outer "toTensor"
  . flip                                      -- flip arguments
  . (#!) {shape = shape1}                     -- outer "fromTensor"
  . map ((#!) {shape = shape2})               -- inner "fromTensor"
  . bend {shape1} {shape2}                    -- "split" the two ranks
  $ t

{-
  -- This is to be read "top down" instead, since all compositions
  -- have been replaced with let-bindings (otherwise, it's equivalent)
  let t1 = bend {shape1} {shape2}                t
      tf = map (toVectFun {shape = shape2})      t1
      ff = toVectFun {shape = shape1}            tf
      gg = flip                                  ff
      tg = fromVectTens {shape = shape2}         gg
      t2 = map (fromVectTens {shape = shape1})   tg
  in flatten {shape1 = shape2} {shape2 = shape1} t2
-}
