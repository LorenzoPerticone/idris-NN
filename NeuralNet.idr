module NeuralNet2

import Data.Fin
import Data.Vect
import Data.HVect

import Tensor
import Layer
import Floating

%access public export
%default total

-- Utilities

crossE : Floating value =>
         Tensor rank shape value ->
         Tensor rank shape value ->
         value
crossE tgt out = - (f_log $ tgt <#> out)

crossE_g : Floating value =>
           Tensor rank shape value ->
           Tensor rank shape value ->
           Tensor rank shape value
crossE_g tgt out = 
  let t = tgt <#> out
      f = map (/ t)
  in f tgt

logistic_activ : Floating value =>
                 value -> value
logistic_activ x = 1 / (1 + f_exp (-x))

logistic : Floating value =>
                 Tensor rank shape value ->
                 Tensor rank shape value
logistic = map logistic_activ

logistic_g : Floating value =>
             Tensor rank shape value ->
             Tensor (rank + rank) (shape ++ shape) value
logistic_g t =
  let f = \is => \js => if eq_index is js
                        then let t_is = t #! is
                             in logistic_activ (-t_is) * logistic_activ t_is
                        else 0
  in flatten . map fromVectTens . fromVectTens $ f

relu : (Ord value, Floating value) =>
             Tensor rank shape value ->
             Tensor rank shape value
relu = map (max 0)

relu_g : (Ord value, Floating value) =>
         Tensor rank shape value ->
         Tensor (rank + rank) (shape ++ shape) value
relu_g {shape} {value} t =
  let f = \is => \js => if eq_index is js
                        then if (t #! is) > 0
                             then the value (fromInteger 1)
                             else the value (fromInteger 0)
                        else 0
  in flatten . map fromVectTens . fromVectTens $ f

softmax : Floating value =>
                Tensor rank shape value ->
                Tensor rank shape value
softmax {value} t =
  let s = map f_exp t
      x = (s <#> pure 1)
      y = map (/ x) s
  in y

softmax_g : Floating value =>
                  Tensor rank shape value ->
                  Tensor (rank + rank) (shape ++ shape) value
softmax_g t = 
  let st = softmax t
      f = \is => \js => if eq_index is js
                        then let xi = st #! is
                             in xi * (1 - xi)
                        else let xi = st #! is
                                 xj = st #! js
                             in - (xi * xj)
  in flatten . map fromVectTens . fromVectTens $ f

-- Neural Networks

typeVect : (depth : Nat) -> Vect depth Nat -> Vect depth Type
typeVect Z [] = []
typeVect (S k) (x :: xs) = Vect x Nat :: typeVect k xs

data NeuralNet : (depth : Nat) ->
                 (ranks : Vect (S depth) Nat) ->
                 (shapes : HVect (typeVect (S depth) ranks)) ->
                 (value : Type) ->
                 Type where
  LstLayer : (Tensor rank shape value -> Tensor rank shape value -> value) ->
             (Tensor rank shape value -> Tensor rank shape value -> Tensor rank shape value) ->
             NeuralNet Z [rank] [shape] value
  AddLayer : NeuralNet depth (o_rank :: ranks) (o_shape :: shapes) value ->
             Layer i_rank o_rank i_shape o_shape value ->
             NeuralNet (S depth) (i_rank :: o_rank :: ranks) (i_shape :: o_shape :: shapes) value

sLast : {depth : Nat} -> {ranks : Vect (S depth) Nat} ->
        HVect (typeVect (S depth) ranks) -> Vect (last ranks) Nat
sLast {depth = Z} {ranks = r :: []} (sh :: []) = sh
sLast {depth = S depth} {ranks = r1 :: r2 :: ranks} (sh1 :: sh2 :: shapes) =
  sLast {depth} {ranks = r2 :: ranks} (sh2 :: shapes)

outputTypeNN : (shapes : HVect (typeVect (S depth) (ranks))) ->
               Type -> Type
outputTypeNN {depth} {ranks = r :: ranks} (sh :: shapes) value =
  let shape = sLast {ranks = r :: ranks} (sh :: shapes)
  in Tensor (last (r :: ranks)) shape value

inputTypeNN : (shapes : HVect (typeVect (S depth) ranks)) ->
              Type -> Type
inputTypeNN {depth} {ranks = r :: rs} (s :: shs) value  = Tensor r s value


fromLayersType : (depth : Nat) ->
                 (ranks : Vect (S depth) Nat) ->
                 (shapes : HVect (typeVect (S depth) ranks)) ->
                 Type ->
                 Vect (S depth) Type
fromLayersType Z     (r1 :: [])       (s1 :: []) value =
  let l = HVect [Tensor r1 s1 value -> Tensor r1 s1 value -> value,
                 Tensor r1 s1 value -> Tensor r1 s1 value -> Tensor r1 s1 value]
  in l :: []
fromLayersType (S d) (r1 :: r2 :: rs) (sh1 :: sh2 :: shs) value = 
  let l = Layer r1 r2 sh1 sh2 value
      rest = fromLayersType d (r2 :: rs) (sh2 :: shs) value
  in l :: rest

fromLayers : HVect (fromLayersType depth ranks shapes value) -> 
             NeuralNet depth ranks shapes value
fromLayers {depth = Z}       {ranks = r :: []}           {shapes = s :: []}            (cpl :: []) =
  let (f :: g :: []) = cpl
  in LstLayer f g
fromLayers {depth = S depth} {ranks = r1 :: r2 :: ranks} {shapes = s1 :: s2 :: shapes} (l :: ls)   =
  let n = fromLayers ls
  in AddLayer n l

implementation Show value => Show (NeuralNet depth ranks shapes value) where
  show net = helper 1 net
    where helper : Show v => Nat -> NeuralNet d rs shs v -> String
          helper k (LstLayer _ _) = ""
          helper k (AddLayer n (MkLayer w b _ _)) =
            let header = "layer" ++ show k ++ ":\n"
                weigth = "weigths:\n" ++ show w ++ "\n"
                biases = "biases:\n" ++ show b ++ "\n"
            in header ++ weigth ++ biases ++ helper (S k) n

-- Only here to register instances for Num, Neg, Functor and Applicative.
-- These aren't technically correct instances, but as long as
-- `fromInteger`'d and `pure`'d values are manipulated correctly,
-- everything should be fine.
-- Never, ever try to evaluate `bad_metric` on a tensor whose
-- shape contains a zero. Better yet, never evaluate `bad_metric`
-- at all.
bad_metric : Tensor rank shape value ->
             Tensor rank shape value ->
             value
bad_metric (Scalar x)         (Scalar y)         = y
bad_metric (Vector [])        (Vector [])        = believe_me ()
bad_metric (Vector (x :: xs)) (Vector (y :: ys)) = bad_metric x y

bad_metric_g : Tensor rank shape value ->
               Tensor rank shape value ->
               Tensor rank shape value
bad_metric_g t1 t2 = t2

implementation Num value => Num (NeuralNet depth ranks shapes value) where
  (LstLayer _  _ ) + (LstLayer f  g ) = LstLayer f g
  (AddLayer n1 l1) + (AddLayer n2 l2) = AddLayer (n1 + n2) (l1 + l2)
  
  (LstLayer _  _ ) * (LstLayer f  g ) = LstLayer f g
  (AddLayer n1 l1) * (AddLayer n2 l2) = AddLayer (n1 * n2) (l1 * l2)
  
  fromInteger {depth = Z}       {ranks = r :: []}        {shapes = sh :: []}          n =
    LstLayer bad_metric bad_metric_g
  fromInteger {depth = S depth} {ranks = r1 :: r2 :: rs} {shapes = sh1 :: sh2 :: shs} n =
    AddLayer (fromInteger n) (fromInteger n)

implementation Neg value => Neg (NeuralNet depth ranks shapes value) where
  negate (LstLayer f g) = LstLayer f g
  negate (AddLayer n l) = AddLayer (negate n) (negate l)
  
  (LstLayer _  _ ) - (LstLayer f  g ) = LstLayer f g
  (AddLayer n1 l1) - (AddLayer n2 l2) = AddLayer (n1 - n2) (l1 - l2)

implementation Functor (NeuralNet depth ranks shapes) where
  map f (LstLayer _ _) = LstLayer bad_metric bad_metric_g
  map f (AddLayer n l) = AddLayer (map f n) (map f l)

implementation Applicative (NeuralNet depth ranks shapes) where
  pure {depth = Z}       {ranks = r :: []}        {shapes = sh :: []}          x =
    LstLayer bad_metric bad_metric_g
  pure {depth = S depth} {ranks = r1 :: r2 :: rs} {shapes = sh1 :: sh2 :: shs} x =
    AddLayer (pure x) (pure x)

  (LstLayer _  _ ) <*> (LstLayer _  _ ) = LstLayer bad_metric bad_metric_g
  (AddLayer n1 l1) <*> (AddLayer n2 l2) = AddLayer (n1 <*> n2) (l1 <*> l2)

-- Execution utilities
runNet : Num value =>
         NeuralNet depth ranks shapes value ->
         inputTypeNN shapes value ->
         outputTypeNN shapes value
runNet (LstLayer _ _) i = i
runNet (AddLayer n l) i = runNet n (runLayer l i)

runNetBatch : Num value =>
              NeuralNet depth ranks shapes value ->
              Tensor b_rank b_shape (inputTypeNN shapes value) ->
              Tensor b_rank b_shape (outputTypeNN shapes value)
runNetBatch net is = map (runNet net) is

runNet_bp : Floating value =>
            NeuralNet depth ranks shapes value ->
            inputTypeNN shapes value ->         -- <- input
            outputTypeNN shapes value ->        -- <- target
            (value,                             -- <- score
             NeuralNet depth ranks shapes value,-- <- Net of corrections
             inputTypeNN shapes value)          -- <- gradient wrt `input`
runNet_bp (LstLayer f g) i t = 
  let score       = f t i
      corrections = LstLayer f g
      out_grad    = g t i
  in (score, corrections, out_grad)
runNet_bp (AddLayer n l) i t = 
  let (score, cnet, in_grad) = runNet_bp n (runLayer l i) t
      (out_grad, cl) = runLayer_backprop l i in_grad
  in (score, AddLayer cnet cl , out_grad)

runNetBatch_bp : Floating value =>
                 NeuralNet depth ranks shapes value ->
                 Tensor b_rank b_shape (inputTypeNN shapes value) ->
                 Tensor b_rank b_shape (outputTypeNN shapes value) ->
                 (value,                                             -- <- average score
                  NeuralNet depth ranks shapes value,                -- <- Net of corrections
                  Tensor b_rank b_shape (inputTypeNN shapes value))  -- <- gradients wrt `input`
runNetBatch_bp {b_shape} {value} n is ts =
  let (scores, nets, grads) = unzip3 $ map (runNet_bp n) is <*> ts
      num = the value . fromInteger . toIntegerNat $ foldl (*) 1 b_shape
      net = map (/ num) $ foldlT (+) (pure 0) nets
      score = (foldlT (+) 0 scores) / num
  in (score, net, grads)
