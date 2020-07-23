module Layer

import Data.Fin
import Data.Vect
import Data.HVect

import Tensor

%access public export
%default total

data Layer : (i_rank : Nat) ->
             (o_rank : Nat) ->
             (i_shape : Vect i_rank Nat) ->
             (o_shape : Vect o_rank Nat) ->
  (value : Type) -> Type where
  MkLayer : (w : Tensor (o_rank + i_rank) (o_shape ++ i_shape) value) ->
            (b : Tensor o_rank o_shape value) ->
            (f : Tensor o_rank o_shape value ->
                 Tensor o_rank o_shape value) ->
            (g : Tensor o_rank o_shape value ->
                 Tensor (o_rank + o_rank) (o_shape ++ o_shape) value) ->
            Layer i_rank o_rank i_shape o_shape value

implementation Eq value => Eq (Layer i_rank o_rank i_shape o_shape
                                     value) where

  (MkLayer w1 b1 _ _) == (MkLayer w2 b2 _ _) =
    (w1 == w2) && (b1 == b2)

implementation Show value => Show (Layer i_rank o_rank i_shape o_shape
                                         value) where
  show (MkLayer w b _ _) = show w ++ "\n\n" ++ show b

-- For every operation it keeps the rhs in (n1 + n2) wrt
-- activation `f`unction and it's generalized `g`radient,
-- so I strongly advocate on using `foldrT (+)  results n`
-- or `foldrT (*) results n` for whatever function relative
-- to training, where `n` is the network you're training,
-- if using a tensor as a batch: it comes pre-equipped with
-- operations being lifted pointwise to every scalar
-- contained by the network.
activation_def : Tensor rank shape value -> Tensor rank shape value
activation_def = id
activation_g_def : Tensor rank shape value ->
                   Tensor (rank + rank) (shape ++ shape) value
activation_g_def = flatten . pure . id

-- fromInteger defaults to `activation_def` and `activation_g_def`
implementation Num value => Num (Layer i_rank o_rank i_shape o_shape
                                       value) where
  (MkLayer w b _ _) + (MkLayer w' b' f g) = 
    MkLayer (w + w') (b + b') f g
  
  (MkLayer w b _ _) * (MkLayer w' b' f g) = 
    MkLayer (w * w') (b * b') f g
    
  fromInteger x = 
    let w = pure $ fromInteger x
        b = pure $ fromInteger x
    in MkLayer w b activation_def activation_g_def
    
implementation Neg value => Neg (Layer i_rank o_rank i_shape o_shape
                                       value) where
  negate (MkLayer w b f g) = MkLayer (negate w) (negate b) f g
  (MkLayer w1 b1 _ _) - (MkLayer w2 b2 f g) =
    MkLayer (w1 - w2) (b1 - b2) f g

-- defaults to `activation_def` and `activation_g_def`
implementation Functor (Layer i_rank o_rank i_shape o_shape) where
  map f (MkLayer w b _ _) = 
    let w' = map f w
        b' = map f b
    in MkLayer w'  b' activation_def activation_g_def

-- defaults to `activation_def` and `activation_g_def`
implementation Applicative (Layer i_rank o_rank i_shape o_shape) where
  pure x =
    let w = pure x
        b = pure x
    in MkLayer w b activation_def activation_g_def

  (MkLayer w1 b1 _ _) <*> (MkLayer w2 b2 _ _) =
    MkLayer (w1 <*> w2) (b1 <*> b2) activation_def activation_g_def

-- Execution utilities
runLayer : Num value =>
           Layer i_rank o_rank i_shape o_shape value ->
           Tensor i_rank i_shape value ->
           Tensor o_rank o_shape value
runLayer (MkLayer w b f _) input =
  f $ (w |#> input) + b

runLayerBatch : Num value =>
                Layer i_rank o_rank i_shape o_shape value ->
                Tensor b_rank b_shape (Tensor i_rank i_shape value) ->
                Tensor b_rank b_shape (Tensor o_rank o_shape value)
runLayerBatch l inputs =
  map (runLayer l) inputs

-- Utilities
splitHVect : HVect (indexType (shape1 ++ shape2)) ->
             (HVect (indexType shape1),
              HVect (indexType shape2))
splitHVect {shape1 = []}        xs        = ([], xs)
splitHVect {shape1 = s::shape1} (v :: vs) =
  let (v1, v2) = splitHVect vs
  in (v :: v1, v2)

delta : Num value => Tensor (rank + rank) (shape ++ shape) value
delta = fromVectTens $ boolToDouble . (uncurry eq_index) . splitHVect
  where boolToDouble : Num value => Bool -> value
        boolToDouble True = 1
        boolToDouble False = 0

-- Backpropagation
runLayer_backprop : Num value =>
                    Layer i_rank o_rank i_shape o_shape value ->
                    Tensor i_rank i_shape value ->
                    Tensor o_rank o_shape value ->
                    (Tensor i_rank i_shape value,
                     Layer i_rank o_rank i_shape o_shape value)
runLayer_backprop (MkLayer w b f g) input in_grad =
  let affine = (w |#> input) + b
      f_grad = g affine
      b_grad = fromVectTens $ helper input
      w' = in_grad <#| (f_grad |#| b_grad)
      b' = in_grad <#| (f_grad |#| delta)
      bp = in_grad <#| (f_grad |#| w)
      l = MkLayer w' b' f g
  in (bp, l)
  where helper : Num value => Tensor i_rank i_shape value ->
                HVect (indexType (o_shape ++ o_shape ++ i_shape)) ->
                value
        helper input vs = let (v1, vs') = splitHVect vs
                              (v2, v3)  = splitHVect vs'
                          in if eq_index v1 v2
                             then input #! v3
                             else 0
