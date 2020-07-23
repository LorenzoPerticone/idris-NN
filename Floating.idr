module Floating

%access public export
%default total

infixl 0 ^

interface (Num a, Neg a, Abs a, Fractional a) => Floating a where
  (^)    : a -> a -> a
  f_sqrt : a -> a
  
  f_acos : a -> a
  f_asin : a -> a
  f_atan : a -> a
  
  f_cos  : a -> a
  f_sin  : a -> a
  f_tan  : a -> a
  
  f_cosh : a -> a
  f_sinh : a -> a
  f_tanh : a -> a
  
  f_exp  : a -> a
  f_log  : a -> a

implementation Floating Double where
  (^)    = pow
  f_sqrt = sqrt
  
  f_acos = acos
  f_asin = asin
  f_atan = atan
  
  f_cos  = cos
  f_sin  = sin
  f_tan  = tan
  
  f_cosh = cosh
  f_sinh = sinh
  f_tanh = tanh
  
  f_exp  = exp
  f_log  = log
