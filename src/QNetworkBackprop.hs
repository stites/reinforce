{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PatternSynonyms #-}
module QNetworkBackprop where

import Zoo.Prelude
import Control.MonadEnv.Internal
import Environments.Gym.ToyText.FrozenLakeV0 hiding (Left, Right)

import qualified System.Random.MWC       as MWC (createSystemRandom)
import qualified Control.MonadMWCRandom  as MWC
import qualified Control.MonadEnv        as Env
import qualified Data.Logger             as Logger
import qualified Data.DList              as DL
import qualified Data.Vector             as V
import qualified Numeric.Backprop        as BP
import qualified Numeric.LinearAlgebra   as LA
import qualified Numeric.LinearAlgebra.Static  as LA
import qualified Generics.SOP            as SOP

type Event = Logger.Event Reward StateFL Action

type KnownNat2 a b   = (KnownNat a, KnownNat b)
type KnownNat3 a b c = (KnownNat a, KnownNat b, KnownNat c)

instance SOP.Generic (R o)

(&<) :: f a -> f b -> Prod f '[a, b]
a &< b = a :< BP.only b
infix 1 &<

(&&<) :: a -> b -> Prod I '[a, b]
a &&< b = a ::< b ::< _z
infix 1 &&<


-- =========================================== --
-- describe a NN
-- =========================================== --

data Layer i o =
    Layer { _lWeights :: !(L o i)
          , _lBiases  :: !(R o)
          }
  deriving (Show, Generic)

instance SOP.Generic (Layer i o)
instance NFData (Layer i o)

data Network i h o =
    Net { _nLayer1 :: !(Layer i h)
        , _nLayer2 :: !(Layer h o)
        }
  deriving (Show, Generic)

instance SOP.Generic (Network i h o)
instance NFData (Network i h o)

instance (KnownNat2 i o) => Num (Layer i o) where
    Layer w1 b1 + Layer w2 b2 = Layer (w1 + w2) (b1 + b2)
    Layer w1 b1 - Layer w2 b2 = Layer (w1 - w2) (b1 - b2)
    Layer w1 b1 * Layer w2 b2 = Layer (w1 * w2) (b1 * b2)
    abs    (Layer w b)        = Layer (abs    w) (abs    b)
    signum (Layer w b)        = Layer (signum w) (signum b)
    negate (Layer w b)        = Layer (negate w) (negate b)
    fromInteger x             = Layer (fromInteger x) (fromInteger x)

instance (KnownNat3 i h o) => Num (Network i h o) where
    Net a b + Net c d = Net (a + c) (b + d)
    Net a b - Net c d = Net (a - c) (b - d)
    Net a b * Net c d = Net (a * c) (b * d)
    abs    (Net a b)  = Net (abs    a) (abs    b)
    signum (Net a b)  = Net (signum a) (signum b)
    negate (Net a b)  = Net (negate a) (negate b)
    fromInteger x     = Net (fromInteger x) (fromInteger x)

instance (KnownNat2 i o) => Fractional (Layer i o) where
    Layer w1 b1 / Layer w2 b2 = Layer (w1 / w2) (b1 / b2)
    recip (Layer w b)         = Layer (recip w) (recip b)
    fromRational x            = Layer (fromRational x) (fromRational x)

instance (KnownNat3 i h o) => Fractional (Network i h o) where
    Net a b / Net c d = Net (a / c) (b / d)
    recip (Net a b)   = Net (recip a) (recip b)
    fromRational x    = Net (fromRational x) (fromRational x)


-------------------------------------------------------------------------------
-- Define backprop primatives
-------------------------------------------------------------------------------

-- matrix-vector multiplication primitive, giving an explicit gradient function.
matVec :: forall m n . (KnownNat2 m n) => Op '[ L m n, R n ] (R m)
matVec = BP.op2' $ \m v ->
  let
    back :: Maybe (R m) -> (L m n, R n)
    back (fromMaybe 1 -> g) = (g `outer` v, tr m #> g)
  in
    (m #> v, back)

-- dot products
dot :: forall n . KnownNat n => Op '[ R n, R n ] Double
dot = BP.op2' $ \x y ->
    let
      back :: Maybe Double -> (R n, R n)
      back = \case Nothing -> (y, x)
                   Just g  -> (konst g * y, x * konst g)
    in
      (x <.> y, back)

-- dot products
squaredSum :: forall n . KnownNat n => Op '[ R n ] Double
squaredSum = BP.op1' $ \x ->
    let
      back :: Maybe Double -> R n
      back = \case Nothing -> x
                   Just g  -> x * konst g
    in
      (x <.> x, back)


vsum :: forall n . KnownNat n => Op '[ R n ] Double
vsum = BP.op1' $ \x -> (LA.sumElements (LA.extract x), maybe 1 konst)

scale :: forall n . KnownNat n => Op '[ Double, R n ] (R n)
scale = BP.op2' $ \a x ->
  let
    back :: Maybe (R n) -> (Double, R n)
    back = \case Nothing -> (LA.sumElements (LA.extract      x ), konst a    )
                 Just g  -> (LA.sumElements (LA.extract (x * g)), konst a * g)
  in
    (konst a * x, back)

vkonst :: forall n . KnownNat n => Op '[ R n ] (R n)
vkonst = BP.op1' $ \x -> (x, \_ -> x)


-------------------------------------------------------------------------------
-- describe how to run a neural network
-------------------------------------------------------------------------------

runLayer :: (KnownNat2 i o) => BPOp s '[ R i, Layer i o ] (R o)
runLayer = BP.withInps $ \(x :< l :< _) -> do
    w :< b :< _ <- gTuple #<~ l
    y <- matVec ~$ (w :< only x)
    return $ y + b


runNetwork :: (KnownNat3 i h o) => BPOp s '[ R i, Network i h o ] (R o)
runNetwork = BP.withInps $ \(inp :< net :< _) -> do
    l1 :< l2 :< _ <- gTuple #<~ net
    y <- runLayer -$ (inp        :< only l1)
    z <- runLayer -$ (logistic y :< only l2)
    softmax       -$ (only z               )
  where
    -- for layer activation
    logistic :: Floating a => a -> a
    logistic x = 1 / (1 + exp (-x))

    softmax :: KnownNat n => BPOp s '[ R n ] (R n)
    softmax = BP.withInps $ \(x :< _) -> do
      expX <- BP.bindVar (exp x)
      totX <- vsum ~$ only expX
      scale ~$ (1 / totX :< only expX)


-------------------------------------------------------------------------------
-- create our inference and gradient functions
-------------------------------------------------------------------------------
_z :: Prod f '[]
_z = Ø


runNetOnInputs :: KnownNat3 i h o => R i -> Network i h o -> R o
runNetOnInputs input net = BP.evalBPOp runNetwork (input ::< net ::< _z)


getGradient :: KnownNat3 i h o => R i -> Network i h o -> Network i h o
getGradient input net =
  case BP.gradBPOp runNetwork (input ::< net ::< _z) of
     (gradInpts ::< gradNet ::< _) -> gradNet


loss :: forall s n .  (KnownNat n) => R n -> BPOp s '[ R n ] Double
loss target = BP.withInps $ \(out :< _) ->
  let
    err :: BVar s '[R n] (R n)
    err = BP.constVar target - out
  in
    dot ~$ (err &< err)

train :: forall i h o . KnownNat3 i h o => Double -> R i -> R o -> Network i h o -> Network i h o
train gamma xs trueRwds net =
  case BP.gradBPOp op (xs &&< net) of
    _ ::< gradNet ::< _ -> net - (realToFrac gamma * gradNet)

  where
    op :: BPOp s '[ R i, Network i h o ] Double
    op = do
      pred <- runNetwork
      loss trueRwds -$ only pred


trainList :: forall i h o . KnownNat3 i h o => Double -> [(R i, R o)] -> Network i h o -> Network i h o
trainList gamma observed net = foldl' (\n (x, y) -> train gamma x y n) net observed



