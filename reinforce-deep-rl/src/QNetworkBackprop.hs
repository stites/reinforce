{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
module QNetworkBackprop where

import Zoo.Prelude
import Control.MonadEnv.Internal
import Environments.Gym.ToyText.FrozenLakeV0 hiding (Left, Right)
import System.Random.MWC (Variate(..))

import qualified System.Random.MWC       as MWC (withSystemRandom, createSystemRandom)
import qualified System.Random.MWC.Distributions as MWC (uniformShuffle)
import qualified Control.MonadMWCRandom  as MWC
import qualified Control.MonadEnv        as Env
import qualified Data.Logger             as Logger
import qualified Data.DList              as DL
import qualified Data.Vector             as V
import qualified Numeric.Backprop        as BP
import qualified Numeric.LinearAlgebra   as LA (sumElements)
import qualified Numeric.LinearAlgebra.Static  as LA
import qualified Generics.SOP            as SOP
import qualified GHC.Exts            as Exts

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

_z :: Prod f '[]
_z = Ø

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

instance KnownNat n => MWC.Variate (R n) where
  uniform g = LA.randomVector <$> MWC._uniform g <*> pure LA.Uniform
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC._uniform g

instance (KnownNat2 m n) => MWC.Variate (L m n) where
  uniform g = LA.uniformSample <$> MWC._uniform g <*> pure 0 <*> pure 1
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC._uniform g

instance (KnownNat2 i o) => MWC.Variate (Layer i o) where
  uniform g = Layer <$> MWC._uniform g <*> MWC._uniform g
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC._uniform g

instance (KnownNat3 i h o) => MWC.Variate (Network i h o) where
  uniform g = Net <$> MWC._uniform g <*> MWC._uniform g
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC._uniform g

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
    y <- matVec ~$ (w &< x)
    return $ y + b


runNetwork :: (KnownNat3 i h o) => BPOp s '[ R i, Network i h o ] (R o)
runNetwork = BP.withInps $ \(inp :< net :< _) -> do
    l1 :< l2 :< _ <- gTuple #<~ net
    y <- runLayer -$ (inp        &< l1)
    z <- runLayer -$ (logistic y &< l2)
    softmax       -$ (only z          )
  where
    -- for layer activation
    logistic :: Floating a => a -> a
    logistic x = 1 / (1 + exp (-x))

    softmax :: KnownNat n => BPOp s '[ R n ] (R n)
    softmax = BP.withInps $ \(x :< _) -> do
      expX <- BP.bindVar (exp x)
      totX <- vsum ~$ only expX
      scale ~$ (1 / totX &< expX)


-------------------------------------------------------------------------------
-- create our inference and gradient functions
-------------------------------------------------------------------------------

runNetOnInputs :: KnownNat3 i h o => R i -> Network i h o -> R o
runNetOnInputs input net = BP.evalBPOp runNetwork (input &&< net)


getGradient :: KnownNat3 i h o => R i -> Network i h o -> Network i h o
getGradient input net =
  case BP.gradBPOp runNetwork (input &&< net) of
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

-------------------------------------------------------------------------------
-- run on environment
-------------------------------------------------------------------------------

maxEpisodes :: Int
maxEpisodes = 3000

maxSteps :: Int
maxSteps = 99

gamma :: Double
gamma = 0.99

type FLNetwork = Network 16 4 4

main :: IO ()
main =
  MWC.createSystemRandom
  >>= MWC.runMWCRandT (runDefaultEnvironmentT False learn)
  >>= \case
    Left err                       -> throwString (show err)
    Right (histToRewards -> rlist) -> report maxEpisodes rlist

  where
    histToRewards :: DList Event -> [Reward]
    histToRewards (DL.toList->rs) = fmap sum episodes
      where
        episodes :: [[Reward]]
        episodes = (fmap.fmap) snd $
          groupBy (\l r -> fst l == fst r) $
            fmap (\(Logger.Event i r _ _) -> (i, r)) rs


report :: MonadIO io => Int -> [Reward] -> io ()
report epn rwds = do
  let s = sum rwds :: Double
  let l = fromIntegral (genericLength rwds) :: Double
  let last50Per = sum (lastN 50 rwds) / 50 :: Double
  printIO $ show epn ++ " - Percent successful episodes: " ++ show (s / l)   ++ "%"
  printIO $ show epn ++ " - Percent successful last 50 : " ++ show last50Per ++ "%"
  printIO $ show epn ++ " - Total successful episodes  : " ++ show s
  printIO $ show epn ++ " - Total rewards recorded     : " ++ show l


learn :: EnvironmentT (MWCRandT IO) (DList (DList (Reward, Action)))
learn = do
  (net0 :: FLNetwork) <- lift $ MWC.uniformR (-0.5, 0.5)
  snd . snd <$> runStateT (forM_ [1..] runEpisode) (net0, mempty)
  where
    runEpisode :: Int -> StateT (FLNetwork, DList (DList (Reward, Action))) (EnvironmentT (MWCRandT IO)) ()
    runEpisode epNum = do
      (net, hist) <- getState
      obs <- lift Env.reset
      case obs of
        EmptyEpisode -> return ()
        Initial s    -> do
          (net, rs) <- lift $ rolloutEpisode epNum net 0 (decayEpsilon epNum) s DL.empty
          when (epNum `divisibleBy` 100) $ report epNum (fst <$> DL.toList rs)
          putState (net, hist `DL.snoc` rs)


decayEpsilon :: Int -> Float
decayEpsilon epNum = 1.0 / ((epNum // 50) + 10)


epsilonGreedyChoice :: Action -> Float -> EnvironmentT (MWCRandT IO) Action
epsilonGreedyChoice a e =
  lift (MWC.uniformR (0, 1))
  >>= \p->
    if p < e
    then randomAction
    else pure a


randomAction :: EnvironmentT (MWCRandT IO) Action
randomAction = toEnum <$> (lift $ MWC.uniformR (0, fromEnum (maxBound :: Action)))


rolloutEpisode
  :: Int
  -> FLNetwork
  -> Int
  -> Float
  -> StateFL
  -> DList (Reward, Action)
  -> EnvironmentT (MWCRandT IO) (FLNetwork, DList (Reward, Action))
rolloutEpisode episodeNum model stepNum epsilon state dl
  | stepNum > maxSteps = return (model, dl)
  | otherwise = do
    let
      qs :: R 4
      a' :: Action
      (qs, a') = chooseAction model state

    a        <- epsilonGreedyChoice a' epsilon
    next     <- Env.step a
    case next of
      Next rwd nextState -> do
        newNet <- liftIO $ updateAgent state a rwd nextState qs model
        rolloutEpisode episodeNum newNet (stepNum+1) epsilon nextState (dl `DL.snoc` (rwd, a))

      Done rwd (Just nextState) -> do
        newNet <- liftIO $ updateAgent state a rwd nextState qs model
        pure $ (newNet, dl `DL.snoc` (rwd, a))

  where
    updateAgent :: MonadIO io => StateFL -> Action -> Double -> StateFL -> R 4 -> (FLNetwork) -> io (FLNetwork)
    updateAgent s a r s' qs net = do
      let
        q_next :: Double
        q_next = maximum . Exts.toList . LA.extract . fst $ chooseAction net s'

        targetQ :: Double
        targetQ = realToFrac $ r + gamma * q_next

        newNet :: FLNetwork
        newNet = train gamma (stateToR16 state) (oneHotStatic r a) net

      -- when (r /= 0) (printIO ("woot!", r, s, s'))
      pure net

oneHotStatic :: Reward -> Action -> R 4
oneHotStatic r a = LA.vector $ fmap builder [minBound..maxBound]
  where
    builder :: Action -> Double
    builder i = if fromEnum a == fromEnum i then r else 0

chooseAction :: FLNetwork -> StateFL -> (R 4, Action)
chooseAction net state = (qs, a)
  where
    qs :: R 4
    qs = runNetOnInputs (stateToR16 state) net

    a :: Action
    a = toEnum . V.maxIndex . V.fromList . Exts.toList . LA.extract $ qs

stateToR16 :: StateFL -> R 16
stateToR16 state = LA.vector (fmap fromIntegral $ toList $ toVector state)



