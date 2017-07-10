{-# LANGUAGE OverloadedLists #-}
module QNetwork where

import Zoo.Prelude
import Control.MonadEnv.Internal
import Environments.Gym.ToyText.FrozenLakeV0 hiding (Left, Right)

import qualified System.Random.MWC       as MWC
import qualified Control.MonadMWCRandom  as MMWC
import qualified Control.MonadEnv        as Env
import qualified Data.Logger             as Logger
import qualified Data.DList              as DL
import qualified Data.Vector             as V
import qualified TensorFlow.Gradient     as TF
import qualified TensorFlow.Core         as TF
import qualified TensorFlow.Ops          as TF
import qualified TensorFlow.GenOps.Core  as TF (square)


type Event = Logger.Event Reward StateFL Action

-- | An action must have an integral representation.
type TFAction = Int32


maxEpisodes :: Int
maxEpisodes = 3000

maxSteps :: Int
maxSteps = 99

gamma :: Double
gamma = 0.99

main :: IO ()
main = do
  g <- MWC.createSystemRandom
  TF.runSession (MMWC.runMWCRandT (runDefaultEnvironmentT False learn) g)
  >>= \case
    Left err                       -> throwString (show err)
    Right (histToRewards -> rlist) -> report rlist

  where
    eventToRewards :: Event -> (Integer, Reward)
    eventToRewards (Logger.Event i r o a) = (i, r)

    histToRewards :: DList Event -> [Reward]
    histToRewards rs = fmap sum episodes
      where
        episodes :: [[Reward]]
        episodes = (fmap.fmap) snd $
          groupBy (\l r -> fst l == fst r)
            (fmap eventToRewards . DL.toList $ rs)


report :: MonadIO io => [Reward] -> io ()
report rwds = do
  let per = sum rwds / fromIntegral (genericLength rwds)
  let last50Per = sum (lastN 50 rwds) / 50
  printIO $ "Percent successful episodes: " ++ show per       ++ "%"
  printIO $ "Percent successful last 50 : " ++ show last50Per ++ "%"


learn :: EnvironmentT (MWCRandT Session) [DList (Reward, Action)]
learn = do
  m   <- liftTF (TF.build createAgent)
  mapM (\epn -> go m epn) ([0..maxEpisodes]::[Int])
  where
    go :: Model -> Int -> EnvironmentT (MWCRandT Session) (DList (Reward, Action))
    go m epn = do
      obs <- Env.reset
      case obs of
        EmptyEpisode -> return DL.empty
        Initial s    -> do
          rs <- rolloutEpisode epn m 0 (decayEpsilon epn) s DL.empty
          when (epn `divisibleBy` 100) $ report (fst <$> DL.toList rs)
          pure rs


epsilonGreedyChoice :: Action -> Float -> EnvironmentT (MWCRandT Session) Action
epsilonGreedyChoice a e =
  lift (MMWC.uniformR (0, 1))
  >>= \p->
    if p < e
    then randomAction
    else pure a


randomAction :: EnvironmentT (MMWC.MWCRandT Session) Action
randomAction = toEnum <$> (lift $ MMWC.uniformR (0, fromEnum (maxBound :: Action)))


rolloutEpisode
  :: Int
  -> Model
  -> Int
  -> Float
  -> StateFL
  -> DList (Reward, Action)
  -> EnvironmentT (MWCRandT Session) (DList (Reward, Action))
rolloutEpisode episodeNum model stepNum epsilon state dl
  | stepNum > maxSteps = return dl
  | otherwise = do
    (a', qs) <- liftTF (chooseAction model state)
    a        <- epsilonGreedyChoice a' epsilon
    next     <- Env.step a
    case next of
      Next rwd nextState -> do
        updateAgent state a rwd nextState qs model
        rolloutEpisode episodeNum model (stepNum+1) epsilon nextState (dl `DL.snoc` (rwd, a))

      Done rwd (Just nextState) -> do
        updateAgent state a rwd nextState qs model
        pure $ dl `DL.snoc` (rwd, a)

  where
    rewardTD :: Float -> Action -> Vector Float -> TensorData Float
    rewardTD targetQ a qs = TF.encodeTensorData [1, 4] (qs V.// [(fromEnum a, targetQ)])

    updateAgent :: StateFL -> Action -> Double -> StateFL -> Vector Float -> Model -> EnvironmentT (MWCRandT Session) ()
    updateAgent s a r s' qs Model{updateModel, chooseAction} = do
      q_next <- (realToFrac . maximum . snd) <$> liftTF (chooseAction s')
      let targetQ = realToFrac $ r + gamma * q_next
      liftTF $ updateModel (encodeState s) (rewardTD targetQ a qs)
      when (r /= 0) (printIO ("woot!", r, s, s'))


decayEpsilon :: Int -> Float
decayEpsilon epNum = 1.0 / ((epNum // 50) + 10)

-- ========================================================================= --

data Model = Model
  { chooseAction :: StateFL -> Session (Action, Vector Float)
  , updateModel  :: TensorData Float -> TensorData Float -> Session ()
  }


data GraphRefs = GraphRefs
  { inputs  :: !(Tensor Value    Float)
  , weights :: !(Tensor   Ref    Float)
  , qOut    :: !(Tensor Build    Float)
  , predict :: !(Tensor Value TFAction)
  , nextQs  :: !(Tensor Value    Float)
  }


-- Build a tensorflow graph representation
createAgent :: Build Model
createAgent = mkAgentGraph
  >>= \vars -> Model
    <$> (runReaderT infer vars)
    <*> (runReaderT train vars)


mkAgentGraph :: TF.Build GraphRefs
mkAgentGraph = do
  inputs  <- TF.placeholder [1, 16]
  weights <- TF.initializedVariable =<< randomParam 16 [16, 4]
  let qOut = inputs `TF.matMul` weights
  predict <- TF.render . TF.cast $ TF.argMax qOut (TF.scalar (1 :: TFAction))
  nextQs  <- TF.placeholder [ 1, 4]
  return GraphRefs
    { inputs  = inputs
    , weights = weights
    , predict = predict
    , qOut    = qOut
    , nextQs  = nextQs
    }

  where
    -- | Create tensor with random values where the stddev depends on the width.
    randomParam :: Int64 -> Shape -> Build (Tensor Build Float)
    randomParam width (TF.Shape shape) =
      TF.truncatedNormal (TF.vector shape) >>= pure . (`TF.mul` stddev)
      where
        stddev :: Tensor Build Float
        stddev = TF.scalar (1 / sqrt (fromIntegral width))


infer :: ReaderT GraphRefs Build (StateFL -> Session (Action, Vector Float))
infer = do
  GraphRefs{inputs, qOut, predict} <- ask
  return $ \st -> do
    a  <- decodeAction <$> TF.runWithFeeds [TF.feed inputs (encodeState st)] predict
    qs <-                  TF.runWithFeeds [TF.feed inputs (encodeState st)] qOut
    pure (a, qs)

  where
    decodeAction :: Vector TFAction -> Action
    decodeAction = toEnum . fromIntegral . (V.! 0)


encodeState :: StateFL -> TensorData Float
encodeState s = TF.encodeTensorData [1, 16] (asFloats $ toVector s)


encodeAction :: Action -> TensorData Float
encodeAction a = TF.encodeTensorData [1, 4] $ oneHot 4 (fromEnum a)


train :: ReaderT GraphRefs Build (TensorData Float -> TensorData Float -> Session ())
train = do
  GraphRefs{inputs, weights, nextQs, qOut} <- ask
  let
    loss :: Tensor Build Float
    loss = TF.reduceSum $ TF.square (nextQs `TF.sub` qOut)

    params :: [Tensor Ref Float]
    params = [weights]

  grads     <- lift $ TF.gradients     loss params
  trainStep <- lift $ applyGradients params  grads

  return $ \inFeed rwdFeed -> do
    TF.runWithFeeds_
      [ TF.feed inputs inFeed
      , TF.feed nextQs rwdFeed
      ] trainStep

  where
    applyGradients :: [Tensor Ref Float] -> [Tensor Value Float] -> Build ControlNode
    applyGradients params grads = zipWithM applyGrad params grads >>= TF.group
      where
        applyGrad :: Tensor Ref Float -> Tensor Value Float -> Build (Tensor Ref Float)
        applyGrad param grad = TF.assign param $ param `TF.sub` (lr `TF.mul` grad)

        lr :: Tensor Build Float
        lr = TF.scalar 0.00001


