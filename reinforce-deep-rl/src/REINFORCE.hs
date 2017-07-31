{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE OverloadedLists #-}
module REINFORCE where

import Zoo.Prelude
import Zoo.Tensorflow
import Zoo.Internal

import Control.Monad.Identity (Identity(..))
import Environments.Gym.ClassicControl.CartPoleV0 (EnvironmentT(..))
import Data.List.NonEmpty (NonEmpty)
import Data.CartPole (Action, StateCP)
import Spaces.State hiding (oneHot)
import qualified Control.MonadEnv        as Env
import qualified Control.MonadMWCRandom  as MMWC
import qualified Data.DList              as DL
import qualified Data.List.NonEmpty      as NE
import qualified Data.Text               as T
import qualified TensorFlow.Gradient     as TF
import qualified TensorFlow.Core         as TF
import qualified TensorFlow.Ops          as TF
import qualified TensorFlow.BuildOp      as TF
import qualified TensorFlow.Output       as TF
import qualified TensorFlow.GenOps.Core  as TF (square, log, real, bitcast, placeholderV2')
import qualified Data.Vector as V
import Lens.Micro.Platform



maxEpisodes, maxSteps, batchSize, hiddenSize :: Int
gamma :: Float

maxEpisodes = 2000
maxSteps    = 99
batchSize   = 5
gamma        = 0.99

actionSize = 1
stateSize  = 1
hiddenSize = 8

lr :: Tensor Build Float
lr = TF.scalar 0.1

-- ========================================================================= --
-- Build Tensorflow graph
-- ========================================================================= --
data Model = Model
  { trainBatch :: TensorData Float -> TensorData Float -> Session ()
  , predict :: StateCP -> Session (Vector Float)
  , weights :: [Tensor Ref Float]
  , getGrads :: TensorData Float -> TensorData Float -> Session [Tensor Ref Float]
  , size :: (Int, Int)
  }


run :: (Model -> TensorData Float -> TensorData Float -> Session x) -> Model -> [Action] -> [Reward] -> Session x
run fn m rs as = fn m (encode $ fmap cf rs) (encode $ fmap (ci.fromEnum) as)
  where
    encode :: [Float] -> TensorData Float
    encode = TF.encodeTensorData [] . V.fromList


getGrads' :: Model -> [Action] -> [Reward] -> Session [Tensor Ref Float]
getGrads' = run getGrads

trainBatch' :: Model -> [Action] -> [Reward] -> Session ()
trainBatch' = run trainBatch


createAgent :: Build Model
createAgent = do
  network <- mkFullyConnected actionSize stateSize hiddenSize

  let
    (asize, ssize) = (actionSize, stateSize)

    outputs :: Tensor Build Float
    outputs = getOutputs network

    inputs  :: Tensor Value Float
    inputs  = getInputs network

    weights :: [Tensor Ref Float]
    weights = getWeights network


  rewardHolder :: Tensor Value Float    <- TF.placeholder []
  actionHolder :: Tensor Value TFAction <- TF.placeholder []

  let
    one_hot_actions     = TF.bitcast (TF.oneHot actionHolder (tfs asize) (tfs 1) (tfs 0))
    one_hot_outputs     = outputs `TF.mul` one_hot_actions
    responsible_outputs = TF.sum one_hot_outputs (tfs 1)     -- row-wise sum, so basically, remove one_hot
    loss                = TF.neg $ TF.mean (TF.mul (TF.log responsible_outputs) rewardHolder) (tfs 1 {-???-})

  predicted <- TF.render . TF.cast $ outputs
  grads     <- TF.gradients      loss weights
  trainStep <- applyGradients weights   grads

  let
    trainBatch :: TensorData Float -> TensorData Float -> Session ()
    trainBatch inFeed rwdFeed =
      TF.runWithFeeds_
        [ TF.feed inputs    inFeed
        , TF.feed predicted rwdFeed
        ] trainStep

    predict :: StateCP -> Session (Vector Float)
    predict st = TF.runWithFeeds [TF.feed inputs (encodeState' ssize st)] predicted

    getGrads :: TensorData Float -> TensorData Float -> Session [Tensor Ref Float]
    getGrads inFeed rwdFeed =
      TF.runWithFeeds
        [ TF.feed inputs    inFeed
        , TF.feed predicted rwdFeed
        ] grads

  pure $ Model trainBatch predict weights getGrads (asize, ssize)

  where
    tfs :: Int -> Tensor Build Int32
    tfs = TF.scalar . ci

    indexAsName :: Int -> TF.PendingNodeName
    indexAsName idx = TF.ExplicitName . T.pack $ show idx ++"_holder"

    applyGradients :: [Tensor Ref Float] -> [Tensor Value Float] -> Build ControlNode
    applyGradients weights grads = zipWithM applyGrad weights grads >>= TF.group
      where
        applyGrad :: Tensor Ref Float -> Tensor Value Float -> Build (Tensor Ref Float)
        applyGrad param grad = TF.assign param $ param `TF.sub` (lr `TF.mul` grad)


type EnvSession = EnvironmentT (MWCRandT Session)
type GradBuffer = [Tensor Build Float]
type Reward     = Env.Reward


trainer
  :: forall m . Monad m
  => (Int -> GradBuffer -> m GradBuffer)
  -> GradBuffer
  -> Int
  -> m GradBuffer
trainer fn bf maxEpn = go bf 0
  where
    go :: GradBuffer -> Int -> m GradBuffer
    go !gradbuffer !epn = do
      if epn > maxEpn
      then pure gradbuffer
      else do
        nxt <- fn epn gradbuffer
        go nxt (epn+1)


learn :: EnvSession ()
learn = do
  m <- liftTF (TF.build createAgent)
  finalGrad <- trainer (runLearner m) (resetBuffer (weights m)) maxEpisodes
  undefined

  where
    resetBuffer :: [Tensor v Float] -> GradBuffer
    resetBuffer = fmap (0 `TF.mul`)

    runLearner :: Model -> Int -> GradBuffer -> EnvSession GradBuffer
    runLearner m epn buffer = do
      a <- Env.reset
      case a of
        EmptyEpisode -> pure buffer
        Initial s    -> do
          (hist, grads) <- rolloutEpisode epn m 0.1 s DL.empty Nothing

          let
            gradbuffer = zipWith TF.add grads buffer
            (rs, as)   = unzip (DL.toList hist)

          when (epn `divisibleBy` 100) $ report rs

          if epn `divisibleBy` batchSize && epn /= 0
          then do
            liftTF (trainBatch' m as rs)
            pure (resetBuffer buffer)

          else pure gradbuffer


rolloutEpisode
  :: Int
  -> Model
  -> Float
  -> StateCP
  -> DList (Reward, Action)
  -> Maybe GradBuffer
  -> EnvSession (DList (Reward, Action), GradBuffer)
rolloutEpisode episodeNum model epsilon state dl mbuff =
  go 0 epsilon state dl mbuff 0

  where
    go :: Int -> Float -> StateCP -> DList (Reward, Action) -> Maybe GradBuffer -> Float -> EnvSession (DList (Reward, Action), GradBuffer)
    go stepNum epsilon state dl mbuff runningReward
      | stepNum > maxSteps = case mbuff of
          Nothing   -> error "should never happen!"
          Just buff -> return (dl, buff)

      | otherwise = liftTF (predict model state)
        >>= toEnum . fst <$> MMWC.sampleFrom . V.toList . V.map realToFrac
        >>= \a -> Env.step a
        >>= \case
          Next rwd nextState -> do

            go (stepNum+1) epsilon nextState (dl `DL.snoc` (rwd, a)) mbuff (runningReward + cf rwd)

          Done rwd _ -> do

            pure $ (dl `DL.snoc` (rwd, a), mbuff)
--        grads = sess.run(self.gradients, feed_dict={
--                self.inputs: _c(np.vstack, np.array, _p(lmap, _0))(ep_history),
--                self.action_holder: lmap(_1, ep_history),
--                self.reward_holder: eligibility_trace(self.gamma, lmap(_2, ep_history))
--            })
--
--        return running_reward, step_num, grads
