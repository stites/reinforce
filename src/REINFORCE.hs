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
  , size :: (Int, Int)
  }

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

  pure $ Model trainBatch predict weights (asize, ssize)

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
    runLearner m@Model{trainBatch} epn buffer = do
      Env.reset >>= \case
        EmptyEpisode -> pure buffer
        Initial s    -> do
          (hist, grads) <- rolloutEpisode epn m 0 0.1 s DL.empty Nothing

          let
            gradbuffer = zipWith TF.add grads buffer
            (rs, as)   = unzip (DL.toList hist)

          when (epn `divisibleBy` 100) $ report rs

          if epn `divisibleBy` batchSize && epn /= 0
          then do
            liftTF (trainBatch (encode $ fmap cf rs) (encode $ fmap (ci.fromEnum) as))
            pure (resetBuffer buffer)

          else pure gradbuffer

    encode :: [Float] -> TensorData Float
    encode = TF.encodeTensorData [] . V.fromList


    -- updateAgent :: StateCP -> Action -> Double -> StateCP -> Vector Float -> Model -> EnvSession ()
    -- updateAgent s a r s' qs (updateModel, chooseAction, _) = do
    --   q_next <- (realToFrac . maximum . snd) <$> liftTF (chooseAction s')
    --   let targetQ = realToFrac $ r + gamma * q_next
    --   liftTF $ updateModel (encodeState s) (rewardTD targetQ a qs)
    --   when (r /= 0) (printIO ("woot!", r, s, s'))

    rolloutEpisode
      :: Int
      -> Model
      -> Int
      -> Float
      -> StateCP
      -> DList (Reward, Action)
      -> Maybe GradBuffer
      -> EnvSession (DList (Reward, Action), GradBuffer)
    rolloutEpisode episodeNum model stepNum epsilon state dl mbuff
      | stepNum > maxSteps = case mbuff of
          Nothing   -> error "should never happen!"
          Just buff -> return (dl, buff)

      | otherwise = liftTF (predict model state)
        >>= toEnum . fst <$> MMWC.sampleFrom . V.toList . V.map realToFrac
        >>= Env.step
        >>= \case
          Next rwd nextState -> do
            updateAgent state a rwd nextState qs model
            rolloutEpisode episodeNum model (stepNum+1) epsilon nextState (dl `DL.snoc` (rwd, a))
--     def _rollout_ep(self, s, eps, sess):
--         running_reward = 0
--         step_num = 0
--         done = False
--         ep_history = []
--
--         while step_num < self.max_steps and not done:
--             action_dist        = sess.run(self.outputs, feed_dict={self.inputs:[s]})
--             action             = choose_action(action_dist, action_dist[0])
--             s_next, rwd, done, _ = self.env.step(action)
--             ep_history.append([s, action, rwd])
--
--             # Book-keeping
--             step_num += 1
--             running_reward += rwd
--             s = s_next


          Done rwd (Just nextState) -> do
            updateAgent state a rwd nextState qs model
            pure $ (dl `DL.snoc` (rwd, a), buff)
--        grads = sess.run(self.gradients, feed_dict={
--                self.inputs: _c(np.vstack, np.array, _p(lmap, _0))(ep_history),
--                self.action_holder: lmap(_1, ep_history),
--                self.reward_holder: eligibility_trace(self.gamma, lmap(_2, ep_history))
--            })
--
--        return running_reward, step_num, grads

