{-# LANGUAGE OverloadedLists #-}
module REINFORCE where

import Zoo.Prelude
import Zoo.Tensorflow

import Control.Monad.Identity (Identity(..))
import Spaces.State hiding (oneHot)
import qualified TensorFlow.Gradient     as TF
import qualified TensorFlow.Core         as TF
import qualified TensorFlow.Ops          as TF
import qualified TensorFlow.GenOps.Core  as TF (square)
import qualified Data.Vector as V


maxEpisodes, maxSteps, batchSize, hiddenSize :: Int32
gamma :: Float

maxEpisodes = 2000
maxSteps    = 99
batchSize   = 5
hiddenSize  = 8
gamma        = 0.99

lr :: Tensor Build Float
lr = TF.scalar 0.1

-- ========================================================================= --
-- Build Tensorflow graph
-- ========================================================================= --

data Model s a = Model
  { choose :: s -> Session (a, Vector Float)
  , update :: TensorData Float -> TensorData Float -> Session ()
  }


data GraphRefs = GraphRefs
  { network  :: !Network
  , predict :: !(Tensor Value TFAction)
  , nextQs  :: !(Tensor Value    Float)
  }


-- Build a tensorflow graph representation
createAgent :: (Enum a, StateSpace s) => Build (Model s a)
createAgent = mkAgentGraph 1 16 32
  >>= \vars -> Model
    <$> runReaderT infer vars
    <*> runReaderT train vars


mkAgentGraph :: Int64 -> Int64 -> Int64 -> TF.Build GraphRefs
mkAgentGraph actionSize stateSize hiddenSize = do
  network <- mkFullyConnected actionSize stateSize hiddenSize
  let qOut   = getOutputs network
      inputs = getInputs network
  predict <- TF.render . TF.cast $ TF.argMax qOut (TF.scalar (1 :: TFAction))
  nextQs  <- TF.placeholder [ 1, actionSize]
  return GraphRefs
    { network = network
    , predict = predict
    , nextQs  = nextQs
    }


infer :: (Enum a, StateSpace s) => ReaderT GraphRefs Build (s -> Session (a, Vector Float))
infer = do
  GraphRefs{network, predict} <- ask
  let
    inputs = getInputs network
    qOut   = getOutputs network
    (_, stateSize, _)  = shape network
    encodeState = encodeState' stateSize
  return $ \st -> do
    a  <- decodeAction <$> TF.runWithFeeds [TF.feed inputs (encodeState st)] predict
    qs <-                  TF.runWithFeeds [TF.feed inputs (encodeState st)] qOut
    pure (a, qs)


train :: ReaderT GraphRefs Build (TensorData Float -> TensorData Float -> Session ())
train = do
  GraphRefs{network, nextQs} <- ask
  let
    inputs :: Tensor Value Float
    inputs = getInputs network

    qOut :: Tensor Build Float
    qOut = getOutputs network

    weights :: [Tensor Ref Float]
    weights = getWeights network

    loss :: Tensor Build Float
    loss = TF.reduceSum $ TF.square (nextQs `TF.sub` qOut)

  grads     <- lift $ TF.gradients     loss weights
  trainStep <- lift $ applyGradients weights  grads

  return $ \inFeed rwdFeed ->
    TF.runWithFeeds_
      [ TF.feed inputs inFeed
      , TF.feed nextQs rwdFeed
      ] trainStep

  where
    applyGradients :: [Tensor Ref Float] -> [Tensor Value Float] -> Build ControlNode
    applyGradients weights grads = zipWithM applyGrad weights grads >>= TF.group
      where
        applyGrad :: Tensor Ref Float -> Tensor Value Float -> Build (Tensor Ref Float)
        applyGrad param grad = TF.assign param $ param `TF.sub` (lr `TF.mul` grad)




