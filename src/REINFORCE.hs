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


max_episodes, max_steps, batch_size, hidden_size :: Int32
lr, gamma :: Float

max_episodes = 2000
max_steps    = 99
batch_size   = 5
hidden_size  = 8
lr           = 0.1
gamma        = 0.99

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
    <$> (runReaderT infer vars)
    <*> (runReaderT train vars)


mkAgentGraph :: Int64 -> Int64 -> Int64 -> TF.Build GraphRefs
mkAgentGraph action_size state_size hidden_size = do
  network <- mkFullyConnected action_size state_size hidden_size
  let qOut   = getOutputs network
      inputs = getInputs network
  predict <- TF.render . TF.cast $ TF.argMax qOut (TF.scalar (1 :: TFAction))
  nextQs  <- TF.placeholder [ 1, action_size]
  return GraphRefs
    { network = network
    , predict = predict
    , nextQs  = nextQs
    }

infer = undefined
train = undefined



