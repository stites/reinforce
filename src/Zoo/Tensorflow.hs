{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE OverloadedLists #-}
module Zoo.Tensorflow where

import Zoo.Prelude

import Environments.Gym.Internal
import Control.Monad.Identity (Identity(..))
import Spaces.State hiding (oneHot)
import qualified TensorFlow.Gradient     as TF
import qualified TensorFlow.Core         as TF
import qualified TensorFlow.Ops          as TF
import qualified TensorFlow.GenOps.Core  as TF (square)
import qualified Data.Vector as V


type TFAction = Int32

data Layer v = Layer
  { layer_inputs  :: Tensor     v Float
  , layer_weights :: Tensor   Ref Float
  , layer_bias    :: Tensor   Ref Float
  , layer_outputs :: Tensor Build Float
  , layer_shape   :: (Int, Int)
  }

data Network = Network
  { initial :: Layer Value
  , output  :: Layer Build
  , shape   :: (Int, Int, Int)
  }


getInputs  :: Network -> Tensor Value Float
getInputs = layer_inputs . initial


getOutputs :: Network -> Tensor Build Float
getOutputs = layer_outputs . output


getWeights :: Network -> [Tensor Ref Float]
getWeights net = [i, o]
  where
    i = layer_weights $ initial net
    o = layer_weights $ output net


mkInputLayer
  :: Int
  -> Int
  -> (Tensor Build Float -> Tensor Build Float)
  -> TF.Build (Layer Value)
mkInputLayer input_size output_size activation =
  TF.placeholder [1, ci input_size] >>=
  \inputs -> mkLayer' input_size inputs output_size activation


mkHiddenLayer
  :: Layer v
  -> Int
  -> (Tensor Build Float -> Tensor Build Float)
  -> TF.Build (Layer Build)
mkHiddenLayer inLayer output_size activation =
  mkLayer' (snd $ layer_shape inLayer) (layer_outputs inLayer) output_size activation


mkLayer'
  :: Int
  -> Tensor v Float
  -> Int
  -> (Tensor Build Float -> Tensor Build Float)
  -> TF.Build (Layer v)
mkLayer' input_size inputs output_size activation = do
  weights <- TF.initializedVariable =<< randomParam input_size [ci input_size, ci output_size]
  bias    <- TF.zeroInitializedVariable [ci output_size]
  let outputs = activation $ (inputs `TF.matMul` weights) `TF.add` bias
  return $ Layer inputs weights bias outputs (input_size, output_size)


randomParam :: Int -> Shape -> Build (Tensor Build Float)
randomParam width (TF.Shape shape) =
  TF.truncatedNormal (TF.vector shape) >>= pure . (`TF.mul` stddev)
  where
    stddev :: Tensor Build Float
    stddev = TF.scalar (1 / sqrt (fromIntegral width))


mkFullyConnected :: Int -> Int -> Int -> TF.Build Network
mkFullyConnected action_size state_size hidden_size = do
  layer1  <- mkInputLayer  state_size hidden_size TF.relu
  layer2  <- mkHiddenLayer layer1     action_size TF.softmax
  pure $ Network layer1 layer2 (action_size, state_size, hidden_size)


decodeAction :: Enum a => Vector TFAction -> a
decodeAction = toEnum . fromIntegral . (V.! 0)

encodeState' :: StateSpace s => Int -> s -> TensorData Float
encodeState' state_size s = TF.encodeTensorData [1, ci state_size] stateVec
  where
    stateVec :: Vector Float
    stateVec = V.fromList $ fmap cf $ V.toList $ toVector s

encodeAction' :: Enum a => Int -> a -> TensorData Float
encodeAction' action_size a =
  TF.encodeTensorData [1, ci action_size]
    $ oneHot (ci action_size) (fromEnum a)



