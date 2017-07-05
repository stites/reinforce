-------------------------------------------------------------------------------
-- THIS IS A WORK IN PROGRESS AND IS CURRENTLY A BROKEN IMPLEMENTATION
-------------------------------------------------------------------------------
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE NamedFieldPuns #-}
module QNetwork where

import Zoo.Prelude
import Control.MonadEnv.Internal
import Environments.Gym.ToyText.FrozenLakeV0 hiding (Left, Right)
import Control.MonadMWCRandom (MWCRandT(..))
import Control.Monad.Trans

import qualified System.Random.MWC       as MWC
import qualified Control.MonadMWCRandom  as MMWC
import qualified Control.MonadEnv        as Env
import qualified Data.Logger             as Logger
import qualified Data.DList              as DL
import           Data.List               (groupBy)
import           Data.Function           (on)
import qualified Data.Vector             as V
import qualified TensorFlow.Gradient     as TF
import qualified TensorFlow.Core         as TF
import qualified TensorFlow.Ops          as TF

import Debug.Trace


type Event = Logger.Event Reward StateFL Action


numEpisodes :: Int
numEpisodes = 2000


main :: IO ()
main = do
  g <- MWC.createSystemRandom
  let learn100eps = replicateM 100 (learn e)
  TF.runSession (MMWC.runMWCRandT (runDefaultEnvironmentT False learn100eps) g)
  >>= \case
    Left err ->
      throwString (show err)

    Right (histToRewards -> rlist) -> do
      let strPer = show (sum rlist / fromIntegral numEpisodes)
      print $ "Percent succesful episodes: " ++ strPer ++ "%"

  where
    e :: Float
    e = 0.1

    eventToRewards :: Event -> (Integer, Reward)
    eventToRewards (Logger.Event i r o a) = (i, r)

    histToRewards :: DL.DList Event -> [Reward]
    histToRewards rs = fmap sum episodes
      where
        episodes :: [[Reward]]
        episodes = (fmap.fmap) snd $
          groupBy (\l r -> fst l == fst r)
            (fmap eventToRewards . DL.toList $ rs)



liftTF :: Session a -> EnvironmentT (MMWC.MWCRandT Session) a
liftTF = lift . lift


printIO :: (Show s, MonadIO io) => s -> io ()
printIO = liftIO . print


learn :: Float -> EnvironmentT (MMWC.MWCRandT Session) (DL.DList (Reward, Action), Float)
learn e = do
  m   <- liftTF (TF.build createModel)
  obs <- Env.reset
  case obs of
    EmptyEpisode -> return (DL.empty, e)
    Initial s    -> rolloutEpisode m 0 e s DL.empty


rolloutEpisode :: Model -> Int -> Float -> StateFL -> DL.DList (Reward, Action) -> EnvironmentT (MMWC.MWCRandT Session) (DL.DList (Reward, Action), Float)
rolloutEpisode !m@Model{trainit, chooseit} !i !e !s !dl
  | i > 100   = return (dl, e)
  | otherwise = do
    !a'   <- dataToAction <$> liftTF (chooseit (stateToData s))
    !p    <- lift $ MMWC.uniformR (0, 1::Float)
    !a    <- if p < e then randomAction else pure a'
    !next <- Env.step a
    case next of
      Next !rwd !st -> do
        let targetQ = rwd + y
        let ctda = choiceToData a
            stds = stateToData s
        liftTF $ trainit stds (rewardTD targetQ a)
        rolloutEpisode m (i+1) e st (dl `DL.snoc` (rwd, a))
      Done !rwd -> do
        let targetQ = rwd + y
        liftTF $ trainit (choiceToData a) (rewardTD targetQ a)
        pure (dl `DL.snoc` (rwd, a), 1 / ((i//50) + 10))

  where
    y = 0.99
    maxSteps = 100

    (//) :: Int -> Int -> Float
    (//) = (/) `on` fromIntegral

    rewardTD :: Double -> Action -> TensorData Float
    rewardTD (realToFrac->f) a = TF.encodeTensorData [1, 4] ((* f) <$> actionToVector a)

    asFloats :: Vector Int -> Vector Float
    asFloats = fmap fromIntegral

    stateToData :: StateFL -> TensorData Float
    stateToData s = TF.encodeTensorData [1, 16] (asFloats $ toVector s)

    dataToAction :: Vector TFAction -> Action
    dataToAction = toEnum . fromIntegral . (V.! 0)

    choiceToData :: Action -> TensorData Float
    choiceToData = TF.encodeTensorData [1, 4] . actionToVector

    actionToVector :: Action -> Vector Float
    actionToVector a = asFloats $ V.generate 4 (fromEnum . (== fromEnum a))

    randomAction :: EnvironmentT (MMWC.MWCRandT Session) Action
    randomAction = toEnum <$> (lift $ MMWC.uniformR (0, fromEnum maxAct))
      where
        maxAct :: Action
        maxAct = maxBound


-- Build a tensorflow graph representation
createModel :: Build Model
createModel = do
  vars     <- mkVariables
  chooseit <- runReaderT infer'    vars
  trainit  <- runReaderT training' vars
  return Model
    { trainit = trainit
    , chooseit = chooseit
    }

-- ========================================================================= --
-- | An action must have an integral representation.
type TFAction = Int32

data Model = Model
  { chooseit :: TensorData Float -> Session (Vector TFAction)
  , trainit  :: TensorData Float -> TensorData Float -> Session ()
  }


data GraphRefs = GraphRefs
  { inputs  :: !(Tensor Value    Float)
  , weights :: !(Tensor   Ref    Float)
  , predict :: !(Tensor Value TFAction)
  , qOut    :: !(Tensor Build    Float)
  , nextQs  :: !(Tensor   Ref    Float)
  }


mkVariables :: TF.Build GraphRefs
mkVariables = do
  inputs  <- TF.placeholder [1, 16]
  weights <- TF.initializedVariable =<< randomParam 16 [16, 4]
  let qOut = inputs `TF.matMul` weights -- :: Tensor [16 x 4]
  predict <- TF.render . TF.cast $ TF.argMax qOut (TF.scalar (1 :: TFAction))
  nextQs  <- TF.initializedVariable =<< randomParam  1 [ 1, 4]
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


infer' :: ReaderT GraphRefs Build (TensorData Float -> Session (Vector TFAction))
infer' = do
  GraphRefs{inputs, predict} <- ask
  return $ \ins ->
    TF.runWithFeeds [TF.feed inputs ins] predict


training' :: ReaderT GraphRefs Build (TensorData Float -> TensorData Float -> Session ())
training' = do
  GraphRefs{inputs, weights, nextQs, qOut} <- ask
  let
    loss :: Tensor Build Float
    loss = TF.reduceSum $ (nextQs `TF.sub` qOut) `TF.matMul` (nextQs `TF.sub` qOut)

    -- inputs  :: [ 1 x 16]
    -- weights :: [16 x  4]
    -- qOut    :: [ 1 x  4]
    -- predict :: Vector TFAction
    -- nextQs  :: [ 1 x  4]

    params :: [Tensor Ref Float]
    params = [weights]

  grads     <- lift $ TF.gradients     loss params
  trainStep <- lift $ applyGradients params  grads

  return $ \inFeed rwdFeed -> do
    printIO "train"
    printIO (TF.decodeTensorData inFeed  :: Vector Float)
    printIO (TF.decodeTensorData rwdFeed :: Vector Float)

    TF.runWithFeeds_
      [ TF.feed inputs inFeed
      , TF.feed nextQs rwdFeed
      ] trainStep

  where
    applyGradients
      :: [Tensor TF.Ref   Float]
      -> [Tensor TF.Value Float]
      -> Build TF.ControlNode
    applyGradients params grads = zipWithM applyGrad params grads >>= TF.group

    applyGrad
      :: Tensor TF.Ref   Float
      -> Tensor TF.Value Float
      -> Build (Tensor TF.Ref Float)
    applyGrad param grad = TF.assign param $ param `TF.sub` (lr `TF.mul` grad)

    lr :: Tensor Build Float
    lr = TF.scalar 0.00001

