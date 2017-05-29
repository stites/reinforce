-------------------------------------------------------------------------------
-- THIS IS A WORK IN PROGRESS AND IS CURRENTLY A BROKEN IMPLEMENTATION
-------------------------------------------------------------------------------
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE NamedFieldPuns #-}
module QNetwork where

import Baseline.Prelude
import Control.MonadEnv.Internal
import Environments.Gym.ToyText.FrozenLakeV0 hiding (Left, Right)

import qualified Control.MonadEnv     as Env
import qualified Data.Logger          as Logger
import qualified Data.DList           as DL
import qualified TensorFlow.Gradient  as TF
import qualified TensorFlow.Core      as TF
import qualified TensorFlow.Ops       as TF


type Event = Logger.Event Reward StateFL Action

numEpisodes :: Int
numEpisodes = 2000


main :: IO ()
main =
  runDefaultEnvironment False learn
  >>= \case
    Left err ->
      throwString (show err)

    Right (histToRewards -> rlist) -> do
      let strPer = show (sum rlist / fromIntegral numEpisodes)
      print $ "Percent succesful episodes: " ++ strPer ++ "%"

  where
    eventToRewards :: Event -> Reward
    eventToRewards (Logger.Event i r o a) = r

    histToRewards :: DL.DList Event -> [Reward]
    histToRewards = fmap eventToRewards . DL.toList


learn :: Environment ()
learn = undefined


learn' :: Session a
learn' = do
  model <- TF.build $ createModel
  forM_ ([0..numEpisodes]::[Int]) $ \i -> do
    -- chooseit model -- :: TensorData Float -> Session (Vector TFAction)
    -- stepit model -- :: TensorData Float -> TensorData Float -> Session ()
    s <- liftIO $ runDefaultEnvironment True (Env.step Up)
    -- stepit model [0..15] undefined


    --(liftIO Env.reset) >>= \case
    --  EmptyEpisode -> return ()
    --  Initial s -> rollout model 0 s
    undefined
  return undefined


  where
    y = 0.99
    e = 0.1
    maxSteps = 100

    rollout :: Model -> Int -> StateFL -> Environment ()
    rollout m i s
      | i > 100   = return ()
      | otherwise = do
         -- a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
    --     model' <- liftIO . TF.run $ train m undefined undefined
         return ()
          -- putStrLn $ "training error " ++ show (err * 100)

    --     #The Q-Network
    --     while j < 99:
    --         j+=1
    --         #Choose an action by greedily (with e chance of random action) from the Q-network
    --         if np.random.rand(1) < e:
    --             a[0] = env.action_space.sample()
    --         #Get new state and reward from environment
    --         s1,r,d,_ = env.step(a[0])
    --         #Obtain the Q' values by feeding the new state through our network
    --         Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
    --         #Obtain maxQ' and set our target value for chosen action.
    --         maxQ1 = np.max(Q1)
    --         targetQ = allQ
    --         targetQ[0,a[0]] = r + y*maxQ1
    --         #Train our network using target and predicted Q values
    --         _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
    --         rAll += r
    --         s = s1
    --         if d == True:
    --             #Reduce chance of random action as we train the model.
    --             e = 1./((i/50) + 10)
    --             break
    --     jList.append(j)
    --     rList.append(rAll)



-- Build a tensorflow graph representation
createModel :: Build Model
createModel = do
  vars     <- mkVariables
  chooseit <- runReaderT infer'    vars
  stepit   <- runReaderT training' vars
  return Model
    { stepit = stepit
    , chooseit = chooseit
    }

-- ========================================================================= --
-- | An action must have an integral representation.
type TFAction = Int32

data Model = Model
  { chooseit :: TensorData Float -> Session (Vector TFAction)
  , stepit :: TensorData Float -> TensorData Float -> Session ()
  }


data GraphRefs = GraphRefs
  { inputs  :: Tensor Value    Float
  , weights :: Tensor   Ref    Float
  , predict :: Tensor Value TFAction
  , qOut    :: Tensor Build    Float
  , nextQs  :: Tensor   Ref    Float
  }


mkVariables :: TF.Build GraphRefs
mkVariables = do
  inputs  <- TF.placeholder [1, 16]
  weights <- TF.initializedVariable =<< randomParam 16 [16, 4]
  let qOut = (inputs `TF.matMul` weights)
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
  return $ \ins -> TF.runWithFeeds [TF.feed inputs ins] predict


training' :: ReaderT GraphRefs Build (TensorData Float -> TensorData Float -> Session ())
training' = do
  GraphRefs{inputs, weights, nextQs, qOut} <- ask
  let
    loss :: Tensor Build Float
    loss = TF.reduceSum $ (nextQs `TF.sub` qOut) `TF.matMul` (nextQs `TF.sub` qOut)

    params :: [Tensor Ref Float]
    params = [weights]

  grads     <- lift $ TF.gradients loss params
  trainStep <- lift $ applyGradients params grads

  return $ \inFeed rwdFeed ->
    TF.runWithFeeds_
      [ TF.feed inputs inFeed
      , TF.feed nextQs rwdFeed
      ] trainStep

  where
    applyGradients :: [Tensor TF.Ref Float] -> [Tensor TF.Value Float] -> Build TF.ControlNode
    applyGradients params grads = zipWithM applyGrad params grads >>= TF.group

    applyGrad :: Tensor TF.Ref Float -> Tensor TF.Value Float -> Build (Tensor TF.Ref Float)
    applyGrad param grad = TF.assign param $ param `TF.sub` (lr `TF.mul` grad)

    lr :: Tensor Build Float
    lr = TF.scalar 0.00001

