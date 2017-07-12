{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE TypeOperators #-}
module Main where

import Zoo.Prelude
import Reactive.Banana.Combinators
import Data.Aeson (FromJSON, ToJSON)
import qualified Data.DList as DL

import Control.Arrow ((&&&))
import Servant
import Servant.API
import Servant.Server
import qualified Servant.Server as Server
import qualified Network.Wai.Handler.Warp as Warp
import Control.Concurrent.STM

main :: IO ()
main = do
  list <- newTVarIO (mempty::DList Step)
  Warp.run 1337 (app list)
  where
    app :: TVar (DList Step) -> Application
    app = serve reportingAPI . statefulServer

-- episode number, state, action, reward
data Step = Step
  { episodeNum :: Integer
  , observed :: [Double]
  , action :: Int
  , reward :: Double
  } deriving (Eq, Ord, Show, Generic, FromJSON)


type ReportingAPI
  = "step" :> ReqBody '[JSON] Step :> Post '[JSON] ()

  :<|> "episodic"   :> ( "actions" :> Get '[JSON] [(Integer, Int)]
                    :<|> "rewards" :> Get '[JSON] [(Integer, Double)]
                       )
  :<|> "continuous" :> ( "actions" :> Get '[JSON] [Int]
                    :<|> "rewards" :> Get '[JSON] [Double]
                       )

type ReportingM = StateT (DList Step) Server.Handler

statefulServer :: TVar (DList Step) -> Server ReportingAPI
statefulServer mdl = enter reportingToHander server
  where
    reportingToHander :: ReportingM :~> Server.Handler
    reportingToHander = Nat $ \x -> do
      dl <- liftIO $ readTVarIO mdl
      (a, dl') <- runStateT x dl
      liftIO $ atomically (writeTVar mdl dl')
      pure a

server :: ServerT ReportingAPI ReportingM
server = step
  :<|> (episodicActions   :<|> episodicRewards)
  :<|> (continuousActions :<|> continuousRewards)
  where
    step :: Step -> StateT (DList Step) Server.Handler ()
    step st =
      getState >>=
      putState . (`DL.snoc` st)

    episodicActions :: ReportingM [(Integer, Int)]
    episodicActions = getter (episodeNum &&& action)

    episodicRewards :: ReportingM [(Integer, Double)]
    episodicRewards   = getter (episodeNum &&& reward)

    continuousActions :: ReportingM [Int]
    continuousActions  = getter action

    continuousRewards :: ReportingM [Double]
    continuousRewards = getter reward

    getter :: (Step -> x) -> StateT (DList Step) Server.Handler [x]
    getter fn = fmap fn . DL.toList <$> getState


reportingAPI :: Proxy ReportingAPI
reportingAPI = Proxy

