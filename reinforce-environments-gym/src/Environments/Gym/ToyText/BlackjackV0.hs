--------------------------------------------------------------------------------
-- |
-- Module    :  Environment.Gym.ToyText.BlackjackV0
-- Copyright :  (c) Sentenai 2017
-- License   :  BSD3
-- Maintainer:  sam@sentenai.com
-- Stability :  experimental
--
-- The agent controls the movement of a character in a grid world. Some tiles of
-- the grid are walkable, and others lead to the agent falling into the water.
-- Additionally, the movement direction of the agent is uncertain and only
-- partially depends on the chosen direction. The agent is rewarded for
-- finding a walkable path to a goal tile.
--
-- https://gym.openai.com/envs/Blackjack-v0
--------------------------------------------------------------------------------
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE InstanceSigs #-}
module Environments.Gym.ToyText.BlackjackV0
  ( I.Runner
  , Hand(..)
  , Card(..)
  , Action(..)
  , toVector
  , Environment
  , EnvironmentT
  , Environments.Gym.ToyText.BlackjackV0.runEnvironment
  , Environments.Gym.ToyText.BlackjackV0.runEnvironmentT
  , Environments.Gym.ToyText.BlackjackV0.runDefaultEnvironment
  , Environments.Gym.ToyText.BlackjackV0.runDefaultEnvironmentT
  , Action(..)
  ) where

import Control.Monad.IO.Class
import Control.Exception.Safe
import Data.Hashable
import Data.Vector (Vector)
import GHC.Generics
import qualified GHC.Exts as Exts

import Control.MonadEnv
import Environments.Gym.Internal hiding (runEnvironment)
import Environments.Blackjack (Action(..), Hand(..), Card(..))
import qualified Environments.Blackjack as B
import qualified Environments.Gym.Internal as I

import qualified Data.Vector as V
import Data.Aeson
import Data.Aeson.Types
import OpenAI.Gym (GymEnv(BlackjackV0))
import Servant.Client as X (BaseUrl)
import Network.HTTP.Client as X (Manager)


-- | The current position of the agent on the frozen lake
-- newtype StateFL = Position { unPosition :: Int }
--   deriving (Show, Eq, Generic, Ord, Hashable)

-- | Convert 'StateFL' to a computable type
toVector :: Hand Card -> Vector Int
toVector = V.fromList . fmap fromEnum . Exts.toList

-- | Build a BlackjackV0 state, throwing if the position is out of bounds.
-- mkStateFL :: MonadThrow m => Int -> m StateFL
-- mkStateFL i
--   | i < 16 && i >= 0 = pure $ Position i
--   |        otherwise = throwString $ "no state exists for " ++ show i

instance FromJSON Card where
  parseJSON :: Value -> Parser Card
  parseJSON (Number n) = pure $ toEnum (truncate n)
  parseJSON invalid    = typeMismatch "Card" invalid


instance (FromJSON c) => FromJSON (Hand c) where
  parseJSON :: Value -> Parser (Hand c)
  parseJSON (Array a)  = (Exts.fromList . V.toList) <$> V.mapM parseJSON a
  parseJSON invalid    = typeMismatch "Hand" invalid

instance ToJSON Action where
  toJSON :: Action -> Value
  toJSON = toJSON . fromEnum

-- ========================================================================= --
-- | Alias to 'Environments.Gym.Internal.GymEnvironmentT' with BlackjackV0 type dependencies
type EnvironmentT t = GymEnvironmentT (Hand Card, Card) Action t

-- | Alias to 'EnvironmentT' in IO
type Environment = EnvironmentT IO

-- | Alias to 'Environments.Gym.Internal.runEnvironmentT'
runEnvironmentT :: MonadIO t => Manager -> BaseUrl -> I.RunnerT (Hand Card, Card) Action t x
runEnvironmentT = I.runEnvironmentT BlackjackV0

-- | Alias to 'Environments.Gym.Internal.runEnvironment' in IO
runEnvironment :: Manager -> BaseUrl -> I.RunnerT (Hand Card, Card) Action IO x
runEnvironment = I.runEnvironmentT BlackjackV0

-- | Alias to 'Environments.Gym.Internal.runDefaultEnvironmentT'
runDefaultEnvironmentT :: MonadIO t => I.RunnerT (Hand Card, Card) Action t x
runDefaultEnvironmentT = I.runDefaultEnvironmentT BlackjackV0

-- | Alias to 'Environments.Gym.Internal.runDefaultEnvironment' in IO
runDefaultEnvironment :: I.RunnerT (Hand Card, Card) B.Action IO x
runDefaultEnvironment = I.runDefaultEnvironmentT BlackjackV0

instance (MonadThrow t, MonadIO t) => MonadEnv (EnvironmentT t) (Hand Card, Card) B.Action Reward where
  reset = I._reset
  step = I._step

