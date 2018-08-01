{-# LANGUAGE DataKinds #-}
{-# LANGUAGE LambdaCase #-}
module Main where

import Control.Monad (void)
import Control.Monad.IO.Class
import Control.MonadEnv
import Reinforce.Gridworlds.Foursquare
import System.Random.MWC
import Data.HashMap.Strict (HashMap)
import Data.List (maximumBy)
import qualified Data.HashMap.Strict as HM
import Data.Function (on)

main :: IO ()
main = do
  print ""
  withSystemRandom $ \g ->
    void . runWorld g 0.3 $ agent

type ActionRewards = HashMap (Discrete 4) Double
type QTable = HashMap (Word, Word) ActionRewards

agent = undefined

rolloutEpisode :: GenIO -> QTable -> World QTable
rolloutEpisode g qtable0
  = reset >>= \case
    EmptyEpisode -> pure qtable0
    Initial s -> greedyStepper s qtable0
 where
  greedyStepper :: (Word, Word) -> QTable -> World QTable
  greedyStepper s qtable = do
    a <- liftIO $ choose
    undefined
    where
      choose :: IO (Discrete 4)
      choose = case HM.lookup s qtable of
        Nothing  -> toEnum <$> uniformR (0, 4) g
        Just ars -> pure $ fst $ maximumBy (compare `on` fst) (HM.toList ars)

