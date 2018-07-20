-------------------------------------------------------------------------------
-- |
-- Module    :  Main
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Monte Carlo methods
-------------------------------------------------------------------------------
module Main where

import Reinforce.Agents
import Reinforce.Agents.QTable
import Reinforce.Algorithms.QLearning

import Control.MonadEnv
import Control.Monad.IO.Class
import Control.Monad
import Environments.Blackjack

-- | Every-visit MC Prediction
--
-- Average the returns following all visits to each state-action pair, in all episodes
everyVisit = undefined

-- | First-visit MC Prediction
--
-- For each episode, we only consider the first visit to the state-action pair. Pseudocode:
--
-- Input: policy \pi, positive integer _num\_episodes_
-- Output: value
firstVisit = undefined


main :: IO ()
main = do
  void . runEnvironment $ do
    reset >>= liftIO . print
    step Hit >>= liftIO . print
    step Hit >>= liftIO . print
    step Hit >>= liftIO . print
    step Hit >>= liftIO . print
    pure ()
  pure ()



