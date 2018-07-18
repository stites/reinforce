module Main where

import Reinforce.Agents
import Reinforce.Agents.QTable
import Reinforce.Algorithms.QLearning

import Environments.Gym.ToyText.BlackjackV0 (runDefaultEnvironment)

main :: IO ()
main = do
  x <- runDefaultEnvironment False $
         runQTable defaultConfigs (Left 0.85) $
           runLearner (Just 10) (Just 10)
             rolloutQLearning
  print x

