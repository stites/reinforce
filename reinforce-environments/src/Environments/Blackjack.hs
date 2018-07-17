-------------------------------------------------------------------------------
-- |
-- Module    :  Environments.Blackjack
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Simple blackjack environment
--
-- Blackjack is a card game where the goal is to obtain cards that sum to as
-- near as possible to 21 without going over.  They're playing against a fixed
-- dealer.
-- Face cards (Jack, Queen, King) have point value 10.
-- Aces can either count as 11 or 1, and it's called 'usable' at 11.
-- This game is placed with an infinite deck (or with replacement).
-- The game starts with each (player and dealer) having one face up and one
-- face down card.
-- The player can request additional cards (hit=1) until they decide to stop
-- (stick=0) or exceed 21 (bust).
-- After the player sticks, the dealer reveals their facedown card, and draws
-- until their sum is 17 or greater.  If the dealer goes bust the player wins.
-- If neither player nor dealer busts, the outcome (win, lose, draw) is
-- decided by whose sum is closer to 21.  The reward for winning is +1,
-- drawing is 0, and losing is -1.
-- The observation of a 3-tuple of: the players current sum,
-- the dealer's one showing card (1-10 where 1 is ace),
-- and whether or not the player holds a usable ace (0 or 1).
-- This environment corresponds to the version of the blackjack problem
-- described in Example 5.1 in Reinforcement Learning: An Introduction
-- by Sutton and Barto (1998).
-- http://incompleteideas.net/sutton/book/the-book.html
-------------------------------------------------------------------------------
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
module Environments.Blackjack where

import Control.MonadEnv

import GHC.Generics (Generic)
import System.Random.MWC (Seed, save, restore, GenIO, uniformR)
import Data.Hashable (Hashable)
import Data.HashSet (HashSet)
import Control.Monad.State.Strict
import qualified Data.HashSet as HS

data Action = Stay | Hit

data Hand
  = Start (Card, Card)
  | Extended Card Hand

instance Eq Hand where
  a == b = asSet a == asSet b

instance Ord Hand where
  compare a b =
    case (va > vb, va < vb) of
      (False, True) -> LT
      (True, False) -> GT
      _ -> EQ
    where
      (va, vb) = (score a, score b)

asSet :: Hand -> HashSet Card
asSet = HS.fromList . asList

asList :: Hand -> [Card]
asList = \case
  Start (a, b) -> [a, b]
  Extended c h -> c:asList h

first :: Hand -> Card
first (Start (a, _)) = a
first (Extended a _) = a

data Card
  = Ace
  | C2
  | C3
  | C4
  | C5
  | C6
  | C7
  | C8
  | C9
  | C10
  | Jack
  | Queen
  | King
  deriving stock    (Eq, Show, Enum, Bounded, Generic)
  deriving anyclass (Hashable)

value :: Card -> Int
value c
  | fromEnum c > 10 = 10
  | otherwise = fromEnum c

-- 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck :: [Card]
deck = [minBound .. maxBound]

drawCard :: GenIO -> IO Card
drawCard g = toEnum <$> uniformR (value (minBound::Card), value (maxBound::Card)) g

drawHand :: GenIO -> IO (Card, Card)
drawHand g = (,) <$> drawCard g <*> drawCard g

-- Does this hand have a usable ace?
usableAce :: Hand -> Bool
usableAce h = Ace `elem` cs && sum (fmap value cs) <= 21
  where
    cs = asList h

-- Return current hand total
sumHand :: Hand -> Int
sumHand h = sum (value <$> asList h) + (if usableAce h then 10 else 0)

isBust :: Hand -> Bool
isBust h = sumHand h > 21

score :: Hand -> Int
score h =
  if isBust h
  then 0
  else sumHand h

isNatural :: Hand -> Bool
isNatural = \case
  Start (Ace, C10) -> True
  Start (C10, Ace) -> True
  _                -> False

newtype BlackJack x = BlackJack (StateT (Seed, Hand, Hand) IO x)
  deriving stock (Functor)
  deriving newtype (Applicative, Monad, MonadIO, MonadState (Seed, Hand, Hand))

instance MonadEnv BlackJack (Hand, Card) Action Float where
  -- slightly redundant but this lets us reset as many times as we want.
  reset = do
    ( _, d) <- lifted (fmap Start . drawHand)
    (s', h) <- lifted (fmap Start . drawHand)
    put (s', d, h)
    pure $ Initial (h, first d)

  step Hit = do
    (_, d, h) <- get
    (s', h') <- lifted (fmap (`Extended` h) . drawCard)
    put (s', d, h')
    if isBust h'
    then pure $ Done (-1) Nothing
    else pure $ Next 0 (h', first d)

  step Stay = do
    (_, d, h) <- get
    (s', d') <- lifted (`rollout` d)
    let r = fromEnum $ compare (score d') (score h)
    put (s', d', h)
    if isNatural d && isNatural h && r == 1
    then pure $ Done 1.5 Nothing
    else pure $ Done (fromIntegral r) Nothing
   where
    rollout :: GenIO -> Hand -> IO Hand
    rollout g h =
      if sumHand h < 17
      then drawCard g >>= \c -> rollout g (Extended c h)
      else pure h

lifted :: (GenIO -> IO x) -> BlackJack (Seed, x)
lifted randact = do
  (s, _, _) <- get
  g  <- liftIO $ restore s
  x  <- liftIO $ randact g
  s' <- liftIO $ save g
  pure (s', x)

