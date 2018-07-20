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
{-# LANGUAGE ScopedTypeVariables #-}
{- LANGUAGE OverloadedLists #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleInstances #-}
module Environments.Blackjack where

import Control.MonadEnv

import GHC.Generics (Generic)
import GHC.Exts (IsList(..))
import System.Random.MWC (Seed, save, restore, GenIO, uniformR, withSystemRandom)
import Data.Hashable (Hashable)
import Data.HashSet (HashSet)
import Control.Monad.State.Strict
import qualified Data.HashSet as HS

data Action = Stay | Hit
  deriving (Eq, Show, Enum, Bounded, Generic, Hashable)

data Hand c
  = Start (c, c)
  | Extended c (Hand c)
  deriving (Show, Generic, Hashable)

instance Eq c => Eq (Hand c) where
  a == b = toList a == toList b

instance (Eq c, Enum c) => Ord (Hand c) where
  compare a b =
    case (va > vb, va < vb) of
      (False, True) -> LT
      (True, False) -> GT
      _ -> EQ
    where
      (va, vb) = (score a, score b)

instance IsList (Hand c) where
  type Item (Hand c) = c

  toList :: Hand c -> [c]
  toList = \case
    Start (a, b) -> [a, b]
    Extended c h -> c:toList h

  fromList :: [c] -> Hand c
  fromList [a,b]  = Start (a, b)
  fromList (c:cs) = Extended c (fromList cs)
  fromList _      = error "hands must have at least two items!"


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

value :: Enum e => e -> Int
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
usableAce :: (Eq e, Enum e) => Hand e -> Bool
usableAce h = toEnum 0 `elem` cs && sum (fmap value cs) <= 21
  where
    cs = toList h

-- Return current hand total
sumHand :: (Eq e, Enum e) => Hand e -> Int
sumHand h = sum (value <$> toList h) + (if usableAce h then 10 else 0)

isBust :: (Eq e, Enum e) => Hand e -> Bool
isBust h = sumHand h > 21

score :: (Eq e, Enum e) => Hand e -> Int
score h =
  if isBust h
  then 0
  else sumHand h

isNatural :: Hand Card -> Bool
isNatural = \case
  Start (Ace, C10) -> True
  Start (C10, Ace) -> True
  _                -> False

newtype BlackJack x = BlackJack (StateT (Seed, Hand Card, Hand Card) IO x)
  deriving stock (Functor)
  deriving newtype (Applicative, Monad, MonadIO, MonadState (Seed, Hand Card, Hand Card))

instance MonadEnv BlackJack (Hand Card, Card) Action Float where
  -- slightly redundant but this lets us reset as many times as we want.
  reset = do
    ( _, d) <- lifted (fmap Start . drawHand)
    (s', h) <- lifted (fmap Start . drawHand)
    put (s', d, h)
    pure $ Initial (h, head $ toList d)

  -- step :: Action -> BlackJack (Obs Float (Hand', Card))
  step Hit = do
    (_, d::Hand Card, h) <- get
    (s', h') <- lifted (fmap (`Extended` h) . drawCard)
    put (s', d, h')
    if isBust h'
    then pure $ Done (-1) Nothing
    else pure $ Next 0 (h', head $ toList d)

  step Stay = do
    (_, d, h) <- get
    (s', d') <- lifted (`rollout` d)
    let r = fromEnum $ compare (score d') (score h)
    put (s', d', h)
    if isNatural d && isNatural h && r == 1
    then pure $ Done 1.5 Nothing
    else pure $ Done (fromIntegral r) Nothing
   where
    rollout :: GenIO -> Hand Card -> IO (Hand Card)
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

runEnvironment :: BlackJack x -> IO (x, (Hand Card, Hand Card))
runEnvironment (BlackJack program) = withSystemRandom $ \g -> do
  dealer <- Start <$> drawHand g
  player <- Start <$> drawHand g
  s <- save g
  (x, (_, d, h)) <- runStateT program (s, dealer, player)
  pure (x, (d, h))
