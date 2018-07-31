-------------------------------------------------------------------------------
-- |
-- Module    :  Reinforce.Gridworlds.Foursquare
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- A simple four-square gridworld with the following description:
--
--    ,--------.-------.
--    | P01    |  P11  |
--    |        |       |
--    :-------+++------:
--    | P00   ||| P10  |
--    | start ||| end  |
--    `-------'''------'
--
-------------------------------------------------------------------------------

{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Reinforce.Gridworlds.Foursquare where

import Data.Word (Word)
import Data.Proxy (Proxy(Proxy))
import GHC.TypeLits (KnownNat, Nat, natVal)
import Control.MonadEnv (MonadEnv(reset, step), Obs(Next, Done), Initial(Initial))
import Control.Monad.State (StateT, get, put, runStateT)
import Control.Monad.IO.Class (MonadIO, liftIO)
import System.Random.MWC (GenIO, uniformR)

-- | World monad
newtype World x = World { getWorld :: StateT (GenIO, Double, (Word, Word)) IO x }
  deriving (Functor, Applicative, Monad, MonadIO)

data ArrowKeys
  = AUp
  | ADown
  | ALeft
  | ARight
  deriving (Eq, Ord, Show, Enum, Bounded)


-- | Discrete action space
newtype Discrete (n::Nat) = Discrete Word
  deriving (Eq, Ord, Show)

instance KnownNat n => Bounded (Discrete n) where
  minBound = Discrete 0
  maxBound = Discrete . fromIntegral $ natVal (Proxy :: Proxy n) - 1

instance KnownNat n => Enum (Discrete n) where
  fromEnum (Discrete x) = fromIntegral x
  toEnum i = unsafeDiscrete (fromIntegral i)

size :: forall n .  KnownNat n => Discrete n -> Int
size _ = fromIntegral mx - fromIntegral mn + 1
  where
    Discrete mn = minBound :: Discrete n
    Discrete mx = maxBound :: Discrete n

-- | make a discrete action
mDiscrete :: forall n . KnownNat n => Word -> Maybe (Discrete n)
mDiscrete a
  | fromIntegral a < natVal (Proxy :: Proxy n) = Just (Discrete a)
  | otherwise = Nothing

-- | brazenly make a discrete action without
unsafeDiscrete :: forall n . KnownNat n => Word -> Discrete n
unsafeDiscrete w =
  case mDiscrete w of
    Just a  -> a
    Nothing -> error $
      "called unsafeDiscrete with a number >= "
        ++ show (size (maxBound :: Discrete n))

-- | run a world with a given probability with how "slippery" each action is.
runWorld :: GenIO -> Double -> World x -> IO (x, (GenIO, Double, (Word, Word)))
runWorld g p (World ms)
  | p <= 0 || p > 1 = error "p must be bounded by (0, 1]"
  | otherwise       = runStateT ms (g, p, (0, 0))

-- | evaluate a world with a given probability with how "slippery" each action is.
evalWorld :: GenIO -> Double -> World x -> IO x
evalWorld g p w = fst <$> runWorld g p w

instance MonadEnv World (Word, Word) (Discrete 4) Double where
  reset = World $ do
    let pos = (0,0)
    (g, p, _) <- get
    put (g, p, pos)
    pure (Initial pos)

  step w = World $ do
    (g, pslip, pos) <- get

    w' <- liftIO $ do
           p' <- uniformR (0,1) g
           if p' <= pslip
           then unsafeDiscrete . toEnum <$> uniformR (0,4) g
           else pure w

    case (toEnum $ fromEnum w', pos) of
      (ADown,  (1, 1)) -> pure $ Done 10 (Just (1, 0)) -- you just won!
      (ARight, (0, 0)) -> pure $ Next (-1) (0, 0)      -- trying to walk into wall

      (AUp,    (x, 1)) -> pure $ Next (-1) (x, 1)   -- already at top
      (AUp,    (x, y)) -> pure $ Next (-1) (x, y+1)

      (ADown,  (x, 0)) -> pure $ Next (-1) (x, 0)   -- already at bottom
      (ADown,  (x, y)) -> pure $ Next (-1) (x, y-1)

      (ARight, (1, y)) -> pure $ Next (-1) (1,   y) -- already at right
      (ARight, (x, y)) -> pure $ Next (-1) (x+1, y)

      (ALeft,  (0, y)) -> pure $ Next (-1) (0,   y) -- already at right
      (ALeft,  (x, y)) -> pure $ Next (-1) (x-1, y)


