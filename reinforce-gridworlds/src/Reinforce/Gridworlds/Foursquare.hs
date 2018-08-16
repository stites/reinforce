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
import Control.Monad.Trans.State (StateT, get, put, runStateT)
import Control.Monad.IO.Class (MonadIO, liftIO)
import System.Random.MWC (GenIO, uniformR)

import Reinforce.Spaces.Action

-- | World monad
newtype World x = World { getWorld :: StateT (GenIO, Double, (Word, Word)) IO x }
  deriving (Functor, Applicative, Monad, MonadIO)

-- | run a world with a given probability with how "slippery" each action is.
runWorld :: GenIO -> Double -> World x -> IO (x, (GenIO, Double, (Word, Word)))
runWorld g p (World ms)
  | p <= 0 || p > 1 = error "p must be bounded by (0, 1]"
  | otherwise       = runStateT ms (g, p, (0, 0))

-- | evaluate a world with a given probability with how "slippery" each action is.
evalWorld :: GenIO -> Double -> World x -> IO x
evalWorld g p w = fst <$> runWorld g p w

instance MonadEnv World (Word, Word) (Discrete1d 4) Double where
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
           then unsafeDiscrete1d . toEnum <$> uniformR (0,4) g
           else pure w

    pure $
      case (toEnum $ fromEnum w', pos) of
        (ADown,  (1, 1)) -> Done 10 (Just (1, 0)) -- you just won!
        (ARight, (0, 0)) -> Next (-1) (0, 0)      -- trying to walk into wall

        (AUp,    (x, 1)) -> Next (-1) (x, 1)   -- already at top
        (AUp,    (x, y)) -> Next (-1) (x, y+1)

        (ADown,  (x, 0)) -> Next (-1) (x, 0)   -- already at bottom
        (ADown,  (x, y)) -> Next (-1) (x, y-1)

        (ARight, (1, y)) -> Next (-1) (1,   y) -- already at right
        (ARight, (x, y)) -> Next (-1) (x+1, y)

        (ALeft,  (0, y)) -> Next (-1) (0,   y) -- already at right
        (ALeft,  (x, y)) -> Next (-1) (x-1, y)



