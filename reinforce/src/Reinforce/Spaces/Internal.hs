-------------------------------------------------------------------------------
-- |
-- Module    :  Reinforce.Spaces.Internal
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-- 
-- Data types which may or may not be used in spaces and actions (has yet to be
-- determined).
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE InstanceSigs #-}
module Reinforce.Spaces.Internal
  ( Discrete
  , discrete
  , fromDiscrete
  , unsafeDiscrete
  ) where

import GHC.Word (Word)
import GHC.TypeLits (natVal, Nat, KnownNat)
import Data.Proxy (Proxy(..))
import Data.Maybe (fromJust)

-- | A discrete space representation. Holds a type-level nat which represents
-- the size of the space and a runtime representation of the position of that space.
newtype Discrete (size::Nat) = Discrete Word
  deriving (Eq, Ord, Show)

instance KnownNat sz => Bounded (Discrete sz) where
  minBound = Discrete 0
  maxBound = Discrete (fromIntegral $ natVal (Proxy :: Proxy sz) - 1)

-- | Get the representation of a descrete space value.
fromDiscrete :: Discrete sz -> Word
fromDiscrete (Discrete w) = w

-- | Make a descrete space value.
discrete :: forall size i . (KnownNat size, Integral i) => i -> Maybe (Discrete size)
discrete i
  | fromIntegral i <= fromDiscrete (maxBound :: Discrete size) = Just (Discrete $ fromIntegral i)
  | otherwise = Nothing

-- | Make a descrete space value with arrogant confidence.
unsafeDiscrete :: (KnownNat size, Integral i) => i -> Discrete size
unsafeDiscrete = fromJust . discrete


