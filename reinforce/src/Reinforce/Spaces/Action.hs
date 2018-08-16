-------------------------------------------------------------------------------
-- |
-- Module    :  Spaces.Action
-- Copyright :  (c) Sentenai 2017
-- License   :  BSD3
-- Maintainer:  sam@sentenai.com
-- Stability :  experimental
-- Portability: non-portable
--
-- typeclass for a discrete action space, as well as helper functions
-------------------------------------------------------------------------------
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
module Reinforce.Spaces.Action
  where
  -- ( DiscreteActionSpace(..)
  -- -- , oneHot
  -- , oneHot'
  -- , allActions
  -- , randomChoice
  -- ) where

import Control.Monad.IO.Class
-- import Numeric.LinearAlgebra.Static (R)
-- import qualified Numeric.LinearAlgebra.Static as LA

import Control.MonadMWCRandom
import GHC.TypeLits
import Data.Proxy
import Data.Vector (Vector)
import qualified Data.Vector as V

data ArrowKeys
  = AUp
  | ADown
  | ALeft
  | ARight
  deriving (Eq, Ord, Show, Enum, Bounded)

numels :: forall n . (Enum n, Bounded n) => Proxy n -> Int
numels _ = fromIntegral mx - fromIntegral mn + 1
  where
    mn = fromEnum (minBound :: n)
    mx = fromEnum (maxBound :: n)

class DimList b where
  dimList :: b -> [Word]


randomChoice :: forall n m . (Enum n, Bounded n, MonadIO m, MonadMWCRandom m) => m n
randomChoice = do
  p :: Float <- uniform
  pure . toEnum . truncate $ fromIntegral (fromEnum (maxBound :: n)) * p


-- ========================================================================= --

-- * Discrete 1d

-- | Discrete action space
newtype Discrete1d (n::Nat) = Discrete1d Word
  deriving (Eq, Ord, Show)

instance KnownNat n => Bounded (Discrete1d n) where
  minBound = Discrete1d 0
  maxBound = Discrete1d . fromIntegral $ natVal (Proxy :: Proxy n) - 1

instance KnownNat n => Enum (Discrete1d n) where
  fromEnum (Discrete1d x) = fromIntegral x
  toEnum i = unsafeDiscrete1d (fromIntegral i)

instance KnownNat n => DimList (Discrete1d n) where
  dimList (Discrete1d w) = [w]

-- | make a discrete action
mDiscrete1d :: forall n . KnownNat n => Word -> Maybe (Discrete1d n)
mDiscrete1d a
  | fromIntegral a < natVal (Proxy :: Proxy n) = Just (Discrete1d a)
  | otherwise = Nothing

-- | brazenly make a discrete action without regard for your program's safety
unsafeDiscrete1d :: forall n . KnownNat n => Word -> Discrete1d n
unsafeDiscrete1d w =
  case mDiscrete1d w of
    Just a  -> a
    Nothing -> error $
      "called unsafeDiscrete1d with a number outside the boundaries of "
        ++ show (maxBound :: Discrete1d n)


-- ========================================================================= --

-- * Discrete 2d

-- | Discrete action space
newtype Discrete2d (n0::Nat) (n1::Nat) = Discrete2d (Word, Word)
  deriving (Eq, Ord, Show)

instance (KnownNat n0, KnownNat n1) => Bounded (Discrete2d n0 n1) where
  minBound = Discrete2d (0,0)
  maxBound = Discrete2d (asMx (Proxy @ n0),asMx (Proxy @ n1))
    where
       asMx :: KnownNat x => Proxy x -> Word
       asMx p = fromIntegral $ natVal p - 1

instance DimList (Discrete2d n0 n1) where
  dimList (Discrete2d (n0, n1)) = [n0, n1]

instance (KnownNat n0, KnownNat n1) => Enum (Discrete2d n0 n1) where
  fromEnum d = product ((+1) . fromIntegral <$> dimList d) - 1
  toEnum i = unsafeDiscrete2d (fromIntegral (fromIntegral i `rem` natVal (Proxy @n0))) (fromIntegral ((fromIntegral i `rem` natVal (Proxy @n1))+1))

-- | make a discrete action
mDiscrete2d :: forall n0 n1 . (KnownNat n0, KnownNat n1) => Word -> Word -> Maybe (Discrete2d n0 n1)
mDiscrete2d a b
  |  fromIntegral a < natVal (Proxy :: Proxy n0)
  && fromIntegral b < natVal (Proxy :: Proxy n1) = Just (Discrete2d (a, b))

  | otherwise = Nothing

-- | brazenly make a discrete action without regard for your program's safety
unsafeDiscrete2d :: forall n0 n1 . (KnownNat n0, KnownNat n1) => Word -> Word -> Discrete2d n0 n1
unsafeDiscrete2d w0 w1 =
  case mDiscrete2d w0 w1 of
    Just a  -> a
    Nothing -> error $
      "called unsafeDiscrete2d with a number outside the boundaries of "
        ++ show (maxBound :: Discrete2d n0 n1)



