-------------------------------------------------------------------------------
-- |
-- Module    :  Reinforce.Gridworlds
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Reinforce.Gridworlds where

import Control.MonadEnv
import Data.Word (Word)
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HM

data Tile
  = Finish   Word Word
  | Start    Word Word
  | Empty    Word Word
  | ThinWall Tile Tile

-- | a gridworld like:
--      0     1
--   .-----.-----.
-- 0 |     |     |
--   ;-----;-----;
-- 1 |  S ||| T  |
--   `-----'-----`
slipperyQuarters :: [Tile]
slipperyQuarters =
  [ Start  1 0
  , Finish 1 1
  , Empty  0 0
  , Empty  0 1
  , ThinWall s f
  ]
 where
  s = Start  1 0
  f = Finish 1 1


