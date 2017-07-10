{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Zoo.Gridworld where

import Zoo.Prelude

newtype Gridworld x = Gridworld { unGridworld :: StateT (Obj, [Obj]) IO x }
  deriving
    ( Functor
    , Applicative
    , Monad
    , MonadState (Obj, [Obj])
    )


data Directions
  = Four
  | Two

data UDLR = ActUp | ActDown | ActLeft | ActRight

data GridworldConf = GridworldConf
  { sizeX   :: Int
  , sizeY   :: Maybe Int
  , actions :: Directions
  , partial :: Bool
  }

data Obj = Obj
  { xPos  :: Int
  , yPos  :: Int
  , value :: Int
  }

mkGridworld :: GridworldConf -> Gridworld x
mkGridworld conf = Gridworld $ do
  undefined

getHero :: Gridworld Obj
getHero = Gridworld $ fst <$> get

setHero :: Obj -> Gridworld ()
setHero h = Gridworld $ do
  env <- snd <$> get
  put (h, env)

step :: GridworldConf -> UDLR -> Gridworld ()
step GridworldConf{sizeX, sizeY} a = do
  let sizeY' = fromMaybe sizeX sizeY
  hero <- getHero
  let heroX    = xPos hero
      heroY    = yPos hero
  case a of
    ActUp    -> if heroY >= 1          then moveUp    hero else heroUnmoved
    ActDown  -> if heroY <= sizeY' - 2 then moveDown  hero else heroUnmoved
    ActLeft  -> if heroX >= 1          then moveLeft  hero else heroUnmoved
    ActRight -> if heroX <= sizeX  - 2 then moveRight hero else heroUnmoved

  where
    moveUp    hero = setHero $ hero { yPos = yPos hero - 1 }
    moveDown  hero = setHero $ hero { yPos = yPos hero + 1 }
    moveLeft  hero = setHero $ hero { xPos = xPos hero - 1 }
    moveRight hero = setHero $ hero { xPos = xPos hero + 1 }
    heroUnmoved = undefined
