module Zoo.Gridworld where

class GridworldEnv(gym.Env):

    class GridObj:
        def __init__(self, size, intensity, channel, reward, name):
            self.x         = coordinates[0]
            self.y         = coordinates[1]
            self.size      = size
            self.intensity = intensity
            self.channel   = channel
            self.reward    = reward
            self.name      = name

    metadata = {'render.modes': ['human']}

newtype Gridworld x = Gridworld { unGridworld :: StateT (Obj, [Obj]) IO x }
  deriving ()

data Directions
  = Four
  | Two

data UDLR = Up | Down | Left | Right
data LR   =             Left | Right

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
mkGridworld conf = do
  undefined

getHero :: Gridworld Obj
getHero = fst <$> get

setHero :: Obj -> Gridworld ()
setHero h = do
  env <- snd <$> get
  set (h, env)

4step :: GridworldConf -> UDLR -> StateT Gridworld IO x
4step GridworldConf{sizeX, sizeY} a =
  hero <- getHero
  let heroX    = xPos hero
      heroY    = yPos hero
  case a of
    Up    -> if heroY >= 1         then moveUp    hero else heroUnmoved
    Down  -> if heroY <= sizeY - 2 then moveDown  hero else heroUnmoved
    Left  -> if heroX >= 1         then moveLeft  hero else heroUnmoved
    Right -> if heroX <= sizeX - 2 then moveRight hero else heroUnmoved

  where
    moveUp    hero = setHero $ hero { yPos = yPos hero - 1 }
    moveDown  hero = setHero $ hero { yPos = yPos hero + 1 }
    moveLeft  hero = setHero $ hero { xPos = xPos hero - 1 }
    moveRight hero = setHero $ hero { xPos = xPos hero + 1 }
    heroUnmoved = undefined
