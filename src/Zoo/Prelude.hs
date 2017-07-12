module Zoo.Prelude
  ( module X
  , UVector
  , GVector
  , identity
  , (//)
  , asFloats
  , liftTF
  , printIO
  , tdf2vec
  , lastN
  , divisibleBy
  , oneHot
  , getState
  , putState
  ) where

import Prelude                       as X hiding (id)
import Control.Applicative           as X
import Control.Arrow                 as X
import Control.Exception.Safe        as X
import Control.Monad                 as X
import Control.MonadEnv              as X (Initial(..), Obs(..))
import Control.MonadMWCRandom        as X (MWCRandT(..), MWCRand(..))
import Control.Monad.IO.Class        as X (liftIO, MonadIO)
import Control.Monad.Reader.Class    as X (MonadReader)
import Control.Monad.State.Class     as X (MonadState)
import Control.Monad.Trans           as X
import Control.Monad.Trans.Except    as X hiding (liftCallCC, liftListen, liftPass)
import Control.Monad.Trans.Maybe     as X (MaybeT(..))
import Control.Monad.Trans.Reader    as X hiding (liftCallCC, liftCatch)
import Control.Monad.Trans.State     as X
import Control.DeepSeq               as X
import Data.Bitraversable            as X
import Data.DList                    as X (DList)
import Data.Function                 as X (on)
import Data.Functor.Identity         as X (Identity)
import Data.Foldable                 as X
import Data.Int                      as X (Int32, Int64)
import Data.List                     as X (genericLength, genericTake, groupBy)
import Data.Maybe                    as X
import Data.Monoid                   as X
import Data.Traversable              as X
import Data.Tuple                    as X
import Data.Type.Combinator          as X
import Data.Type.Index               as X
import Data.Type.Product             as X (Prod(..), only, pattern (::<))
import Data.Vector                   as X (Vector)
import Debug.Trace                   as X
import GHC.Generics                  as X (Generic)
import GHC.TypeLits                  as X
import GHC.IO                        as X (evaluate)
import Numeric.Backprop              as X (Op, BPOp, BPOpI, BVar, (#<~), (~$), (.$), (-$), gTuple)
import Numeric.LinearAlgebra.Static  as X (L, R, (#>), outer, tr, (<.>), konst, extract)
import TensorFlow.Core               as X (Tensor, Shape, Build, TensorData, Session, Ref, Value, ControlNode)
import Text.Printf                   as X


-- ========================================================================= --
-- Prelude customizations

import Data.Vector.Unboxed as UV (Vector)
import Data.Vector.Generic as G (Vector)
import qualified TensorFlow.Core as TF
import qualified Prelude         as P
import qualified Data.Vector     as V

type UVector = UV.Vector
type GVector = G.Vector

identity :: a -> a
identity = P.id

(//) :: Int -> Int -> Float
(//) = (/) `on` fromIntegral

asFloats :: Functor f => f Int -> f Float
asFloats = fmap fromIntegral

liftTF :: (MonadTrans t0, MonadTrans t1, Monad (t1 Session)) => Session a -> (t0 (t1 Session)) a
liftTF = lift . lift

printIO :: (Show s, MonadIO io) => s -> io ()
printIO = liftIO . print

tdf2vec :: TensorData Float -> V.Vector Float
tdf2vec = TF.decodeTensorData

lastN :: Int -> [x] -> [x]
lastN c xs = drop (length xs - c) xs

divisibleBy :: Int -> Int -> Bool
divisibleBy a b = a `mod` b == 0 && a /= 0

oneHot :: Num f => Int -> Int -> V.Vector f
oneHot len x
  | len > x   = V.replicate len 0 `V.unsafeUpd` [(fromIntegral x, 1)]
  | otherwise = error "cannot one-hot encode a int that is greater than size of vector"

getState :: Monad m => StateT s m s
getState = get

putState :: Monad m => s -> StateT s m ()
putState = put
