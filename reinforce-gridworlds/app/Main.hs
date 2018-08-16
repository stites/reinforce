{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecursiveDo #-}
module Main where

import Data.Char     (toUpper)
import Control.Monad (forever)
import System.IO     (BufferMode(..), hSetEcho, hSetBuffering, stdin)
import Control.Concurrent

import Reactive.Banana
import Reactive.Banana.Frameworks
-- import qualified Termbox as TB hiding (main)
import Termbox.Banana hiding (Event, main)
import qualified Termbox.Banana as TB (Event, Key, main)

-- blackbird combinator
(.:) :: (b -> c) -> (a0 -> a1 -> b) -> a0 -> a1 -> c
(.:) = (.) . (.)

-- Filter and transform events at the same time.
filterMapJust :: (a -> Maybe b) -> Event a -> Event b
filterMapJust = filterJust .: fmap

{-
type Octave = Int

data Pitch = PA | PB | PC | PD | PE | PF | PG
    deriving (Eq, Enum)

-- Mapping between pitch and the char responsible for it.
pitchChars :: [(Pitch, Char)]
pitchChars = [(p, toEnum $ fromEnum 'a' + fromEnum p) |
              p <- [PA .. PG]]

-- Reverse of pitchChars
charPitches :: [(Char, Pitch)]
charPitches = [(b, a) | (a, b) <- pitchChars]

data Note = Note Octave Pitch

instance Show Pitch where
    show p = case lookup p pitchChars of
        Nothing -> error "cannot happen"
        Just c  -> [toUpper c]

instance Show Note where
    show (Note o p) = show p ++ show o


-- Change the original octave by adding a number of octaves, taking
-- care to limit the resulting octave to the 0..10 range.
changeOctave :: Int -> Octave -> Octave
changeOctave d = max 0 . min 10 . (d+)

-- Get the octave change for the '+' and '-' chars.
getOctaveChange :: Char -> Maybe Int
getOctaveChange c = case c of
    '+' -> Just 1
    '-' -> Just (-1)
    _ -> Nothing


makeNetworkDescription :: AddHandler Char -> MomentIO ()
makeNetworkDescription addKeyEvent = do
    eKey <- fromAddHandler addKeyEvent

    let eOctaveChange :: Event Int
        eOctaveChange = filterMapJust getOctaveChange eKey

        foo :: Event (Octave -> Octave)
        foo = (changeOctave <$> eOctaveChange)

    bOctave <- accumB 3 foo

    let ePitch = filterMapJust (`lookup` charPitches) eKey
    bPitch <- stepper PC ePitch

    let
        bNote = Note <$> bOctave <*> bPitch
        foo = Note 0 PA

    eNoteChanged <- changes bNote
    reactimate' $ fmap (\n -> putStrLn ("Now playing " ++ show n))
                 <$> eNoteChanged

-}


maybeCharEvt :: TB.Event -> Maybe Char
maybeCharEvt = \case
    EventKey (KeyChar c) _ -> Just c
    _ -> Nothing

terminate :: TB.Event -> Maybe ()
terminate = \case
    EventKey KeyEsc _ -> Just ()
    _ -> Nothing


main :: IO ()
main = TB.main (InputModeEsc MouseModeNo) OutputModeNormal $ \keyEvent behav -> do

  let ekeypress :: Event Char
      ekeypress = filterMapJust maybeCharEvt keyEvent

      renderChar :: Char -> Scene
      renderChar c = Scene (set 10 10 (Cell c white black)) NoCursor

      emptyScene :: Scene
      emptyScene = Scene mempty NoCursor

      eTerminate :: Event ()
      eTerminate = filterMapJust terminate keyEvent

      renderStr :: [Char] -> Cells
      renderStr cs = foldMap (\(i,c) -> set (i+10) 10 (Cell c white black)) $ zip [0..] cs

  rec bTyped <- stepper [] (pure (\memo c -> memo ++ [c]) <*> bTyped <@> ekeypress)

  let bRendered :: Behavior Scene
      bRendered = (`Scene` NoCursor) .  renderStr <$> bTyped

  pure (bRendered, eTerminate)




