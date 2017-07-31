module Zoo.Internal where

import Zoo.Prelude

report :: (Show r, Fractional r, MonadIO io) => [r] -> io ()
report rwds = do
  let per = sum rwds / fromIntegral (genericLength rwds)
  let last50Per = sum (lastN 50 rwds) / 50
  printIO $ "Percent successful episodes: " ++ show per       ++ "%"
  printIO $ "Percent successful last 50 : " ++ show last50Per ++ "%"


