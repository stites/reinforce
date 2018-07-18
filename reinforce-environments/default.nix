{ mkDerivation, aeson, base, containers, dlist, hashable, mtl
, mwc-random, reinforce, safe-exceptions, statistics, stdenv
, transformers, unordered-containers, vector
}:
mkDerivation {
  pname = "reinforce-environments";
  version = "0.0.1.0";
  src = ./.;
  libraryHaskellDepends = [
    aeson base containers dlist hashable mtl mwc-random reinforce
    safe-exceptions statistics transformers unordered-containers vector
  ];
  homepage = "https://github.com/Sentenai/reinforce#readme";
  description = "Reinforcement learning in Haskell";
  license = stdenv.lib.licenses.bsd3;
}
