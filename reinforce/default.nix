{ mkDerivation, base, mtl, mwc-random, primitive, safe-exceptions
, statistics, stdenv, text, transformers, vector
}:
mkDerivation {
  pname = "reinforce";
  version = "0.0.1.0";
  src = ./.;
  libraryHaskellDepends = [
    base mtl mwc-random primitive safe-exceptions statistics text
    transformers vector
  ];
  homepage = "https://github.com/Sentenai/reinforce#readme";
  description = "Reinforcement learning in Haskell";
  license = stdenv.lib.licenses.bsd3;
}
