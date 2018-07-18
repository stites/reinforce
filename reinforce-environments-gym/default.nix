{ mkDerivation, aeson, base, dlist, gym-http-api, hashable
, http-client, mtl, reinforce, reinforce-environments
, safe-exceptions, servant-client, stdenv, text, transformers
, vector
}:
mkDerivation {
  pname = "reinforce-environments-gym";
  version = "0.0.1.0";
  src = ./.;
  libraryHaskellDepends = [
    aeson base dlist gym-http-api hashable http-client mtl reinforce
    reinforce-environments safe-exceptions servant-client text
    transformers vector
  ];
  homepage = "https://github.com/Sentenai/reinforce#readme";
  description = "Reinforcement learning in Haskell";
  license = stdenv.lib.licenses.bsd3;
}
