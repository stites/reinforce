{ mkDerivation, base, mwc-random, reinforce, reinforce-algorithms
, reinforce-environments, reinforce-environments-gym, stdenv
, vector
}:
mkDerivation {
  pname = "reinforce-zoo";
  version = "0.0.1.0";
  src = ./.;
  isLibrary = false;
  isExecutable = true;
  executableHaskellDepends = [
    base mwc-random reinforce reinforce-algorithms
    reinforce-environments reinforce-environments-gym vector
  ];
  homepage = "https://github.com/Sentenai/reinforce#readme";
  description = "Reinforcement learning agents";
  license = stdenv.lib.licenses.bsd3;
}
