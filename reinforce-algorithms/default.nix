{ mkDerivation, base, hashable, hasktorch-core, microlens-platform
, mtl, reinforce, stdenv, transformers, unordered-containers
}:
mkDerivation {
  pname = "reinforce-algorithms";
  version = "0.0.1.0";
  src = ./.;
  libraryHaskellDepends = [
    base hashable hasktorch-core microlens-platform mtl reinforce
    transformers unordered-containers
  ];
  license = stdenv.lib.licenses.unfree;
  hydraPlatforms = stdenv.lib.platforms.none;
}
