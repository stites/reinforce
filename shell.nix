{ compilerVersion ? "ghc843" }:
# pkgs.haskell.packages.ghc843.ghcWithHoogle (hpkgs: with hpkgs; [ reinforce lens ])

let
  config = {
    packageOverrides = pkgs: rec {
      haskell = pkgs.haskell // {
        packages = pkgs.haskell.packages // {
          "${compilerVersion}" = pkgs.haskell.packages."${compilerVersion}".override {
            overrides = haskellPackagesNew: haskellPackagesOld: rec {
            };
          };
        };
      };
    };
  };
  # pkgs = import (fetchGit (import ./version.nix)) { inherit config; };
  pkgs = import <nixpkgs> { inherit config; };
in
  pkgs.haskell.lib.buildStackProject {
    # inherit ghc;
    name = "reinforce-shell";
    buildInputs = [ pkgs.zlib ];
  }

