{ compilerVersion ? "ghc843" }:

let
  config = {
    allowUnfree = true;
    packageOverrides = pkgs: rec {
      haskell = pkgs.haskell // {
        packages = pkgs.haskell.packages // {
          "${compilerVersion}" = pkgs.haskell.packages."${compilerVersion}".override {
            overrides = haskellPackagesNew: haskellPackagesOld: rec {
              # hasktorch-core             = haskellPackagesNew.callPackage ./hasktorch/core { };
              # hasktorch-indef            = haskellPackagesNew.callPackage ./hasktorch/indef { };
              # hasktorch-signatures       = haskellPackagesNew.callPackage ./hasktorch/signatures { };
              # hasktorch-signatures-types = haskellPackagesNew.callPackage ./hasktorch/signatures/types { };
              # hasktorch-partial          = haskellPackagesNew.callPackage ./hasktorch/signatures/partial { };
              # hasktorch-types-th         = haskellPackagesNew.callPackage ./hasktorch/types/th { };
              # hasktorch-types-thc        = haskellPackagesNew.callPackage ./hasktorch/types/thc { };
              # hasktorch-raw-th           = haskellPackagesNew.callPackage ./hasktorch/raw/th { };
              # hasktorch-raw-thc          = haskellPackagesNew.callPackage ./hasktorch/raw/thc { };
              # hasktorch-raw-tests        = haskellPackagesNew.callPackage ./hasktorch/raw/tests { };
              # hasktorch-examples         = haskellPackagesNew.callPackage ./hasktorch/examples { };
              # type-combinators           = haskellPackagesNew.callPackage ./type-combinators.nix;
              gym-http-api               = haskellPackagesNew.callPackage ./gym-http-api/binding-hs { };

              reinforce                  = haskellPackagesNew.callPackage ./reinforce { };
              reinforce-algorithms       = haskellPackagesNew.callPackage ./reinforce-algorithms { };
              reinforce-environments     = haskellPackagesNew.callPackage ./reinforce-environments { };
              reinforce-environments-gym = haskellPackagesNew.callPackage ./reinforce-environments-gym { };
            };
          };
        };
      };
    };
  };
  # pkgs = import (fetchGit (import ./version.nix)) { inherit config; };
  pkgs = import <nixpkgs> { inherit config; };
in
  {
    reinforce                  = pkgs.haskell.packages.${compilerVersion}.reinforce;
    reinforce-algorithms       = pkgs.haskell.packages.${compilerVersion}.reinforce-algorithms;
    reinforce-environments     = pkgs.haskell.packages.${compilerVersion}.reinforce-environments;
    reinforce-environments-gym = pkgs.haskell.packages.${compilerVersion}.reinforce-environments-gym;

    # reinforce-zoo:
    # ==============

    # zoo-egreedy-bandits  = haskellApp ./reinforce-zoo;
    # qtable-cartpole-example
    # qtable-frozenlake-example
    # random-agent-example

  }

