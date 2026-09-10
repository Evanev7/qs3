{
  description = "quasar3";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
        nixpkgs.follows = "nixpkgs";
      };
    };

  };

  nixConfig = {
    extra-trusted-public-keys = "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M=";
    extra-substituters = "https://cache.nixos-cuda.org";
  };

  outputs =
    inputs:
    inputs.flake-utils.lib.eachSystem [ "aarch64-linux" "x86_64-linux" ] (
      system:
      let
        inherit (inputs.nixpkgs) lib;
        pkgs =
          (import inputs.nixpkgs {
            inherit system;
            config = {
              allowUnfreePredicate = pkgs._cuda.lib.allowUnfreeCudaPredicate;
              cudaForwardCompat = false;
              cudaSupport = true;
              allowUnfree = true;
              cudaCapabilities = [ "12.1" ];
            };
            overlays = [ ];
          }).cudaPackages_13_0.pkgs;
        inherit (pkgs) cudaPackages;
        cudaLibs = with cudaPackages; [
          cuda_crt
          cuda_cudart
          cuda_cccl
          cuda_cupti
          cuda_nvrtc
          cuda_nvtx
          cudnn
          libcufile
          libcublas
          libcufft
          libcurand
          libcusolver
          libcusparse
          libcusparse_lt
          libnvjitlink
          #libnvshmem
          #nccl
          cuda_nvcc
        ];
        cudaRoot = pkgs.symlinkJoin {
          name = "cuda-merged-qs3";
          paths = builtins.concatMap (p: [
            (lib.getInclude p)
            (lib.getBin p)
            (lib.getLib p)
            (lib.getDev p)
          ]) cudaLibs;
        };
        craneLib = inputs.crane.mkLib pkgs;
        src = lib.cleanSourceWith {
          src = ./.;
          filter =
            path: type:
            baseNameOf path != "3pty"
            && (
              craneLib.filterCargoSources path type || builtins.match ".*\\.(c|cu|h|inc|ninja)$" path != null
            );
        };
        flashinferSrc = pkgs.fetchFromGitHub {
          owner = "flashinfer-ai";
          repo = "flashinfer";
          rev = "b3baedbbef2686df91b6dc43818ee56fe26ceba2";
          hash = "sha256-a++G2Bdm9eJIfu4ERa5CCS01dOHxp94Or5vHiXYIwns=";
          fetchSubmodules = true;
        };
        qsNative = pkgs.stdenv.mkDerivation {
          name = "qs3-native";
          src = lib.cleanSourceWith {
            name = "qs3-native-source";
            src = ./.;
            filter =
              path: type:
              let
                relative = lib.removePrefix "${toString ./.}/" path;
              in
              (type == "directory" && (path == toString ./. || relative == "build_tools"))
              || builtins.match "[^/]*\\.(c|cu|h|inc)" relative != null
              || builtins.elem relative [
                "build_tools/build.ninja"
                "build_tools/generate_macros.c"
              ];
          };
          strictDeps = true;
          buildInputs = cudaLibs;
          nativeBuildInputs = with pkgs; [
            ninja
            tinycc
            cudaPackages.cuda_nvcc
          ];
          dontConfigure = true;
          buildPhase = ''
            runHook preBuild
            mkdir -p 3pty build
            ln -s ${flashinferSrc} 3pty/flashinfer
            cp build_tools/build.ninja build/build.ninja
            ninja -C build -j "$NIX_BUILD_CORES" libqs_native.a
            runHook postBuild
          '';
          installPhase = ''
            runHook preInstall
            install -Dm644 build/libqs_native.a "$out/lib/libqs_native.a"
            runHook postInstall
          '';
        };
        buildTools = pkgs.callPackage nix/venv.nix {
          inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
        };
        qsTriton = pkgs.callPackage build_tools/qstriton { inherit buildTools; };
        commonArgs = {
          inherit src;
          inherit (craneLib.crateNameFromCargoToml { inherit src; }) version;
          CARGO_PROFILE = "release";
          cargoArtifacts = craneLib.buildDepsOnly (commonArgs // { buildInputs = cudaLibs; });
          strictDeps = true;
          buildInputs = cudaLibs ++ [ qsNative ];
          # Link Driver API symbols with the toolkit stub; load the host driver at runtime.
          LIBRARY_PATH = "${cudaPackages.cuda_cudart}/lib/stubs";
          nativeBuildInputs = with pkgs; [
            rustPlatform.bindgenHook
          ];
          doCheck = false;
          preBuild = ''
            mkdir -p build
            ln -s ${qsNative}/lib/libqs_native.a build/libqs_native.a
            ln -s ${qsTriton} build/triton
          '';
        };
      in
      {
        packages = {
          default = craneLib.buildPackage (commonArgs // { cargoBuildExtraArgs = "--lib"; });
          native = qsNative;
          triton = qsTriton;
          benchmark = craneLib.buildPackage (
            commonArgs
            // {
              pnameSuffix = "-benchmark";
              cargoBuildExtraArgs = "--bin qs3-bench";
              meta.mainProgram = "qs3-bench";
            }
          );
          venv = buildTools;
        };
        devShells.default = pkgs.mkShell rec {
          buildInputs = with pkgs; [
            uv
            cargo
            rustc
            rust-analyzer
            rustfmt
            clippy
            gcc
            clang-tools
            tinycc
            python3
            gdb
            just
            ninja
            cudaRoot
          ];

          env = {
            RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
            TORCH_CUDA_ARCH_LIST = lib.concatStringsSep ";" cudaPackages.flags.cudaCapabilities;
            FLASHINFER_CUDA_ARCH_LIST = lib.concatStringsSep " " cudaPackages.flags.cudaCapabilities;
            CUDA_HOME = "${cudaRoot}";
          };
          shellHook = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib.makeLibraryPath buildInputs}";
        };
      }
    );
}
