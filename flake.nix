{
  description = "quasar3";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.nixpkgs.follows = "nixpkgs";
  };

  nixConfig = {
    extra-trusted-public-keys = "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M=";
    extra-substituters = "https://cache.nixos-cuda.org";
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachSystem [ "aarch64-linux" "x86_64-linux" ] (system:
      let
        pkgs = (import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfreePredicate = pkgs._cuda.lib.allowUnfreeCudaPredicate;
            cudaForwardCompat = false;
            cudaSupport = true;
            allowUnfree = true;
            cudaCapabilities = [ "12.1" ];
          };
        }).cudaPackages_13_0.pkgs;

        inherit (inputs.nixpkgs) lib;
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
          name = "cuda-merged-exo";
          paths = builtins.concatMap 
            (p: [ (lib.getInclude p) (lib.getBin p) (lib.getLib p) (lib.getDev p) ]) cudaLibs;
        };
      in
    {
      devShells.default = pkgs.mkShell rec {
        buildInputs = with pkgs; [
          cargo
          rustc
          rust-analyzer
          rustfmt
          clippy
          gcc
          clang-tools
          tinycc
          python3
          uv
          gdb
          just
          ninja
          cudaRoot
        ];

        env = {
          RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
          LD_LIBRARY_PATH = 
            "$LD_LIBRARY_PATH:${builtins.toString (pkgs.lib.makeLibraryPath buildInputs)}";
          TORCH_CUDA_ARCH_LIST = lib.concatStringsSep ";" cudaPackages.flags.cudaCapabilities;
          FLASHINFER_CUDA_ARCH_LIST = lib.concatStringsSep " " cudaPackages.flags.cudaCapabilities;
          CUDA_HOME = "${cudaRoot}";
        };
      };
    });
}
