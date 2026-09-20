{
  lib,
  callPackage,
  python314,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
}:

let
  workspace = uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ../build_tools;
  };

  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
    dependencies.build-tools = [ ];
  };

  pythonSet =
    (callPackage pyproject-nix.build.packages {
      python = python314;
    }).overrideScope
      (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
          (_: prev: {
            nvidia-cutlass-dsl-libs-base = prev.nvidia-cutlass-dsl-libs-base.overrideAttrs (_: {
              autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
            });
          })
        ]
      );
in
pythonSet.mkVirtualEnv "qs3_build_tools" {
  build-tools = [ ];
}
