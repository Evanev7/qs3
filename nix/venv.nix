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
    dependencies.qwen36-vectors = [ ];
    dependencies.qs3-triton = [ ];
  };

  pythonSet =
    (callPackage pyproject-nix.build.packages {
      python = python314;
    }).overrideScope
      (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
        ]
      );
in
pythonSet.mkVirtualEnv "qs3_build_tools" {
  qwen36-vectors = [ ];
  qs3-triton = [ ];
}
