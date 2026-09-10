{
  runCommand,
  writeText,
  buildTools,
}:
let
  config = import ../../models/config.nix;
  configJson = writeText "qs3-triton-config.json" (builtins.toJSON config);
in
runCommand "qs3-triton" { nativeBuildInputs = [ buildTools ]; } ''
  export TRITON_CACHE_DIR="$TMPDIR/triton-cache"
  qstriton --config ${configJson} --out "$out" ${./kernels}/lm_head.py
''
