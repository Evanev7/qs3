# CuTe exports a host launcher and embedded device code; launch geometry lives
# in the explicit source entrypoint. No runtime compiler or Python is linked.
let
  config = import ../../models/config.nix;
  names = builtins.filter (name: config.kernels.${name}.provider == "cute") (
    builtins.attrNames config.kernels
  );
  argument = value:
    assert builtins.match "[^$ ]*" value != null;
    "'" + builtins.replaceStrings [ "'" ] [ "'\\''" ] value + "'";
  edge = name:
    let entry = config.kernels.${name}; in
    assert builtins.match "[A-Za-z_][A-Za-z0-9_]*" name != null;
    assert builtins.match "[^$: ]*" entry.source != null;
    ''
      build cute/${name}.rs | cute/${name}.o cute/${name}.json: qscute ../${entry.source} | $qscute ../build_tools/pyproject.toml ../build_tools/uv.lock
        spec = ${argument (builtins.toJSON entry.spec)}
        kernel = ${name}
    '';
in
''
  rule qscute
    command = "$qscute" --source $in --spec $spec --target ${argument (builtins.toJSON config.target)} --prefix cute/$kernel
    description = QSCUTE $kernel
    depfile = cute/$kernel.d
    deps = gcc

''
+ builtins.concatStringsSep "\n" (map edge names)
+ ''
  build cute/kernels: phony ${builtins.concatStringsSep " " (map (name: "cute/${name}.rs") names)}

  rule cute_archive
    command = rm -f $out && ar rcs $out $in
    description = AR $out
  build libqscute.a: cute_archive ${builtins.concatStringsSep " " (map (name: "cute/${name}.o") names)}

  rule cute_runtime
    command = "$qscute_runtime" $out
    description = CUTE RUNTIME $out
  build libcuda_dialect_runtime_static.a: cute_runtime | $qscute_runtime ../build_tools/pysrc/qscute/runtime.py ../build_tools/pyproject.toml ../build_tools/uv.lock
''
