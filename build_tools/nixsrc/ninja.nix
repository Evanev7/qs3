# The configured Triton specializations. Build orchestration stays handwritten.
let
  config = import ../../models/config.nix;
  names = builtins.filter (name: config.kernels.${name}.provider == "triton") (
    builtins.attrNames config.kernels
  );
  # Preserve JSON quotes through the shell before reaching argparse.
  argument = value:
    assert builtins.match "[^$ ]*" value != null;
    "'" + builtins.replaceStrings [ "'" ] [ "'\\''" ] value + "'";
  edge = name:
    let entry = config.kernels.${name}; in
    assert builtins.match "[A-Za-z_][A-Za-z0-9_]*" name != null;
    assert builtins.match "[^$: ]*" entry.source != null;
    ''
      build triton/${name}.rs | triton/${name}.cubin triton/${name}.ptx triton/${name}.json: qstriton ../${entry.source} | $qstriton ../build_tools/pyproject.toml ../build_tools/uv.lock
        spec = ${argument (builtins.toJSON entry.spec)}
        kernel = ${name}
    '';
in
''
  rule qstriton
    command = "$qstriton" --source $in --spec $spec --target ${argument (builtins.toJSON config.target)} --prefix triton/$kernel
    description = QSTRITON $kernel
    depfile = triton/$kernel.d
    deps = gcc

''
+ builtins.concatStringsSep "\n" (map edge names)
+ ''
  build triton/kernels: phony ${builtins.concatStringsSep " " (map (name: "triton/${name}.rs") names)}
''
