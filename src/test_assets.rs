use std::{env, path::PathBuf};

// The real-model oracles were recorded against this BF16 snapshot.
const QWEN36_BF16_SNAPSHOT: &str =
    "models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0";

pub(crate) fn real_qwen36_model_dir() -> PathBuf {
    model_dir(QWEN36_BF16_SNAPSHOT)
}

#[cfg(test)]
pub(crate) fn real_selected_model_dir() -> PathBuf {
    use crate::constants::engine::MODEL;
    use std::process::Command;

    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(MODEL)
        .join("source.nix");
    // Resolve test assets at test time; source revisions are not build constants.
    let output = Command::new("nix")
        .args(["eval", "--offline", "--raw", "--file"])
        .arg(source)
        .args([
            "--apply",
            r#"source: "models--${builtins.replaceStrings ["/"] ["--"] source.repo}/snapshots/${source.rev}""#,
        ])
        .output()
        .expect("resolve the selected test checkpoint with nix");
    assert!(
        output.status.success(),
        "failed to resolve the selected test checkpoint: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    model_dir(std::str::from_utf8(&output.stdout).unwrap())
}

fn model_dir(snapshot: &str) -> PathBuf {
    if let Some(path) = env::var_os("QS3_QWEN36_MODEL_DIR") {
        let path = PathBuf::from(path);
        assert!(
            path.is_dir(),
            "QS3_QWEN36_MODEL_DIR is not a directory: {}",
            path.display()
        );
        return path;
    }
    PathBuf::from(env::var_os("HOME").expect("set HOME or QS3_QWEN36_MODEL_DIR"))
        .join(".cache/huggingface/hub")
        .join(snapshot)
}

pub(crate) fn require_real_qwen36_model_dir() -> PathBuf {
    let path = real_qwen36_model_dir();
    assert!(
        path.is_dir(),
        "pinned BF16 snapshot is missing: {}; set QS3_QWEN36_MODEL_DIR to its directory",
        path.display()
    );
    path
}
