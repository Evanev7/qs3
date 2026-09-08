use std::{env, path::PathBuf};

fn main() {
    println!(
        "cargo:rustc-env=QS3_BUILD_PROFILE={}",
        env::var("PROFILE").unwrap()
    );
    let rustc = std::process::Command::new(env::var_os("RUSTC").unwrap())
        .arg("-V")
        .output()
        .expect("query Rust compiler version");
    assert!(rustc.status.success(), "query Rust compiler version");
    println!(
        "cargo:rustc-env=QS3_RUSTC_VERSION={}",
        String::from_utf8(rustc.stdout).unwrap().trim()
    );
    println!("cargo:rerun-if-changed=qs_ffi.h");
    println!("cargo:rerun-if-changed=qs_info.h");
    println!("cargo:rerun-if-changed=qs_tensor.h");
    println!("cargo:rerun-if-changed=qsfi.h");
    println!("cargo:rerun-if-changed=qsfi_internal.h");
    println!("cargo:rerun-if-changed=qsfi_native_common.h");
    println!("cargo:rerun-if-changed=qscu.h");
    println!("cargo:rerun-if-changed=qscb.h");
    println!("cargo:rerun-if-changed=qsfi.cu");
    println!("cargo:rerun-if-changed=qsfi_context.cu");
    println!("cargo:rerun-if-changed=qsfi_attn.cu");
    println!("cargo:rerun-if-changed=qscu_gdn.cu");
    println!("cargo:rerun-if-changed=qsfi_moe.cu");
    println!("cargo:rerun-if-changed=qsfi_norm_rope.cu");
    println!("cargo:rerun-if-changed=qscu.cu");
    println!("cargo:rerun-if-changed=qscb.cu");
    println!("cargo:rerun-if-changed=build_tools/build.ninja");
    println!("cargo:rerun-if-changed=build_tools/generate_macros.c");
    println!("cargo:rerun-if-changed=build/libqs_native.a");

    println!("cargo:rustc-link-search=build");
    println!("cargo:rustc-link-lib=static=qs_native");
    for lib in ["cudart", "cublasLt", "stdc++"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }

    let bindings = bindgen::Builder::default()
        .header("qs_ffi.h")
        .allowlist_function("(qsfi|qscu|qscb)_.*")
        .allowlist_type("(qsfi|qscu|qscb)_.*")
        .allowlist_var("(QSFI|QSCU|QSCB)_.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .prepend_enum_name(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("failed to generate FFI bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));
    bindings
        .write_to_file(out_path.join("ffi_bindings.rs"))
        .expect("failed to write FFI bindings");
}
