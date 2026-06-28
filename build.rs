use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-changed=qs_bindings.h");
    println!("cargo:rerun-if-changed=qs_info.h");
    println!("cargo:rerun-if-changed=qs_tensor.h");
    println!("cargo:rerun-if-changed=qsfi.h");
    println!("cargo:rerun-if-changed=qsfi_internal.h");
    println!("cargo:rerun-if-changed=qscu.h");
    println!("cargo:rerun-if-changed=qscb.h");
    println!("cargo:rerun-if-changed=qsfi_context.cu");
    println!("cargo:rerun-if-changed=qsfi_attn.cu");
    println!("cargo:rerun-if-changed=qsfi_gdn.cu");
    println!("cargo:rerun-if-changed=qsfi_moe.cu");
    println!("cargo:rerun-if-changed=qsfi_norm_rope.cu");
    println!("cargo:rerun-if-changed=qscu.cu");
    println!("cargo:rerun-if-changed=qscb.cu");
    println!("cargo:rerun-if-changed=build_tools/build.ninja");
    println!("cargo:rerun-if-changed=build_tools/generate_macros.c");
    println!("cargo:rerun-if-changed=build/qsfi_context.o");
    println!("cargo:rerun-if-changed=build/qsfi_attn.o");
    println!("cargo:rerun-if-changed=build/qsfi_gdn.o");
    println!("cargo:rerun-if-changed=build/qsfi_moe.o");
    println!("cargo:rerun-if-changed=build/qsfi_norm_rope.o");
    println!("cargo:rerun-if-changed=build/qscu.o");
    println!("cargo:rerun-if-changed=build/qscb.o");

    link_qsfi_for_tests();

    let bindings = bindgen::Builder::default()
        .header("qs_bindings.h")
        .allowlist_function("(qsfi|qscu|qscb)_.*")
        .allowlist_type("(qsfi|qscu|qscb)_.*")
        .allowlist_var("(QSFI|QSCU|QSCB)_.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .prepend_enum_name(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("failed to generate qs_bindings.h bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));
    bindings
        .write_to_file(out_path.join("qsfi_bindings.rs"))
        .expect("failed to write qs_bindings.h bindings");
}

fn link_qsfi_for_tests() {
    let qsfi_objects = [
        Path::new("build/qsfi_context.o"),
        Path::new("build/qsfi_attn.o"),
        Path::new("build/qsfi_gdn.o"),
        Path::new("build/qsfi_moe.o"),
        Path::new("build/qsfi_norm_rope.o"),
        Path::new("build/qscu.o"),
        Path::new("build/qscb.o"),
    ];
    if qsfi_objects.iter().copied().any(|object| !object.exists()) {
        println!(
            "cargo:warning=qsfi CUDA objects not found; run `just build` before CUDA-backed Rust tests"
        );
        return;
    }

    for object in qsfi_objects {
        println!(
            "cargo:rustc-link-arg-tests={}",
            object
                .canonicalize()
                .expect("failed to canonicalize qsfi CUDA object")
                .display()
        );
    }

    for arg in [
        "-Wl,-Bstatic",
        "-lcudart_static",
        "-lcudadevrt",
        "-Wl,-Bdynamic",
        "-lcuda",
        "-lcublas",
        "-lcublasLt",
        "-lstdc++",
        "-ldl",
        "-lrt",
        "-lpthread",
        "-lm",
        "-lc",
    ] {
        println!("cargo:rustc-link-arg-tests={arg}");
    }
}
