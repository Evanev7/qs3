use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-changed=qsfi.h");
    println!("cargo:rerun-if-changed=qsfi_internal.h");
    println!("cargo:rerun-if-changed=qsfi_context.cu");
    println!("cargo:rerun-if-changed=qsfi_attn.cu");
    println!("cargo:rerun-if-changed=qsfi_moe.cu");
    println!("cargo:rerun-if-changed=build.ninja");
    println!("cargo:rerun-if-changed=qsfi_flashinfer_check_macros.h");
    println!("cargo:rerun-if-changed=build_tools/flashinfer-trtllm-routing-custom-no-tvm-ffi.patch");
    println!("cargo:rerun-if-changed=qsfi_build_constants.h");
    println!("cargo:rerun-if-changed=build/qsfi_context.o");
    println!("cargo:rerun-if-changed=build/qsfi_attn.o");
    println!("cargo:rerun-if-changed=build/qsfi_moe.o");
    println!("cargo:rerun-if-changed=build/flashinfer.a");

    link_qsfi_for_tests();

    let bindings = bindgen::Builder::default()
        .header("qsfi.h")
        .allowlist_function("qsfi_.*")
        .allowlist_type("qsfi_.*")
        .allowlist_var("QSFI_.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .prepend_enum_name(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("failed to generate qsfi.h bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));
    bindings
        .write_to_file(out_path.join("qsfi_bindings.rs"))
        .expect("failed to write qsfi.h bindings");
}

fn link_qsfi_for_tests() {
    let qsfi_objects = [
        Path::new("build/qsfi_context.o"),
        Path::new("build/qsfi_attn.o"),
        Path::new("build/qsfi_moe.o"),
    ];
    let flashinfer_archive = Path::new("build/flashinfer.a");
    if qsfi_objects
        .iter()
        .copied()
        .any(|object| !object.exists())
        || !flashinfer_archive.exists()
    {
        println!(
            "cargo:warning=qsfi CUDA objects/archive not found; run `just build` before CUDA-backed Rust tests"
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
    println!(
        "cargo:rustc-link-arg-tests={}",
        flashinfer_archive
            .canonicalize()
            .expect("failed to canonicalize FlashInfer archive")
            .display()
    );

    for arg in [
        "-Wl,-Bstatic",
        "-lcudart_static",
        "-lcudadevrt",
        "-Wl,-Bdynamic",
        "-lcuda",
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
