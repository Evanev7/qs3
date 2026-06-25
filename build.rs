use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-changed=qsfi.h");
    println!("cargo:rerun-if-changed=qsfi.cu");
    println!("cargo:rerun-if-changed=qsfi_build_constants.h");
    println!("cargo:rerun-if-changed=build/qsfi.o");

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
    let qsfi_object = Path::new("build/qsfi.o");
    if !qsfi_object.exists() {
        println!(
            "cargo:warning=build/qsfi.o not found; run `just build` before CUDA-backed Rust tests"
        );
        return;
    }

    println!(
        "cargo:rustc-link-arg-tests={}",
        qsfi_object
            .canonicalize()
            .expect("failed to canonicalize build/qsfi.o")
            .display()
    );

    for arg in [
        "-Wl,-Bstatic",
        "-lcudart_static",
        "-lcudadevrt",
        "-Wl,-Bdynamic",
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
