use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=qsfi.h");

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
