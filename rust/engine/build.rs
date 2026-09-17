use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

#[allow(dead_code)]
#[path = "src/search/utility/score_value_table.rs"]
mod score_value_table;

fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=src/search/utility/score_value_table.rs");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_MIRI");
    if env::var_os("CARGO_CFG_MIRI").is_none() {
        return Ok(());
    }
    let path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("score_value_table.bin");
    let mut output = BufWriter::new(File::create(path)?);
    for value in score_value_table::build_expected_score_value_table() {
        output.write_all(&value.to_le_bytes())?;
    }
    output.flush()
}
