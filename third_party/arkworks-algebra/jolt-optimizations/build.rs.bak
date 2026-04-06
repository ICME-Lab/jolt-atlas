use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("power_of_2_decompositions.rs");

    // Path to the sage script
    let sage_script = "scripts/bn254_table.sage";

    println!("cargo:rerun-if-changed={}", sage_script);
    println!("cargo:rerun-if-changed=build.rs");

    // Run the sage script to generate the table
    let output = Command::new("sage")
        .arg(sage_script)
        .output()
        .expect("Failed to execute sage command. Make sure SageMath is installed and 'sage' is in your PATH.");

    if !output.status.success() {
        panic!(
            "Sage script failed with status {}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Write the generated table to the output file
    let table_content =
        String::from_utf8(output.stdout).expect("Sage script output is not valid UTF-8");

    fs::write(&dest_path, table_content).expect("Failed to write generated table to output file");

    println!(
        "Generated power of 2 decompositions table at: {}",
        dest_path.display()
    );
}
