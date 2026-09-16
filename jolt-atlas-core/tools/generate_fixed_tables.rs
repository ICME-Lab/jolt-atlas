//! Regenerate the fixed scale-14 tables with the reference floating formulas.
//! Run with rustc, then pass the fixed_tables output directory as its argument.
#![allow(dead_code)]
use std::{fs, path::Path};
mod consts {
    pub mod general {
        include!("../../common/src/consts/general.rs");
    }
    pub mod trig {
        include!("../../common/src/consts/trig.rs");
    }
}
fn save(path: &Path, name: &str, values: impl Iterator<Item = i32>) {
    let bytes: Vec<u8> = values.flat_map(i32::to_le_bytes).collect();
    fs::write(path.join(name), bytes).unwrap();
}
fn main() {
    use consts::{
        general::MODEL_SCALE,
        trig::{TRIG_DOWNSCALE_BITS, TRIG_PERIOD_MODULUS},
    };
    assert_eq!(MODEL_SCALE, 14);
    let out = std::env::args().nth(1).expect("output directory");
    let path = Path::new(&out);
    fs::create_dir_all(path).unwrap();
    let bits = TRIG_PERIOD_MODULUS.next_power_of_two().ilog2() - TRIG_DOWNSCALE_BITS;
    let reduced = (1u64 << (MODEL_SCALE as u32 - TRIG_DOWNSCALE_BITS)) as f64;
    let factor = 1i32 << TRIG_DOWNSCALE_BITS;
    save(
        path,
        "sin14.bin",
        (0..1u32 << bits)
            .map(|i| (reduced * (f64::from(i) / reduced).sin()).round() as i32 * factor),
    );
    save(
        path,
        "cos14.bin",
        (0..1u32 << bits)
            .map(|i| (reduced * (f64::from(i) / reduced).cos()).round() as i32 * factor),
    );
    let scale = (1u64 << MODEL_SCALE) as f64;
    let n = MODEL_SCALE + 4;
    save(
        path,
        "sigmoid14.bin",
        (0..1i32 << n).map(|i| {
            let signed = if i >= 1 << (n - 1) { i - (1 << n) } else { i };
            (scale / (1.0 + (-(f64::from(signed) / scale)).exp())).round() as i32
        }),
    );
    println!("scale={MODEL_SCALE} trig_period={TRIG_PERIOD_MODULUS} trig_downscale={TRIG_DOWNSCALE_BITS} trig_vars={bits} activation_vars={n}");
}
