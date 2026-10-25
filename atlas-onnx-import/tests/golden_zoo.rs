//! Golden end-to-end outputs over the model zoo.
//!
//! These replace the retired differential suite against `atlas-onnx-tracer`:
//! every pinned hash was produced by a pipeline that had been verified
//! byte-identical to the old tract-based pipeline on the same model and
//! inputs (Qwen against its unpadded ground truth — the old pipeline's
//! padded Qwen execution was incorrect). A hash change means the imported
//! model's semantics changed and must be deliberate.
//!
//! The checked-in zoo (`golden_zoo`) runs in default CI; the downloaded
//! models (`golden_zoo_big`) run in the heavy CI job:
//! `cargo test -p atlas-onnx-import --test golden_zoo -- --ignored golden_zoo_big`.
//!
//! Inputs come from `atlas_onnx_import::fixtures::synth_inputs` with
//! `GOLDEN_SEED`; its draw order is part of this contract.

use std::path::{Path, PathBuf};

use atlas_onnx_import::LoadOptions;
use atlas_onnx_import::fixtures::{GOLDEN_SEED, standard_bindings, synth_inputs};

fn zoo_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn fnv(values: impl IntoIterator<Item = i32>) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for v in values {
        for byte in v.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

/// Import, build deterministic inputs from the descriptors, forward, hash.
fn model_output_hash(name: &str) -> Result<u64, String> {
    let path = zoo_dir().join(name).join("network.onnx");
    let opts = LoadOptions {
        bindings: standard_bindings(16),
        ..Default::default()
    };
    let imported = atlas_onnx_import::load(&path, &opts).map_err(|e| e.to_string())?;
    let inputs = synth_inputs(&imported, GOLDEN_SEED);
    let outputs = imported.model.forward(&inputs).map_err(|e| e.to_string())?;
    Ok(fnv(outputs.iter().flat_map(|t| t.data().iter().copied())))
}

/// The checked-in zoo: the synthetic op gauntlet plus the tiny random-init
/// stand-ins for the real architectures (`scripts/gen_tiny_models.py`).
const GOLDENS: &[(&str, u64)] = &[
    // Repinned (was 0xc20679e19bec4682): the gauntlet's floor idiom
    // Cast(Cast(x→int64)→f32) used to lower the second Cast to identity,
    // leaving the raw floor(x) misread as a scaled value (off by 2^scale,
    // old-crate parity). cast_policy now rescales provably-raw lineages, so
    // the pattern computes the ONNX value.
    ("op_gauntlet", 0xdd4739a294d6d712),
    ("gpt2-tiny", 0x3c08911de86f9b3a),
    ("qwen2-tiny", 0x77cc6155185561ad),
    ("bge-tiny", 0xce922e2e14210317),
];

/// The big downloaded models (hundreds of MB); separate so default golden
/// runs skip them.
const GOLDENS_BIG: &[(&str, u64)] = &[
    ("gpt2", 0x020d40b051a42652),
    ("qwen", 0xc9c49afa97fc2e1c),
    ("bge-small-en-v1.5", 0x69e16675d78f8737),
];

fn run(goldens: &[(&str, u64)], require_present: bool) {
    let mut failures = vec![];
    for &(name, expected) in goldens {
        if !zoo_dir().join(name).join("network.onnx").exists() {
            if require_present {
                println!("FAIL {name}: model not present");
                failures.push(name);
            } else {
                println!("SKIP {name}: model not present");
            }
            continue;
        }
        match model_output_hash(name) {
            Ok(got) if got == expected => println!("OK   {name}"),
            Ok(got) => {
                println!("FAIL {name}: hash {got:#018x}, pinned {expected:#018x}");
                failures.push(name);
            }
            Err(e) => {
                println!("FAIL {name}: {e}");
                failures.push(name);
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}

#[test]
fn golden_zoo() {
    run(GOLDENS, true);
}

#[test]
#[ignore = "requires the downloaded models (scripts/download_*.py)"]
fn golden_zoo_big() {
    run(GOLDENS_BIG, false);
}
