//! Golden end-to-end outputs over the model zoo.
//!
//! These replace the retired differential suite against `atlas-onnx-tracer`:
//! every pinned hash was produced by a pipeline that had been verified
//! byte-identical to the old tract-based pipeline on the same model and
//! inputs (Qwen against its unpadded ground truth — the old pipeline's
//! padded Qwen execution was incorrect). A hash change means the imported
//! model's semantics changed and must be deliberate.
//!
//! Run with `cargo test -p atlas-onnx-import --test golden_zoo -- --ignored`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use atlas_onnx_import::{InputKind, LoadOptions};
use atlas_trace::{Op, Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn zoo_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn bindings() -> HashMap<String, usize> {
    HashMap::from([
        ("batch_size".to_string(), 1),
        ("sequence_length".to_string(), 16),
        ("past_sequence_length".to_string(), 0),
    ])
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
        bindings: bindings(),
        ..Default::default()
    };
    let imported = atlas_onnx_import::load(&path, &opts).map_err(|e| e.to_string())?;
    let s = 1i32 << 14;

    // Smallest gather dictionary bounds valid index inputs.
    let min_dict = imported
        .model
        .graph()
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            Op::GatherSmall(g) => Some(g.dict_len),
            Op::GatherLarge(g) => Some(g.dict_len),
            _ => None,
        })
        .min()
        .unwrap_or(2);

    let mut rng = StdRng::seed_from_u64(0xD1FF);
    let inputs: Vec<Tensor<i32>> = imported
        .input_descriptors
        .iter()
        .map(|d| {
            let len: usize = d.dims.iter().product();
            let lower = d.name.to_lowercase();
            let data: Vec<i32> = if lower.contains("mask") {
                match d.kind {
                    InputKind::Scaled => vec![s; len],
                    InputKind::RawInt => vec![1; len],
                }
            } else if lower.contains("position") {
                match d.kind {
                    InputKind::Scaled => (0..len as i32).map(|i| i * s).collect(),
                    InputKind::RawInt => (0..len as i32).collect(),
                }
            } else if lower.contains("token_type") {
                vec![0; len]
            } else {
                match d.kind {
                    InputKind::RawInt => (0..len)
                        .map(|_| rng.gen_range(0..(min_dict as i32).max(1)))
                        .collect(),
                    InputKind::Scaled => (0..len).map(|_| rng.gen_range(-s..s)).collect(),
                }
            };
            Tensor::new(data, &d.dims).expect("dims")
        })
        .collect();

    let outputs = imported.model.forward(&inputs).map_err(|e| e.to_string())?;
    Ok(fnv(outputs.iter().flat_map(|t| t.data().iter().copied())))
}

const GOLDENS: &[(&str, u64)] = &[
    // Repinned (was 0xc20679e19bec4682): the gauntlet's floor idiom
    // Cast(Cast(x→int64)→f32) used to lower the second Cast to identity,
    // leaving the raw floor(x) misread as a scaled value (off by 2^scale,
    // old-crate parity). cast_policy now rescales provably-raw lineages, so
    // the pattern computes the ONNX value.
    ("op_gauntlet", 0xdd4739a294d6d712),
    ("microgpt", 0xe5dc388d7ed530e2),
    ("multihead_attention", 0x4f61f7105a88d2a5),
    ("transformer", 0x3ce822595b275bbc),
    ("nanoGPT", 0x3f3292352e76de6f),
];

/// The big downloaded models (hundreds of MB); separate so default golden
/// runs skip them.
const GOLDENS_BIG: &[(&str, u64)] = &[
    ("gpt2", 0x75941f09ede32d17),
    ("qwen", 0x509440401f29e855),
    ("bge-small-en-v1.5", 0x512cc8ab097173be),
];

fn run(goldens: &[(&str, u64)]) {
    let mut failures = vec![];
    for &(name, expected) in goldens {
        if !zoo_dir().join(name).join("network.onnx").exists() {
            println!("SKIP {name}: model not present");
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
#[ignore]
fn golden_zoo() {
    run(GOLDENS);
}

#[test]
#[ignore]
fn golden_zoo_big() {
    run(GOLDENS_BIG);
}
