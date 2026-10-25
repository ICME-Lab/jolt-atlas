//! Deterministic model inputs derived from an imported model's input
//! descriptors (feature `test-util`).
//!
//! Shared by the importer's golden tests, jolt-atlas-core's model e2e tests
//! and the `prove` example, so the zoo's input conventions live in exactly
//! one place:
//!
//! * names containing `mask` → all ones (`1` raw, `2^scale` when scaled);
//! * names containing `position` → `0, 1, 2, …` (shifted by the scale when
//!   the descriptor is scaled, e.g. Qwen2's RoPE positions);
//! * names containing `token_type` → zeros;
//! * every other raw-int input → random indices below the smallest gather
//!   dictionary in the model; every other scaled input → random values in
//!   `(-2^scale, 2^scale)`.
//!
//! The draw order (one `StdRng`, descriptors in input order, element by
//! element) is part of the golden-hash contract in
//! `tests/golden_zoo.rs`; do not reorder it.

use std::collections::HashMap;

use atlas_trace::{Model, Op, Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{ImportedModel, InputDescriptor, InputKind};

/// The seed the golden tests use.
pub const GOLDEN_SEED: u64 = 0xD1FF;

/// The symbolic dim bindings every zoo model accepts.
pub fn standard_bindings(seq_len: usize) -> HashMap<String, usize> {
    HashMap::from([
        ("batch_size".to_string(), 1),
        ("sequence_length".to_string(), seq_len),
        ("past_sequence_length".to_string(), 0),
    ])
}

/// The smallest gather dictionary in the model (bounds valid index inputs);
/// `2` when the model has no gather.
pub fn min_gather_dict(model: &Model) -> usize {
    model
        .graph()
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            Op::GatherSmall(g) => Some(g.dict_len),
            Op::GatherLarge(g) => Some(g.dict_len),
            _ => None,
        })
        .min()
        .unwrap_or(2)
}

/// Deterministic inputs for every descriptor of `imported`.
pub fn synth_inputs(imported: &ImportedModel, seed: u64) -> Vec<Tensor<i32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let min_dict = min_gather_dict(&imported.model);
    let scale = imported.model.scale().log2() as u8;
    imported
        .input_descriptors
        .iter()
        .map(|d| synth_one(d, scale, min_dict, &mut rng, None))
        .collect()
}

/// Like [`synth_inputs`] but with the first raw-int non-mask/position/
/// token-type input (the token ids) supplied by the caller, e.g. from a
/// tokenizer. `ids` must match that descriptor's element count.
pub fn inputs_with_ids(imported: &ImportedModel, ids: &[i32], seed: u64) -> Vec<Tensor<i32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let min_dict = min_gather_dict(&imported.model);
    let scale = imported.model.scale().log2() as u8;
    let mut ids = Some(ids);
    imported
        .input_descriptors
        .iter()
        .map(|d| {
            let supplied = if is_generic(d) && d.kind == InputKind::RawInt {
                ids.take()
            } else {
                None
            };
            synth_one(d, scale, min_dict, &mut rng, supplied)
        })
        .collect()
}

fn is_generic(d: &InputDescriptor) -> bool {
    let lower = d.name.to_lowercase();
    !(lower.contains("mask") || lower.contains("position") || lower.contains("token_type"))
}

fn synth_one(
    d: &InputDescriptor,
    scale: u8,
    min_dict: usize,
    rng: &mut StdRng,
    supplied: Option<&[i32]>,
) -> Tensor<i32> {
    let s = 1i32 << scale;
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
    } else if let Some(ids) = supplied {
        assert_eq!(ids.len(), len, "input `{}` expects {len} ids", d.name);
        ids.to_vec()
    } else {
        match d.kind {
            InputKind::RawInt => (0..len)
                .map(|_| rng.gen_range(0..(min_dict as i32).max(1)))
                .collect(),
            InputKind::Scaled => (0..len).map(|_| rng.gen_range(-s..s)).collect(),
        }
    };
    Tensor::new(data, &d.dims).expect("descriptor dims")
}
