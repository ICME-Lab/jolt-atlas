//! Mutation sites over a finished proof and its public IO.
//!
//! A proof is the bytes the verifier's transcript reads, and every read is absorbed, so a
//! mutated proof is usually rejected because the verifier's transcript diverges rather than
//! because an algebraic check fails. That is still the property we want: *every byte is
//! either absorbed or checked*. A site whose mutation is accepted is a dead byte (proof
//! malleability); a verifier panic is a robustness bug.

use atlas_sumcheck::opening_proof::{OpeningId, SumcheckId};
use atlas_trace::{trace::ModelExecutionIO, Model, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::onnx_proof::ONNXProof;

/// One mutation of a proof or its IO.
#[derive(Clone, Debug)]
pub(crate) enum Site {
    /// `proof[i] ^= 1`.
    ProofByte { i: usize },
    /// Drop the last byte of the proof.
    ProofTruncated,
    /// Append a byte to the proof.
    ProofExtended,
    /// The proof of a second honest execution, against this execution's IO.
    ProofOfOtherExecution,
    /// `io.outputs[0][elem] += 1`.
    IoOutput { elem: usize },
    /// `io.inputs[tensor][elem] += 1`.
    IoInput { tensor: usize, elem: usize },
    /// `io.input_indices[i] += 1`.
    IoInputIndex { i: usize },
    /// No output tensor.
    IoOutputCount,
    /// The output tensor truncated by one element (a shape the model does not produce).
    IoOutputShape,
    /// The last input tensor dropped.
    IoInputCount,
}

/// Whether to enumerate every site (`JOLT_SOUNDNESS_EXHAUSTIVE=1`).
pub(crate) fn exhaustive() -> bool {
    std::env::var_os("JOLT_SOUNDNESS_EXHAUSTIVE").is_some_and(|v| v != "0")
}

/// Sampling seed (`JOLT_SOUNDNESS_SEED`, default fixed).
pub(crate) fn sampling_seed() -> u64 {
    std::env::var("JOLT_SOUNDNESS_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x50FF_D00D)
}

fn variant_name(debug: String) -> String {
    debug
        .split(['(', '{', ' '])
        .next()
        .unwrap_or("")
        .to_string()
}

/// The stratum of an opening id: (polynomial kind, sumcheck kind, op kind
/// of the executing node). Sampled sweeps keep one id per stratum.
pub(crate) fn opening_stratum(id: &OpeningId, model: &Model) -> (String, String, String) {
    let node_kind = match id.sumcheck {
        SumcheckId::NodeExecution(idx) if idx < model.graph().nodes.len() => {
            format!(
                "{:?}",
                super::op_kind::OpKind::of(&model.graph().nodes[idx].op)
            )
        }
        _ => String::new(),
    };
    (
        variant_name(format!("{:?}", id.polynomial)),
        variant_name(format!("{:?}", id.sumcheck)),
        node_kind,
    )
}

/// Every mutation site of `proof` / `io`. The verifier must reject each one with an `Err`:
/// never accept, never panic.
pub(crate) fn enumerate_sites(proof: &ONNXProof, io: &ModelExecutionIO) -> Vec<Site> {
    let mut sites = vec![];

    let len = proof.0.len();
    let bytes: Vec<usize> = if exhaustive() {
        (0..len).collect()
    } else {
        let mut rng = StdRng::seed_from_u64(sampling_seed());
        (0..64).map(|_| rng.gen_range(0..len)).collect()
    };
    for i in bytes {
        sites.push(Site::ProofByte { i });
    }
    sites.push(Site::ProofTruncated);
    sites.push(Site::ProofExtended);
    sites.push(Site::ProofOfOtherExecution);

    let out_len = io.outputs[0].data().len();
    for elem in [0, out_len / 2, out_len - 1] {
        sites.push(Site::IoOutput { elem });
    }
    for (tensor, t) in io.inputs.iter().enumerate() {
        let len = t.data().len();
        for elem in [0, len / 2, len - 1] {
            sites.push(Site::IoInput { tensor, elem });
        }
    }
    for i in 0..io.input_indices.len() {
        sites.push(Site::IoInputIndex { i });
    }
    sites.push(Site::IoOutputCount);
    sites.push(Site::IoOutputShape);
    sites.push(Site::IoInputCount);

    // Dedupe sites that collapsed onto the same element.
    let mut seen = std::collections::HashSet::new();
    sites.retain(|s| seen.insert(format!("{s:?}")));
    sites
}

/// Apply `site` to `proof` / `io`. `other` is a second honest proof of the same model on
/// different inputs.
pub(crate) fn apply_site(
    proof: &mut ONNXProof,
    io: &mut ModelExecutionIO,
    other: &ONNXProof,
    site: &Site,
) {
    match site {
        Site::ProofByte { i } => proof.0[*i] ^= 1,
        Site::ProofTruncated => {
            proof.0.pop();
        }
        Site::ProofExtended => proof.0.push(0),
        Site::ProofOfOtherExecution => *proof = other.clone(),
        Site::IoOutput { elem } => {
            let d = io.outputs[0].data_mut();
            d[*elem] = d[*elem].wrapping_add(1);
        }
        Site::IoInput { tensor, elem } => {
            let d = io.inputs[*tensor].data_mut();
            d[*elem] = d[*elem].wrapping_add(1);
        }
        Site::IoInputIndex { i } => io.input_indices[*i] += 1,
        Site::IoOutputCount => io.outputs.clear(),
        Site::IoOutputShape => {
            let out = &io.outputs[0];
            let len = out.data().len() - 1;
            io.outputs[0] = Tensor::from_slice(&out.data()[..len], &[len]).expect("site");
        }
        Site::IoInputCount => {
            io.inputs.pop();
        }
    }
}

/// A short label for grouping sites in reports.
pub(crate) fn site_class(site: &Site) -> &'static str {
    match site {
        Site::ProofByte { .. } => "ProofByte",
        Site::ProofTruncated => "ProofTruncated",
        Site::ProofExtended => "ProofExtended",
        Site::ProofOfOtherExecution => "ProofOfOtherExecution",
        Site::IoOutput { .. } => "IoOutput",
        Site::IoInput { .. } => "IoInput",
        Site::IoInputIndex { .. } => "IoInputIndex",
        Site::IoOutputCount => "IoOutputCount",
        Site::IoOutputShape => "IoOutputShape",
        Site::IoInputCount => "IoInputCount",
    }
}
