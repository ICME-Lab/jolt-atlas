//! Mutation sites over a finished proof and its public IO.
//!
//! Nearly every field of an [`ONNXProof`] is absorbed into the Fiat-Shamir
//! transcript, so a mutated proof is usually rejected because the verifier's
//! transcript diverges rather than because an algebraic check fails. That is
//! still the property we want: *every field is either absorbed or checked*.
//! A site whose mutation is accepted is a dead field (proof malleability) or
//! an orphan claim; a verifier panic is a robustness bug.

use atlas_pcs::{akita::AkitaScheme, commitment_scheme::CommitmentScheme};
use atlas_sumcheck::opening_proof::{OpeningId, SumcheckId};
use atlas_trace::{trace::ModelExecutionIO, Model, Tensor};
use num_traits::One;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::onnx_proof::ONNXProof;

/// A PCS-specific edit of the joint opening proof.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PcsSite {
    /// Akita: `slot_evals[slot] += 1` in the `i`-th class opening.
    SlotEval { i: usize, slot: usize },
    /// Akita: an extra slot evaluation in the `i`-th class opening.
    SlotEvalExtra { i: usize },
    /// Akita: flip a byte of the `i`-th class opening's proof.
    ClassProofByte { i: usize },
    /// Akita: drop the `i`-th class opening.
    ClassOpeningDropped { i: usize },
    /// Akita: swap two class openings.
    ClassOpeningSwap { i: usize, j: usize },
}

/// A PCS-specific edit of the batch commitment.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BatchSite {
    /// Akita: the class's kind tag `+= 1`.
    Kind { class: usize },
    /// Akita: the class's variable count `+= 1`.
    NumVars { class: usize },
    /// Akita: the class's slot count `+= 1`.
    SlotCount { class: usize },
    /// Akita: flip a byte of the class commitment.
    CommitmentByte { class: usize },
    /// Akita: declare the class identically zero.
    Zeroed { class: usize },
    /// Akita: the same class from a second honest execution.
    Replaced { class: usize },
}

/// How a PCS exposes its opening proof and batch commitment to the sweep.
pub(crate) trait PcsMutations: CommitmentScheme<Commitment: PartialEq> {
    fn sites(proof: &Self::Proof) -> Vec<PcsSite>;
    fn apply(proof: &mut Self::Proof, site: PcsSite);

    /// The batch commitment's sites; `other` is a second honest execution's.
    fn batch_sites(
        _batch: &Self::BatchCommitment,
        _other: &Self::BatchCommitment,
    ) -> Vec<BatchSite> {
        vec![]
    }
    fn apply_batch(
        _batch: &mut Self::BatchCommitment,
        _other: &Self::BatchCommitment,
        _site: BatchSite,
    ) {
    }
}

impl PcsMutations for AkitaScheme {
    fn sites(p: &Self::Proof) -> Vec<PcsSite> {
        let mut sites = vec![];
        for (i, class) in p.classes.iter().enumerate() {
            let last = class.slot_evals.len().saturating_sub(1);
            for slot in [0, last] {
                sites.push(PcsSite::SlotEval { i, slot });
            }
            sites.push(PcsSite::SlotEvalExtra { i });
            if !class.proof.is_empty() {
                sites.push(PcsSite::ClassProofByte { i });
            }
            sites.push(PcsSite::ClassOpeningDropped { i });
        }
        if p.classes.len() >= 2 {
            sites.push(PcsSite::ClassOpeningSwap { i: 0, j: 1 });
        }
        sites
    }

    fn apply(p: &mut Self::Proof, site: PcsSite) {
        let one = <Self as CommitmentScheme>::Field::one();
        match site {
            PcsSite::SlotEval { i, slot } => p.classes[i].slot_evals[slot] += one,
            PcsSite::SlotEvalExtra { i } => p.classes[i].slot_evals.push(one),
            PcsSite::ClassProofByte { i } => {
                let proof = &mut p.classes[i].proof;
                let mid = proof.len() / 2;
                proof[mid] ^= 1;
            }
            PcsSite::ClassOpeningDropped { i } => {
                p.classes.remove(i);
            }
            PcsSite::ClassOpeningSwap { i, j } => p.classes.swap(i, j),
        }
    }

    fn batch_sites(batch: &Self::BatchCommitment, other: &Self::BatchCommitment) -> Vec<BatchSite> {
        let mut sites = vec![];
        for (class, c) in batch.classes.iter().enumerate() {
            sites.push(BatchSite::Kind { class });
            sites.push(BatchSite::NumVars { class });
            sites.push(BatchSite::SlotCount { class });
            if c.commitment.is_some() {
                sites.push(BatchSite::CommitmentByte { class });
                sites.push(BatchSite::Zeroed { class });
            }
            if other.classes.get(class).is_some_and(|o| o != c) {
                sites.push(BatchSite::Replaced { class });
            }
        }
        sites
    }

    fn apply_batch(
        batch: &mut Self::BatchCommitment,
        other: &Self::BatchCommitment,
        site: BatchSite,
    ) {
        match site {
            BatchSite::Kind { class } => batch.classes[class].kind += 1,
            BatchSite::NumVars { class } => batch.classes[class].logical_num_vars += 1,
            BatchSite::SlotCount { class } => batch.classes[class].slot_count += 1,
            BatchSite::CommitmentByte { class } => {
                let bytes = batch.classes[class].commitment.as_mut().expect("site");
                let mid = bytes.len() / 2;
                bytes[mid] ^= 1;
            }
            BatchSite::Zeroed { class } => batch.classes[class].commitment = None,
            BatchSite::Replaced { class } => batch.classes[class] = other.classes[class].clone(),
        }
    }
}

/// One mutation of a proof or its IO.
#[derive(Clone, Debug)]
pub(crate) enum Site {
    /// Swap two commitments.
    CommitmentSwap { i: usize, j: usize },
    /// Replace a commitment with the same polynomial's commitment from a
    /// second honest execution.
    CommitmentReplaced { i: usize },
    /// Drop the last commitment.
    CommitmentDropped,
    /// A PCS-level edit of the batch commitment.
    BatchCommitment(BatchSite),
    /// No reduced opening proof at all.
    ReducedRemoved,
    /// The reduced opening proof of a second honest execution.
    ReducedSpliced,
    /// A PCS-level edit of the joint opening proof.
    ReducedPcs(PcsSite),
    /// `narg[i] ^= 1` (a proof-of-work nonce byte).
    NargByte { i: usize },
    /// Drop the last NARG byte.
    NargTruncated,
    /// Append a byte to the NARG string.
    NargExtended,
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
/// never accept, never panic. Round
/// coefficients are sampled (first round, last round, two random rounds per
/// proof) unless exhaustive.
pub(crate) fn enumerate_sites<PCS: PcsMutations>(
    proof: &ONNXProof<PCS::Field, PCS>,
    other: &ONNXProof<PCS::Field, PCS>,
    io: &ModelExecutionIO,
) -> Vec<Site> {
    let mut rng = StdRng::seed_from_u64(sampling_seed());
    let all = exhaustive();
    let mut sites = vec![];

    // Swapping or replacing a commitment is a no-op when the two polynomials
    // (or the two executions' versions of one polynomial) are identical.
    let n = proof.commitments().len();
    if n >= 2 {
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .filter(|&(i, j)| proof.commitments()[i] != proof.commitments()[j])
            .collect();
        let chosen: Vec<(usize, usize)> = if all || pairs.len() <= 24 {
            pairs
        } else {
            pairs.choose_multiple(&mut rng, 24).copied().collect()
        };
        for (i, j) in chosen {
            sites.push(Site::CommitmentSwap { i, j });
        }
    }
    let replaceable: Vec<usize> = (0..n)
        .filter(|&i| proof.commitments()[i] != other.commitments()[i])
        .collect();
    let chosen: Vec<usize> = if all || replaceable.len() <= 16 {
        replaceable
    } else {
        replaceable.choose_multiple(&mut rng, 16).copied().collect()
    };
    for i in chosen {
        sites.push(Site::CommitmentReplaced { i });
    }
    if n > 0 {
        sites.push(Site::CommitmentDropped);
    }
    for s in PCS::batch_sites(proof.batch_commitment(), other.batch_commitment()) {
        sites.push(Site::BatchCommitment(s));
    }

    if let Some(rop) = proof.reduced_opening_proof() {
        sites.push(Site::ReducedRemoved);
        sites.push(Site::ReducedSpliced);
        for s in PCS::sites(rop.joint_opening_proof()) {
            sites.push(Site::ReducedPcs(s));
        }
    }

    let narg_len = proof.narg().len();
    if narg_len > 0 {
        for i in [0, narg_len / 2, narg_len - 1] {
            sites.push(Site::NargByte { i });
        }
        sites.push(Site::NargTruncated);
    }
    sites.push(Site::NargExtended);

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

/// Apply `site` to `proof` / `io`. `other` is a second honest proof of the
/// same model on different inputs (splice material).
pub(crate) fn apply_site<PCS: PcsMutations>(
    proof: &mut ONNXProof<PCS::Field, PCS>,
    io: &mut ModelExecutionIO,
    other: &ONNXProof<PCS::Field, PCS>,
    site: &Site,
) {
    match site {
        Site::CommitmentSwap { i, j } => proof.commitments_mut().swap(*i, *j),
        Site::CommitmentReplaced { i } => {
            proof.commitments_mut()[*i] = other.commitments()[*i].clone()
        }
        Site::CommitmentDropped => {
            proof.commitments_mut().pop();
        }
        Site::BatchCommitment(s) => {
            PCS::apply_batch(proof.batch_commitment_mut(), other.batch_commitment(), *s)
        }
        Site::ReducedRemoved => *proof.reduced_opening_proof_mut() = None,
        Site::ReducedSpliced => {
            *proof.reduced_opening_proof_mut() = other.reduced_opening_proof().cloned()
        }
        Site::ReducedPcs(s) => PCS::apply(
            proof
                .reduced_opening_proof_mut()
                .as_mut()
                .expect("site")
                .joint_opening_proof_mut(),
            *s,
        ),
        Site::NargByte { i } => proof.narg_mut()[*i] ^= 1,
        Site::NargTruncated => {
            proof.narg_mut().pop();
        }
        Site::NargExtended => proof.narg_mut().push(0),
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
        Site::CommitmentSwap { .. } => "CommitmentSwap",
        Site::CommitmentReplaced { .. } => "CommitmentReplaced",
        Site::CommitmentDropped => "CommitmentDropped",
        Site::BatchCommitment(_) => "BatchCommitment",
        Site::ReducedRemoved => "ReducedRemoved",
        Site::ReducedSpliced => "ReducedSpliced",
        Site::ReducedPcs(_) => "ReducedPcs",
        Site::NargByte { .. } => "NargByte",
        Site::NargTruncated => "NargTruncated",
        Site::NargExtended => "NargExtended",
        Site::IoOutput { .. } => "IoOutput",
        Site::IoInput { .. } => "IoInput",
        Site::IoInputIndex { .. } => "IoInputIndex",
        Site::IoOutputCount => "IoOutputCount",
        Site::IoOutputShape => "IoOutputShape",
        Site::IoInputCount => "IoInputCount",
    }
}
