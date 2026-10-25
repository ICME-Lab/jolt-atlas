//! Mutation sites over a finished proof and its public IO.
//!
//! Nearly every field of an [`ONNXProof`] is absorbed into the Fiat-Shamir
//! transcript, so a mutated proof is usually rejected because the verifier's
//! transcript diverges rather than because an algebraic check fails. That is
//! still the property we want: *every field is either absorbed or checked*.
//! A site whose mutation is accepted is a dead field (proof malleability) or
//! an orphan claim; a verifier panic is a robustness bug.

use ark_bn254::Bn254;
use ark_std::One;
use atlas_trace::{trace::ModelExecutionIO, Model};
use common::VirtualPoly;
use joltworks::field::bn254::Fr;
use joltworks::{
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, dory::DoryScheme, hyperkzg::HyperKZG},
        opening_proof::{OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN},
    },
    transcripts::Blake2bTranscript,
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

use crate::onnx_proof::{ONNXProof, ProofId};

/// A PCS-specific edit of the joint opening proof.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PcsSite {
    /// `v[i][j] += 1`.
    V { i: usize, j: usize },
    /// Swap two witness commitments.
    SwapW { i: usize, j: usize },
    /// Swap two polynomial commitments.
    SwapCom { i: usize, j: usize },
    /// Replace a polynomial commitment with a witness commitment.
    ComFromW { i: usize },
}

/// How a PCS exposes its opening proof to the sweep.
pub(crate) trait PcsMutations: CommitmentScheme<Field = Fr, Commitment: PartialEq> {
    fn sites(proof: &Self::Proof) -> Vec<PcsSite>;
    fn apply(proof: &mut Self::Proof, site: PcsSite);
}

impl PcsMutations for HyperKZG<Bn254> {
    fn sites(p: &Self::Proof) -> Vec<PcsSite> {
        let mut sites = vec![];
        for (i, row) in p.v.iter().enumerate() {
            for j in 0..row.len() {
                sites.push(PcsSite::V { i, j });
            }
        }
        if p.w.len() >= 2 {
            sites.push(PcsSite::SwapW { i: 0, j: 1 });
        }
        if p.com.len() >= 2 {
            sites.push(PcsSite::SwapCom { i: 0, j: 1 });
        }
        if !p.com.is_empty() && !p.w.is_empty() {
            sites.push(PcsSite::ComFromW { i: 0 });
        }
        sites
    }

    fn apply(p: &mut Self::Proof, site: PcsSite) {
        match site {
            PcsSite::V { i, j } => p.v[i][j] += ark_bn254::Fr::one(),
            PcsSite::SwapW { i, j } => p.w.swap(i, j),
            PcsSite::SwapCom { i, j } => p.com.swap(i, j),
            PcsSite::ComFromW { i } => p.com[i] = p.w[0],
        }
    }
}

impl PcsMutations for DoryScheme {
    fn sites(_: &Self::Proof) -> Vec<PcsSite> {
        vec![]
    }
    fn apply(_: &mut Self::Proof, _: PcsSite) {}
}

/// One mutation of a proof or its IO.
#[derive(Clone, Debug)]
pub(crate) enum Site {
    /// `opening_claims[id] += 1`.
    Claim(OpeningId),
    /// Remove `opening_claims[id]`.
    ClaimRemoved(OpeningId),
    /// Insert a claim for an id nothing consumes.
    ClaimOrphan,
    /// One compressed round-polynomial coefficient `+= 1`.
    RoundCoeff {
        proof: ProofId,
        round: usize,
        k: usize,
    },
    /// An extra coefficient (degree bound).
    RoundExtraCoeff { proof: ProofId, round: usize },
    /// Drop the last round polynomial.
    RoundDropped { proof: ProofId },
    /// Remove a whole sumcheck proof.
    ProofRemoved { proof: ProofId },
    /// Evaluation-reduction `h` coefficient `+= 1`.
    EvalRedCoeff { node: usize, k: usize },
    /// Remove a node's evaluation-reduction proof.
    EvalRedRemoved { node: usize },
    /// Swap two commitments.
    CommitmentSwap { i: usize, j: usize },
    /// Replace a commitment with the same polynomial's commitment from a
    /// second honest execution.
    CommitmentReplaced { i: usize },
    /// Drop the last commitment.
    CommitmentDropped,
    /// Reduced-opening sumcheck coefficient `+= 1`.
    ReducedRoundCoeff { round: usize, k: usize },
    /// `sumcheck_claims[i] += 1`.
    ReducedClaim { i: usize },
    /// No reduced opening proof at all.
    ReducedRemoved,
    /// The reduced opening proof of a second honest execution.
    ReducedSpliced,
    /// A PCS-level edit of the joint opening proof.
    ReducedPcs(PcsSite),
    /// `io.outputs[0][elem] += 1`.
    IoOutput { elem: usize },
    /// `io.inputs[tensor][elem] += 1`.
    IoInput { tensor: usize, elem: usize },
    /// `io.input_indices[i] += 1`.
    IoInputIndex { i: usize },
}

/// What the verifier must do with a mutated proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Expect {
    /// `Err`, never `Ok`, never a panic.
    MustReject,
    /// `Err` or a panic, never `Ok`. A *structurally* malformed proof (a
    /// missing opening claim) currently panics inside the accumulator
    /// instead of returning an error; that is a robustness issue tracked
    /// separately from soundness, so the sweep counts panics here.
    RejectOrPanic,
    /// `Ok` or `Err`, never a panic. Used for fields the verifier is known
    /// not to read (reported, so the list cannot grow unnoticed).
    Tolerated,
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

/// Number of consumers per node (fan-out nodes have a load-bearing `h`).
fn consumer_counts(model: &Model) -> Vec<usize> {
    let nodes = &model.graph().nodes;
    let mut counts = vec![0usize; nodes.len()];
    for n in nodes {
        for i in &n.inputs {
            counts[i.index()] += 1;
        }
    }
    counts
}

/// Every mutation site of `proof` / `io`, with its expectation. Round
/// coefficients are sampled (first round, last round, two random rounds per
/// proof) unless exhaustive.
pub(crate) fn enumerate_sites<PCS: PcsMutations>(
    proof: &ONNXProof<Fr, Blake2bTranscript, PCS>,
    other: &ONNXProof<Fr, Blake2bTranscript, PCS>,
    io: &ModelExecutionIO,
    model: &Model,
) -> Vec<(Site, Expect)> {
    let mut rng = StdRng::seed_from_u64(sampling_seed());
    let all = exhaustive();
    let mut sites = vec![];

    let mut strata = std::collections::HashSet::new();
    for id in proof.opening_claims().openings().keys() {
        if all || strata.insert(opening_stratum(id, model)) {
            sites.push((Site::Claim(*id), Expect::MustReject));
            sites.push((Site::ClaimRemoved(*id), Expect::RejectOrPanic));
        }
    }
    sites.push((Site::ClaimOrphan, Expect::Tolerated));

    for (id, sc) in proof.proofs() {
        let rounds = sc.compressed_polys.len();
        let chosen: Vec<usize> = if all || rounds <= 4 {
            (0..rounds).collect()
        } else {
            let mut v = vec![0, rounds - 1];
            while v.len() < 4 {
                let r = rng.gen_range(1..rounds - 1);
                if !v.contains(&r) {
                    v.push(r);
                }
            }
            v
        };
        for round in chosen {
            for k in 0..sc.compressed_polys[round].coeffs_except_linear_term.len() {
                sites.push((
                    Site::RoundCoeff {
                        proof: id.clone(),
                        round,
                        k,
                    },
                    Expect::MustReject,
                ));
            }
        }
        if rounds > 0 {
            sites.push((
                Site::RoundExtraCoeff {
                    proof: id.clone(),
                    round: 0,
                },
                Expect::MustReject,
            ));
            sites.push((Site::RoundDropped { proof: id.clone() }, Expect::MustReject));
        }
        sites.push((Site::ProofRemoved { proof: id.clone() }, Expect::MustReject));
    }

    let consumers = consumer_counts(model);
    for (node, er) in proof.eval_reduction_proofs() {
        // A single-consumer node's `h` is never read (short path in the
        // eval-reduction verifier): a documented dead field.
        let expect = if consumers[*node] > 1 {
            Expect::MustReject
        } else {
            Expect::Tolerated
        };
        for k in 0..er.h.coeffs.len() {
            sites.push((Site::EvalRedCoeff { node: *node, k }, expect));
        }
        sites.push((Site::EvalRedRemoved { node: *node }, Expect::MustReject));
    }

    // Swapping or replacing a commitment is a no-op when the two polynomials
    // (or the two executions' versions of one polynomial) are identical,
    // e.g. all-zero one-hot chunks of a clamp bucket nothing saturates in.
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
            sites.push((Site::CommitmentSwap { i, j }, Expect::MustReject));
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
        sites.push((Site::CommitmentReplaced { i }, Expect::MustReject));
    }
    if n > 0 {
        sites.push((Site::CommitmentDropped, Expect::MustReject));
    }

    if let Some(rop) = proof.reduced_opening_proof() {
        for (round, poly) in rop.sumcheck_proof().compressed_polys.iter().enumerate() {
            for k in 0..poly.coeffs_except_linear_term.len() {
                sites.push((Site::ReducedRoundCoeff { round, k }, Expect::MustReject));
            }
        }
        let claims = rop.sumcheck_claims().len();
        let chosen: Vec<usize> = if all || claims <= 8 {
            (0..claims).collect()
        } else {
            (0..claims)
                .collect::<Vec<_>>()
                .choose_multiple(&mut rng, 8)
                .copied()
                .collect()
        };
        for i in chosen {
            sites.push((Site::ReducedClaim { i }, Expect::MustReject));
        }
        sites.push((Site::ReducedRemoved, Expect::MustReject));
        sites.push((Site::ReducedSpliced, Expect::MustReject));
        for s in PCS::sites(rop.joint_opening_proof()) {
            sites.push((Site::ReducedPcs(s), Expect::MustReject));
        }
    }

    let out_len = io.outputs[0].data().len();
    for elem in [0, out_len / 2, out_len - 1] {
        sites.push((Site::IoOutput { elem }, Expect::MustReject));
    }
    for (tensor, t) in io.inputs.iter().enumerate() {
        let len = t.data().len();
        for elem in [0, len / 2, len - 1] {
            sites.push((Site::IoInput { tensor, elem }, Expect::MustReject));
        }
    }
    for i in 0..io.input_indices.len() {
        sites.push((Site::IoInputIndex { i }, Expect::MustReject));
    }

    // Dedupe sites that collapsed onto the same element.
    let mut seen = std::collections::HashSet::new();
    sites.retain(|(s, _)| seen.insert(format!("{s:?}")));
    sites
}

/// Apply `site` to `proof` / `io`. `other` is a second honest proof of the
/// same model on different inputs (splice material).
pub(crate) fn apply_site<PCS: PcsMutations>(
    proof: &mut ONNXProof<Fr, Blake2bTranscript, PCS>,
    io: &mut ModelExecutionIO,
    other: &ONNXProof<Fr, Blake2bTranscript, PCS>,
    site: &Site,
) {
    let one = Fr::one();
    match site {
        Site::Claim(id) => {
            proof
                .opening_claims_mut()
                .openings_mut()
                .get_mut(id)
                .expect("site")
                .1 += one
        }
        Site::ClaimRemoved(id) => {
            proof.opening_claims_mut().openings_mut().remove(id);
        }
        Site::ClaimOrphan => {
            let id = OpeningId::new(
                VirtualPoly::NodeOutput(usize::MAX - 7),
                SumcheckId::NodeExecution(usize::MAX - 7),
            );
            proof.opening_claims_mut().openings_mut().insert(
                id,
                (OpeningPoint::<BIG_ENDIAN, Fr>::default(), Fr::from(7u64)),
            );
        }
        Site::RoundCoeff {
            proof: id,
            round,
            k,
        } => {
            proof
                .proofs_mut()
                .get_mut(id)
                .expect("site")
                .compressed_polys[*round]
                .coeffs_except_linear_term[*k] += one
        }
        Site::RoundExtraCoeff { proof: id, round } => proof
            .proofs_mut()
            .get_mut(id)
            .expect("site")
            .compressed_polys[*round]
            .coeffs_except_linear_term
            .push(one),
        Site::RoundDropped { proof: id } => {
            proof
                .proofs_mut()
                .get_mut(id)
                .expect("site")
                .compressed_polys
                .pop();
        }
        Site::ProofRemoved { proof: id } => {
            proof.proofs_mut().remove(id);
        }
        Site::EvalRedCoeff { node, k } => {
            proof
                .eval_reduction_proofs_mut()
                .get_mut(node)
                .expect("site")
                .h
                .coeffs[*k] += one
        }
        Site::EvalRedRemoved { node } => {
            proof.eval_reduction_proofs_mut().remove(node);
        }
        Site::CommitmentSwap { i, j } => proof.commitments_mut().swap(*i, *j),
        Site::CommitmentReplaced { i } => {
            proof.commitments_mut()[*i] = other.commitments()[*i].clone()
        }
        Site::CommitmentDropped => {
            proof.commitments_mut().pop();
        }
        Site::ReducedRoundCoeff { round, k } => {
            proof
                .reduced_opening_proof_mut()
                .as_mut()
                .expect("site")
                .sumcheck_proof_mut()
                .compressed_polys[*round]
                .coeffs_except_linear_term[*k] += one
        }
        Site::ReducedClaim { i } => {
            proof
                .reduced_opening_proof_mut()
                .as_mut()
                .expect("site")
                .sumcheck_claims_mut()[*i] += one
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
        Site::IoOutput { elem } => {
            let d = io.outputs[0].data_mut();
            d[*elem] = d[*elem].wrapping_add(1);
        }
        Site::IoInput { tensor, elem } => {
            let d = io.inputs[*tensor].data_mut();
            d[*elem] = d[*elem].wrapping_add(1);
        }
        Site::IoInputIndex { i } => io.input_indices[*i] += 1,
    }
}

/// A short label for grouping sites in reports.
pub(crate) fn site_class(site: &Site) -> &'static str {
    match site {
        Site::Claim(_) => "Claim",
        Site::ClaimRemoved(_) => "ClaimRemoved",
        Site::ClaimOrphan => "ClaimOrphan",
        Site::RoundCoeff { .. } => "RoundCoeff",
        Site::RoundExtraCoeff { .. } => "RoundExtraCoeff",
        Site::RoundDropped { .. } => "RoundDropped",
        Site::ProofRemoved { .. } => "ProofRemoved",
        Site::EvalRedCoeff { .. } => "EvalRedCoeff",
        Site::EvalRedRemoved { .. } => "EvalRedRemoved",
        Site::CommitmentSwap { .. } => "CommitmentSwap",
        Site::CommitmentReplaced { .. } => "CommitmentReplaced",
        Site::CommitmentDropped => "CommitmentDropped",
        Site::ReducedRoundCoeff { .. } => "ReducedRoundCoeff",
        Site::ReducedClaim { .. } => "ReducedClaim",
        Site::ReducedRemoved => "ReducedRemoved",
        Site::ReducedSpliced => "ReducedSpliced",
        Site::ReducedPcs(_) => "ReducedPcs",
        Site::IoOutput { .. } => "IoOutput",
        Site::IoInput { .. } => "IoInput",
        Site::IoInputIndex { .. } => "IoInputIndex",
    }
}
