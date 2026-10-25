//! Soundness sweeps: generic, op-agnostic attacks on the proof system.
//!
//! * [`proof_mutation_sweep`]: every field of a finished proof (and of the
//!   public IO) is perturbed in turn; the verifier must reject, or, for the
//!   few documented dead fields, at least not panic.
//! * [`witness_tamper_sweep`]: the execution trace is perturbed *before*
//!   proving (node outputs, fused quotient / remainder, every softmax
//!   witness column), so the prover's own transcript stays self-consistent
//!   and only the algebra can reject. Any effective tamper must be rejected.
//! * [`opening_forgery_sweep`]: every opening claim the prover makes is
//!   forged (`+1`) at the accumulator choke point and the proof re-run;
//!   this is the `malicious_sub` attack of old, for every opening of every
//!   op.
//! * Binding tests (cross-model constants, input rebinding, output
//!   substitution, padded input lanes) and `known_gap_*` pins.
//!
//! Set `JOLT_SOUNDNESS_EXHAUSTIVE=1` to sweep every site instead of the
//! sampled default; `JOLT_SOUNDNESS_SEED` picks the sample.

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use ark_bn254::Bn254;
use atlas_trace::{trace::ModelExecutionIO, Model, NodeId, Tensor, Trace};
use joltworks::field::bn254::Fr;
use joltworks::poly::{
    commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG},
    opening_proof::{forgery, OpeningId},
};
use joltworks::transcripts::{Blake2bProver, Blake2bVerifier};
use rand::{rngs::StdRng, SeedableRng};
use serial_test::serial;

use super::{
    models::{
        padded_add_model, padded_broadcast_sum_model, probe_models, probe_models_alt, square_model,
        sub_const_model, ProbeModel,
    },
    mutate::{
        apply_site, enumerate_sites, exhaustive, opening_stratum, site_class, Expect, PcsMutations,
        Site,
    },
    probe::{self, ProbeGuard},
};
use crate::onnx_proof::{
    proof_serialization::serialize_proof, prove, prove_from_trace, verify,
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing, ONNXProof,
};

/// The BN254 instantiation the sweeps run under; the sweep bodies are generic over the PCS (and
/// its field).
type Pcs = HyperKZG<Bn254>;

/// What happened when a (possibly tampered) proof met the verifier. The
/// payloads are reported through `Debug`.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) enum Outcome {
    Rejected(String),
    Accepted,
    VerifierPanicked(String),
}

fn payload(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Verify without debug lockstep (tampering must surface as `Err`, not as a
/// transcript-compare panic), catching verifier panics.
pub(crate) fn run_verify<PCS: PcsMutations>(
    proof: &ONNXProof<PCS::Field, PCS>,
    vpp: &AtlasVerifierPreprocessing<PCS::Field, PCS>,
    io: &ModelExecutionIO,
) -> Outcome {
    match probe::expecting_panics(|| {
        catch_unwind(AssertUnwindSafe(|| {
            verify::<PCS::Field, Blake2bVerifier, PCS>(proof, vpp, io, None)
        }))
    }) {
        Ok(Ok(())) => Outcome::Accepted,
        Ok(Err(e)) => Outcome::Rejected(format!("{e:?}")),
        Err(p) => Outcome::VerifierPanicked(payload(p)),
    }
}

/// A proof of a tampered trace, or the prover's panic message.
pub(crate) type TamperedProve<PCS> = Result<
    (
        ONNXProof<<PCS as CommitmentScheme>::Field, PCS>,
        ModelExecutionIO,
    ),
    String,
>;

/// Execute, apply `tamper` to the trace, prove from the tampered trace.
/// Returns the tampered trace too, so callers can tell an effective tamper
/// from a no-op one.
pub(crate) fn prove_tampered<PCS: PcsMutations>(
    pp: &AtlasProverPreprocessing<PCS::Field, PCS>,
    inputs: &[Tensor<i32>],
    tamper: &mut dyn FnMut(&mut Trace),
) -> (Trace, TamperedProve<PCS>) {
    let mut trace = pp.model().trace(inputs).expect("honest execution");
    tamper(&mut trace);
    let tampered = trace.clone();
    let res = match probe::expecting_panics(|| {
        catch_unwind(AssertUnwindSafe(move || {
            prove_from_trace::<PCS::Field, Blake2bProver, PCS>(pp, trace)
        }))
    }) {
        Ok((proof, io, _)) => Ok((proof, io)),
        Err(p) => Err(payload(p)),
    };
    (tampered, res)
}

struct Honest<PCS: CommitmentScheme> {
    name: &'static str,
    prover_pp: AtlasProverPreprocessing<PCS::Field, PCS>,
    verifier_pp: AtlasVerifierPreprocessing<PCS::Field, PCS>,
    inputs: Vec<Tensor<i32>>,
    proof: ONNXProof<PCS::Field, PCS>,
    io: ModelExecutionIO,
}

fn honest<PCS: CommitmentScheme>(m: ProbeModel) -> Honest<PCS> {
    let shared = AtlasSharedPreprocessing::preprocess(m.model);
    let prover_pp = AtlasProverPreprocessing::<PCS::Field, PCS>::new(shared);
    let verifier_pp = AtlasVerifierPreprocessing::from(&prover_pp);
    let (proof, io, _) = prove::<PCS::Field, Blake2bProver, PCS>(&prover_pp, &m.inputs);
    verify::<PCS::Field, Blake2bVerifier, PCS>(&proof, &verifier_pp, &io, None)
        .expect("probe model must prove honestly");
    Honest {
        name: m.name,
        prover_pp,
        verifier_pp,
        inputs: m.inputs,
        proof,
        io,
    }
}

fn report(title: &str, rows: &BTreeMap<&'static str, (usize, usize)>) {
    println!("{title}");
    for (class, (total, ok)) in rows {
        println!("  {class:<20} {ok:>5}/{total}");
    }
}

// ── proof-level mutation ─────────────────────────────────────────────────

#[test]
#[serial(soundness_probe)]
fn proof_mutation_sweep() {
    proof_mutation_sweep_for::<Pcs>();
}

fn proof_mutation_sweep_for<PCS: PcsMutations>() {
    let _probe = ProbeGuard::enter();
    let mut violations: Vec<String> = vec![];
    for (m, alt) in probe_models().into_iter().zip(probe_models_alt()) {
        let h = honest::<PCS>(m);
        let (other, _, _) = prove::<PCS::Field, Blake2bProver, PCS>(&h.prover_pp, &alt.inputs);
        let sites = enumerate_sites::<PCS>(&h.proof, &other, &h.io, h.prover_pp.model());
        // Sequential on purpose: a verifier crash (stack overflow) must be
        // attributable to the site being tried, which the exhaustive run
        // logs site by site.
        let mut outcomes: Vec<(Site, Expect, Outcome)> = Vec::with_capacity(sites.len());
        for (site, expect) in sites {
            if exhaustive() {
                eprintln!("  [{}] trying {site:?}", h.name);
            }
            let mut proof = h.proof.clone();
            let mut io = h.io.clone();
            apply_site(&mut proof, &mut io, &other, &site);
            let outcome = run_verify(&proof, &h.verifier_pp, &io);
            outcomes.push((site, expect, outcome));
        }

        let mut rows: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
        let mut panics_tolerated = 0usize;
        let mut model_violations = vec![];
        for (site, expect, outcome) in &outcomes {
            let ok = match (expect, outcome) {
                (Expect::RejectOrPanic, Outcome::VerifierPanicked(_)) => {
                    panics_tolerated += 1;
                    true
                }
                (_, Outcome::VerifierPanicked(_)) => false,
                (Expect::MustReject | Expect::RejectOrPanic, Outcome::Accepted) => false,
                _ => true,
            };
            let row = rows.entry(site_class(site)).or_default();
            row.0 += 1;
            if ok {
                row.1 += 1;
            } else {
                model_violations.push(format!("{}: {site:?} -> {outcome:?}", h.name));
            }
        }
        for v in &model_violations {
            println!("  VIOLATION {v}");
        }
        println!(
            "  structural removals that panicked the verifier (tolerated): {panics_tolerated}"
        );
        violations.extend(model_violations);
        report(
            &format!(
                "proof mutation sweep [{}]: {} sites{}",
                h.name,
                outcomes.len(),
                if exhaustive() {
                    " (exhaustive)"
                } else {
                    " (sampled)"
                }
            ),
            &rows,
        );
        let tolerated_accepts = outcomes
            .iter()
            .filter(|(_, e, o)| *e == Expect::Tolerated && matches!(o, Outcome::Accepted))
            .count();
        println!("  tolerated dead-field accepts: {tolerated_accepts}");
    }
    assert!(
        violations.is_empty(),
        "{} proof mutations were accepted or crashed the verifier:\n{}",
        violations.len(),
        violations.join("\n")
    );
}

// ── witness-level tamper ─────────────────────────────────────────────────

#[derive(Clone, Debug)]
enum WitnessSite {
    NodeOutput {
        node: usize,
        elem: usize,
    },
    FusedQuotient {
        node: usize,
        elem: usize,
    },
    FusedRemainder {
        node: usize,
        elem: usize,
    },
    Softmax {
        node: usize,
        field: &'static str,
        elem: usize,
    },
}

const SOFTMAX_FIELDS: &[&str] = &[
    "x",
    "max_k",
    "exp_q",
    "exp_sum_q",
    "inv_sum",
    "r",
    "z_hi",
    "z_lo",
    "exp_hi",
    "exp_lo",
    "r_exp",
];

fn softmax_field<'a>(
    w: &'a mut atlas_trace::trace::SoftmaxWitness,
    field: &str,
) -> &'a mut Vec<i32> {
    match field {
        "x" => &mut w.x,
        "max_k" => &mut w.max_k,
        "exp_q" => &mut w.exp_q,
        "exp_sum_q" => &mut w.exp_sum_q,
        "inv_sum" => &mut w.inv_sum,
        "r" => &mut w.r,
        "z_hi" => &mut w.z_hi,
        "z_lo" => &mut w.z_lo,
        "exp_hi" => &mut w.exp_hi,
        "exp_lo" => &mut w.exp_lo,
        "r_exp" => &mut w.r_exp,
        _ => unreachable!(),
    }
}

fn probe_elems(len: usize, all: bool) -> Vec<usize> {
    if len == 0 {
        return vec![];
    }
    // One interior element per tensor by default (each is a full prove);
    // exhaustive mode walks every element.
    let mut v: Vec<usize> = if all {
        (0..len).collect()
    } else {
        vec![len / 2]
    };
    v.sort_unstable();
    v.dedup();
    v
}

fn witness_sites(trace: &Trace, all: bool) -> Vec<WitnessSite> {
    let mut sites = vec![];
    for (node, t) in trace.node_outputs.iter().enumerate() {
        for elem in probe_elems(t.data().len(), all) {
            sites.push(WitnessSite::NodeOutput { node, elem });
        }
    }
    for (id, aux) in &trace.aux {
        let node = id.index();
        match aux {
            atlas_trace::trace::NodeAux::Fused(f) => {
                for elem in probe_elems(f.quotient.data().len(), all) {
                    sites.push(WitnessSite::FusedQuotient { node, elem });
                }
                for elem in probe_elems(f.remainder.data().len(), all) {
                    sites.push(WitnessSite::FusedRemainder { node, elem });
                }
            }
            atlas_trace::trace::NodeAux::Softmax(w) => {
                let mut w = (**w).clone();
                for field in SOFTMAX_FIELDS {
                    let len = softmax_field(&mut w, field).len();
                    for elem in probe_elems(len, false) {
                        sites.push(WitnessSite::Softmax { node, field, elem });
                    }
                }
            }
        }
    }
    sites
}

fn apply_witness(trace: &mut Trace, site: &WitnessSite) {
    match site {
        WitnessSite::NodeOutput { node, elem } => {
            let d = trace.node_outputs[*node].data_mut();
            d[*elem] = d[*elem].wrapping_add(1);
        }
        WitnessSite::FusedQuotient { node, elem } => {
            if let Some(atlas_trace::trace::NodeAux::Fused(f)) =
                trace.aux.get_mut(&NodeId(*node as u32))
            {
                let d = f.quotient.data_mut();
                d[*elem] = d[*elem].wrapping_add(1);
            }
        }
        WitnessSite::FusedRemainder { node, elem } => {
            if let Some(atlas_trace::trace::NodeAux::Fused(f)) =
                trace.aux.get_mut(&NodeId(*node as u32))
            {
                let d = f.remainder.data_mut();
                d[*elem] = d[*elem].wrapping_add(1);
            }
        }
        WitnessSite::Softmax { node, field, elem } => {
            if let Some(atlas_trace::trace::NodeAux::Softmax(w)) =
                trace.aux.get_mut(&NodeId(*node as u32))
            {
                let d = softmax_field(w, field);
                d[*elem] = d[*elem].wrapping_add(1);
            }
        }
    }
}

fn op_name(model: &Model, node: usize) -> String {
    let s = model.graph().nodes[node].op.to_string();
    s.split(['(', '{', '[']).next().unwrap_or(&s).to_string()
}

#[test]
#[serial(soundness_probe)]
fn witness_tamper_sweep() {
    witness_tamper_sweep_for::<Pcs>();
}

fn witness_tamper_sweep_for<PCS: PcsMutations>() {
    let _probe = ProbeGuard::enter();
    let all = exhaustive();
    let mut violations: Vec<String> = vec![];
    let mut refusals: Vec<String> = vec![];
    let mut out_of_domain: Vec<String> = vec![];
    for m in probe_models() {
        let h = honest::<PCS>(m);
        let model = h.prover_pp.model().clone();
        let honest_trace = model.trace(&h.inputs).expect("honest execution");
        let honest_bytes = serialize_proof(&h.proof).expect("serializes");
        {
            // The "proof unchanged" oracle below needs a deterministic prover.
            let (again, _, _) = prove::<PCS::Field, Blake2bProver, PCS>(&h.prover_pp, &h.inputs);
            assert_eq!(
                serialize_proof(&again).unwrap(),
                honest_bytes,
                "proving is not deterministic; the witness oracle cannot rely on proof equality"
            );
        }
        let sites = witness_sites(&honest_trace, all);
        let mut rows: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
        let mut neutral = 0usize;
        for site in &sites {
            let (tampered, res) =
                prove_tampered::<PCS>(&h.prover_pp, &h.inputs, &mut |t| apply_witness(t, site));
            let divergences = probe::take_divergences();
            let class: &'static str = match site {
                WitnessSite::NodeOutput { .. } => "NodeOutput",
                WitnessSite::FusedQuotient { .. } => "FusedQuotient",
                WitnessSite::FusedRemainder { .. } => "FusedRemainder",
                WitnessSite::Softmax { .. } => "Softmax",
            };
            let row = rows.entry(class).or_default();
            row.0 += 1;
            let node = match site {
                WitnessSite::NodeOutput { node, .. }
                | WitnessSite::FusedQuotient { node, .. }
                | WitnessSite::FusedRemainder { node, .. }
                | WitnessSite::Softmax { node, .. } => *node,
            };
            let label = format!("{} node {node} {}: {site:?}", h.name, op_name(&model, node));
            match res {
                Err(p) => {
                    // The prover refused to prove the tampered witness. Not a
                    // pass: the verifier was never consulted.
                    refusals.push(format!("{label} -> prover panicked: {p}"));
                }
                Ok((proof, io)) => {
                    // Neutral tampers, which the verifier may accept:
                    // 1. the proof and IO are byte-identical to the honest
                    //    ones: no prover component read the tampered data
                    //    (e.g. the trace tensor of a node whose claim is
                    //    forwarded by a rewire consumer), so the statement
                    //    proven is still the honest one;
                    // 2. re-executing the claimed inputs reproduces the
                    //    tampered trace exactly: a different but true
                    //    statement with a consistent witness.
                    // An input pushed out of its domain (a non-boolean mask)
                    // does not execute at all; that is reported, not judged.
                    let same_proof = io.inputs == h.io.inputs
                        && io.outputs == h.io.outputs
                        && serialize_proof(&proof).expect("serializes") == honest_bytes;
                    if same_proof {
                        neutral += 1;
                        row.1 += 1;
                        continue;
                    }
                    match model.trace(&io.inputs) {
                        Ok(reexec) if reexec == tampered => {
                            neutral += 1;
                            row.1 += 1;
                            continue;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            out_of_domain
                                .push(format!("{label} -> inputs no longer execute ({e})"));
                            row.1 += 1;
                            continue;
                        }
                    }
                    match run_verify(&proof, &h.verifier_pp, &io) {
                        Outcome::Rejected(_) => row.1 += 1,
                        other => violations.push(format!(
                            "{label} -> {other:?} (prover self-checks that would have fired: {divergences:?})"
                        )),
                    }
                }
            }
        }
        report(
            &format!(
                "witness tamper sweep [{}]: {} sites, {neutral} semantically neutral",
                h.name,
                sites.len()
            ),
            &rows,
        );
        for v in violations.iter().filter(|v| v.starts_with(h.name)) {
            println!("  VIOLATION {v}");
        }
        for r in refusals.iter().filter(|v| v.starts_with(h.name)) {
            println!("  REFUSED {r}");
        }
        for r in out_of_domain.iter().filter(|v| v.starts_with(h.name)) {
            println!("  OUT-OF-DOMAIN {r}");
        }
    }
    if !refusals.is_empty() {
        println!(
            "prover refused {} tampered witnesses (inconclusive, not counted as rejections):",
            refusals.len()
        );
        for r in &refusals {
            println!("  {r}");
        }
    }
    assert!(
        violations.is_empty(),
        "{} effective witness tampers were accepted or crashed the verifier:\n{}",
        violations.len(),
        violations.join("\n")
    );
}

// ── opening forgery ──────────────────────────────────────────────────────

/// Stratified sample of opening ids: one per (polynomial kind, sumcheck
/// kind, op kind of the executing node). Exhaustive mode takes all.
fn forgery_targets<PCS: CommitmentScheme>(
    proof: &ONNXProof<PCS::Field, PCS>,
    model: &Model,
    all: bool,
) -> Vec<OpeningId> {
    let mut seen = std::collections::HashSet::new();
    let mut targets = vec![];
    for id in proof.opening_claims().openings().keys() {
        if all || seen.insert(opening_stratum(id, model)) {
            targets.push(*id);
        }
    }
    targets
}

#[test]
#[serial(soundness_probe)]
fn opening_forgery_sweep() {
    opening_forgery_sweep_for::<Pcs>();
}

fn opening_forgery_sweep_for<PCS: PcsMutations>() {
    let _probe = ProbeGuard::enter();
    let all = exhaustive();
    let mut violations: Vec<String> = vec![];
    let mut unreached: Vec<String> = vec![];
    for m in probe_models() {
        let h = honest::<PCS>(m);
        let targets = forgery_targets(&h.proof, h.prover_pp.model(), all);
        let mut rejected = 0usize;
        for id in &targets {
            forgery::set(Some(*id));
            let (_, res) = prove_tampered::<PCS>(&h.prover_pp, &h.inputs, &mut |_| {});
            let hits = forgery::hits();
            forgery::set(None);
            let divergences = probe::take_divergences();
            if hits == 0 {
                unreached.push(format!("{}: {id:?}", h.name));
                continue;
            }
            match res {
                Err(p) => violations.push(format!(
                    "{}: forging {id:?} made the prover panic ({p}); the verifier was never consulted",
                    h.name
                )),
                Ok((proof, io)) => match run_verify(&proof, &h.verifier_pp, &io) {
                    Outcome::Rejected(_) => rejected += 1,
                    other => violations.push(format!(
                        "{}: forged {id:?} -> {other:?} (self-checks: {divergences:?})",
                        h.name
                    )),
                },
            }
        }
        println!(
            "opening forgery sweep [{}]: {} of {} openings targeted{}, {rejected} rejected",
            h.name,
            targets.len(),
            h.proof.opening_claims().openings().len(),
            if all {
                " (exhaustive)"
            } else {
                " (stratified)"
            }
        );
        for v in violations.iter().filter(|v| v.starts_with(h.name)) {
            println!("  VIOLATION {v}");
        }
    }
    assert!(
        unreached.is_empty(),
        "{} opening ids in the proof were never appended through the accumulator:\n{}",
        unreached.len(),
        unreached.join("\n")
    );
    assert!(
        violations.is_empty(),
        "{} forged openings were accepted or crashed a party:\n{}",
        violations.len(),
        violations.join("\n")
    );
}

// ── binding tests ────────────────────────────────────────────────────────

fn prove_verify_pair(model: Model, inputs: &[Tensor<i32>]) -> Honest<Pcs> {
    honest(ProbeModel {
        name: "binding",
        model,
        inputs: inputs.to_vec(),
    })
}

/// A proof made for model `M'` (one constant coordinate differs) must not
/// verify against `M`'s preprocessing.
#[test]
fn binding_cross_model_constant_rejected() {
    let mut rng = StdRng::seed_from_u64(0xC0157);
    let input = Tensor::random_small(&mut rng, &[64]);
    let c: Vec<i32> = Tensor::random_small(&mut rng, &[64]).data().to_vec();
    let mut c_prime = c.clone();
    c_prime[17] += 1;

    let honest_m = prove_verify_pair(sub_const_model(&c), std::slice::from_ref(&input));
    let forged = prove_verify_pair(sub_const_model(&c_prime), std::slice::from_ref(&input));
    assert_ne!(
        forged.io.outputs[0], honest_m.io.outputs[0],
        "precondition: M' computes a different output"
    );
    let outcome = run_verify(&forged.proof, &honest_m.verifier_pp, &forged.io);
    assert!(
        matches!(outcome, Outcome::Rejected(_)),
        "a proof for M' verified against M: {outcome:?}"
    );
}

/// `square(x) == square(-x)`: the proof of `x` must not be accepted for `-x`
/// (issue #230: inputs are bound into the transcript before any challenge).
#[test]
fn binding_input_rebind_rejected() {
    let mut rng = StdRng::seed_from_u64(0x230);
    let x = Tensor::random_small(&mut rng, &[64]);
    let h = prove_verify_pair(square_model(64), std::slice::from_ref(&x));
    let mut io = h.io.clone();
    let neg: Vec<i32> = x.data().iter().map(|v| -v).collect();
    io.inputs[0] = Tensor::from_slice(&neg, &[64]).unwrap();
    assert_eq!(
        h.prover_pp
            .model()
            .trace(&io.inputs)
            .unwrap()
            .io(h.prover_pp.model())
            .outputs,
        io.outputs,
        "precondition: -x squares to the same output"
    );
    let outcome = run_verify(&h.proof, &h.verifier_pp, &io);
    assert!(
        matches!(outcome, Outcome::Rejected(_)),
        "rebound input accepted: {outcome:?}"
    );
}

/// Any change to the claimed output must be rejected.
#[test]
fn binding_output_substitution_rejected() {
    let mut rng = StdRng::seed_from_u64(0x0071);
    let x = Tensor::random_small(&mut rng, &[64]);
    let h = prove_verify_pair(square_model(64), std::slice::from_ref(&x));
    for elem in [0, 31, 63] {
        let mut io = h.io.clone();
        io.outputs[0].data_mut()[elem] += 1;
        let outcome = run_verify(&h.proof, &h.verifier_pp, &io);
        assert!(
            matches!(outcome, Outcome::Rejected(_)),
            "substituted output element {elem} accepted: {outcome:?}"
        );
    }
}

/// With production padding on, the public input tensor carries padded lanes;
/// changing one must still be rejected (inputs are absorbed wholesale).
#[test]
fn binding_padded_input_lane_tamper_rejected() {
    let mut rng = StdRng::seed_from_u64(0x9ADD);
    let model = padded_add_model(&mut rng);
    let x = Tensor::random_small(&mut rng, &[1000]);
    let h = prove_verify_pair(model, std::slice::from_ref(&x));
    let padded_len = h.io.inputs[0].data().len();
    assert_eq!(
        padded_len, 1024,
        "precondition: the IO carries padded inputs"
    );
    let mut io = h.io.clone();
    io.inputs[0].data_mut()[1010] += 1;
    let outcome = run_verify(&h.proof, &h.verifier_pp, &io);
    assert!(
        matches!(outcome, Outcome::Rejected(_)),
        "padded input lane tamper accepted: {outcome:?}"
    );
}

// ── known gaps (must fail today; flip green when fixed) ──────────────────

mod known_gaps {
    use super::*;

    /// `pad_graph` never consults `PadBehavior`: broadcasting a size-1 axis to
    /// a non-power-of-two extent fills the padded lanes with live values, so
    /// the downstream sum proves `4·v` where the semantics say `3·v`, and the
    /// verifier accepts that false statement. Cross-reference:
    /// `atlas-trace/src/model.rs::broadcast_fill_inflates_padded_reduction`.
    #[test]
    #[ignore = "known gap: Broadcast into padded lanes inflates a downstream reduction \
                (proves 4·v for a semantic 3·v); see atlas-trace model.rs::broadcast_fill_inflates_padded_reduction"]
    fn known_gap_broadcast_padded_lane_inflates_sum() {
        let x = Tensor::from_slice(&[5, 7], &[2, 1]).unwrap();
        let h = prove_verify_pair(padded_broadcast_sum_model(), std::slice::from_ref(&x));
        assert_eq!(
            &h.io.outputs[0].data()[..2],
            &[15, 21],
            "the proven output must be the semantic row sum"
        );
    }

    /// The completeness side of the same gap: the semantically correct
    /// witness (padded lanes zeroed, sums `3·v`) is rejected today because
    /// the broadcast relation is checked over all padded lanes.
    #[test]
    #[ignore = "known gap: the semantically correct padded-lane witness for Broadcast→Sum is rejected; \
                see known_gap_broadcast_padded_lane_inflates_sum"]
    fn known_gap_broadcast_padded_lane_zeroed_witness_rejected() {
        let _probe = ProbeGuard::enter();
        let model = padded_broadcast_sum_model();
        let shared = AtlasSharedPreprocessing::preprocess(model);
        let prover_pp = AtlasProverPreprocessing::<Fr, Pcs>::new(shared);
        let verifier_pp = AtlasVerifierPreprocessing::from(&prover_pp);
        let x = Tensor::from_slice(&[5, 7], &[2, 1]).unwrap();
        let (_, res) = prove_tampered::<Pcs>(&prover_pp, &[x], &mut |t| {
            // node 1 = Broadcast output [2,4] padded: zero lane 3 of each row.
            let bc = t.node_outputs[1].data_mut();
            bc[3] = 0;
            bc[7] = 0;
            // node 2 = Sum output: the semantic sums.
            let s = t.node_outputs[2].data_mut();
            s[0] = 15;
            s[1] = 21;
        });
        let (proof, io) = res.expect("prover accepts the corrected witness");
        let outcome = run_verify(&proof, &verifier_pp, &io);
        assert!(
            matches!(outcome, Outcome::Accepted),
            "the semantically correct witness must verify: {outcome:?}"
        );
    }
}
