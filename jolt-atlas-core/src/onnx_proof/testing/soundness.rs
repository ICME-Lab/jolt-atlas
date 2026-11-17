//! Soundness sweeps: generic, op-agnostic attacks on the proof system.
//!
//! * [`proof_mutation_sweep`]: every field of a finished proof (and of the
//!   public IO) is perturbed in turn; the verifier must reject, or, for the
//!   few documented dead fields, at least not panic.
//! * [`witness_tamper_sweep`]: the execution trace is perturbed *before*
//!   proving (node outputs, fused remainders, every softmax
//!   witness column), and so are the range buckets' digits, which the trace
//!   does not hold (out of range with the
//!   value kept, and the top digit raised by the base), so the prover's own
//!   transcript stays self-consistent
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

use atlas_pcs::{akita::AkitaScheme, commitment_scheme::CommitmentScheme};
use atlas_sumcheck::opening_proof::{forgery, OpeningId};
use atlas_trace::{trace::ModelExecutionIO, Model, NodeId, Op, Tensor, Trace};
use atlas_transcript::{Blake2bProver, Blake2bVerifier};
use common::consts::MODEL_SCALE;
use rand::{rngs::StdRng, SeedableRng};
use serial_test::serial;

use super::{
    model_builder::ModelBuilder,
    models::{
        padded_add_model, padded_broadcast_sum_model, probe_models, probe_models_alt, square_model,
        sub_const_model, ProbeModel,
    },
    mutate::{
        apply_site, enumerate_sites, exhaustive, opening_stratum, site_class, PcsMutations, Site,
    },
    probe::{self, ProbeGuard},
};
use crate::onnx_proof::{
    proof_serialization::serialize_proof,
    prove, prove_from_trace,
    range_bucket::{remainder_buckets, BucketWitnesses},
    range_check::{
        self,
        digit_tamper::{self, DigitTamper, DigitTamperForm},
    },
    verify, AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    ONNXProof,
};

/// The instantiation the sweeps run under: Akita over Fp128. The sweep bodies are generic over the
/// PCS (and its field).
type Pcs = AkitaScheme;

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
        let mut outcomes: Vec<(Site, Outcome)> = Vec::with_capacity(sites.len());
        for site in sites {
            if exhaustive() {
                eprintln!("  [{}] trying {site:?}", h.name);
            }
            let mut proof = h.proof.clone();
            let mut io = h.io.clone();
            apply_site(&mut proof, &mut io, &other, &site);
            let outcome = run_verify(&proof, &h.verifier_pp, &io);
            outcomes.push((site, outcome));
        }

        let mut rows: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
        let mut model_violations = vec![];
        for (site, outcome) in &outcomes {
            // Every site must be rejected with an `Err`: an accept is a soundness hole, a
            // panic a verifier crash.
            let ok = matches!(outcome, Outcome::Rejected(_));
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
    FusedRemainder {
        node: usize,
        elem: usize,
    },
    Softmax {
        node: usize,
        field: &'static str,
        elem: usize,
    },
    /// One element of a division, rsqrt or trig-period advice tensor (see [`bump_advice`]).
    Advice {
        node: usize,
        field: &'static str,
        elem: usize,
    },
    /// A range bucket's digits at a cycle of member `node`, tampered through [`digit_tamper`]
    /// (the trace holds no digits). The trace is unchanged, so no neutrality rule applies: the
    /// verifier must reject.
    BucketDigit {
        node: usize,
        bucket: usize,
        cycle: usize,
        form: DigitTamperForm,
    },
}

const SOFTMAX_FIELDS: &[&str] = &["shift", "address", "sum_gap", "prev_sum_gap"];

/// The length of a softmax witness field.
fn softmax_field_len(w: &atlas_trace::trace::SoftmaxWitness, field: &str) -> usize {
    match field {
        "shift" => w.shift.len(),
        "address" => w.address.len(),
        "sum_gap" => w.sum_gap.len(),
        "prev_sum_gap" => w.prev_sum_gap.len(),
        _ => unreachable!(),
    }
}

/// Add one to element `elem` of a softmax witness field.
fn bump_softmax_field(w: &mut atlas_trace::trace::SoftmaxWitness, field: &str, elem: usize) {
    match field {
        "shift" => w.shift[elem] = w.shift[elem].wrapping_add(1),
        "address" => w.address[elem] = w.address[elem].wrapping_add(1),
        "sum_gap" => w.sum_gap[elem] = w.sum_gap[elem].wrapping_add(1),
        "prev_sum_gap" => w.prev_sum_gap[elem] = w.prev_sum_gap[elem].wrapping_add(1),
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

fn witness_sites(model: &Model, trace: &Trace, all: bool) -> Vec<WitnessSite> {
    let mut sites = vec![];
    let bucket_witnesses = BucketWitnesses::build(model, trace);
    for bucket in remainder_buckets(model).expect("bucket layout") {
        let (values, _) = bucket_witnesses.remainder_values(bucket.idx);
        // Every member is its own range claim.
        for member in &bucket.nodes {
            // The keep-value form moves a digit above the least significant one, so it needs a
            // cell that has one: a member whose values are all below the digit base gets the
            // raise-top form only.
            let keep_value_cycle = (member.offset..member.offset + (1 << member.log_t))
                .find(|&t| values[t] >= range_check::DIGIT_BASE);
            let forms = [
                (DigitTamperForm::KeepValue, keep_value_cycle),
                (DigitTamperForm::RaiseTop, Some(member.offset)),
            ];
            for (form, cycle) in forms {
                if let Some(cycle) = cycle {
                    sites.push(WitnessSite::BucketDigit {
                        node: member.idx,
                        bucket: bucket.idx,
                        cycle,
                        form,
                    });
                }
            }
        }
    }
    for (node, t) in trace.node_outputs.iter().enumerate() {
        for elem in probe_elems(t.data().len(), all) {
            sites.push(WitnessSite::NodeOutput { node, elem });
        }
    }
    for (id, aux) in &trace.aux {
        let node = id.index();
        match aux {
            atlas_trace::trace::NodeAux::Fused(f) => {
                for elem in probe_elems(f.remainder.data().len(), all) {
                    sites.push(WitnessSite::FusedRemainder { node, elem });
                }
            }
            atlas_trace::trace::NodeAux::Softmax(w) => {
                for field in SOFTMAX_FIELDS {
                    let len = softmax_field_len(w, field);
                    for elem in probe_elems(len, false) {
                        sites.push(WitnessSite::Softmax { node, field, elem });
                    }
                }
            }
            atlas_trace::trace::NodeAux::Remainder(_)
            | atlas_trace::trace::NodeAux::Rsqrt(_)
            | atlas_trace::trace::NodeAux::TrigPeriod(_) => {
                let len = trace.node_outputs[node].data().len();
                for &field in advice_fields(aux) {
                    for elem in probe_elems(len, all) {
                        sites.push(WitnessSite::Advice { node, field, elem });
                    }
                }
            }
        }
    }
    sites
}

/// The tamperable tensors of a division, rsqrt or trig-period advice.
fn advice_fields(aux: &atlas_trace::trace::NodeAux) -> &'static [&'static str] {
    use atlas_trace::trace::NodeAux;
    match aux {
        NodeAux::Remainder(_) => &["remainder"],
        NodeAux::Rsqrt(_) => &["quotient", "div_remainder", "sqrt_remainder"],
        NodeAux::TrigPeriod(_) => &["trig_period_quotient", "trig_period_remainder"],
        NodeAux::Fused(_) | NodeAux::Softmax(_) => &[],
    }
}

/// Add `delta` to element `elem` of the advice tensor `field`.
fn bump_advice(aux: &mut atlas_trace::trace::NodeAux, field: &str, elem: usize, delta: i64) {
    use atlas_trace::trace::NodeAux;
    let bump = |d: &mut [i32]| d[elem] = d[elem].wrapping_add(delta as i32);
    match (aux, field) {
        (NodeAux::Remainder(r), "remainder") => bump(r.data_mut()),
        (NodeAux::Rsqrt(a), "quotient") => {
            let d = a.quotient.data_mut();
            d[elem] = d[elem].wrapping_add(delta);
        }
        (NodeAux::Rsqrt(a), "div_remainder") => bump(a.div_remainder.data_mut()),
        (NodeAux::Rsqrt(a), "sqrt_remainder") => bump(a.sqrt_remainder.data_mut()),
        (NodeAux::TrigPeriod(a), "trig_period_quotient") => bump(a.quotient.data_mut()),
        (NodeAux::TrigPeriod(a), "trig_period_remainder") => bump(a.remainder.data_mut()),
        (_, field) => panic!("no advice field `{field}` on this node"),
    }
}

fn apply_witness(trace: &mut Trace, site: &WitnessSite) {
    match site {
        WitnessSite::NodeOutput { node, elem } => {
            let d = trace.node_outputs[*node].data_mut();
            d[*elem] = d[*elem].wrapping_add(1);
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
                bump_softmax_field(w, field, *elem);
            }
        }
        WitnessSite::Advice { node, field, elem } => {
            let aux = trace
                .aux
                .get_mut(&NodeId(*node as u32))
                .expect("the site was enumerated from this node's advice");
            bump_advice(aux, field, *elem, 1);
        }
        WitnessSite::BucketDigit {
            bucket,
            cycle,
            form,
            ..
        } => digit_tamper::set(Some(DigitTamper {
            bucket: *bucket,
            cycle: *cycle,
            form: *form,
        })),
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

/// Which cells of each node some model output depends on, by the graph's structure alone. A
/// model output's cells are all live; a `Slice` reads only the cells it selects; an elementwise
/// op reads cell `i` of a same-shaped operand for its cell `i`; every other op is taken to read
/// all of its operands whenever one of its cells is live.
fn live_cells(model: &Model) -> Vec<Vec<bool>> {
    let nodes = &model.graph().nodes;
    let len = |dims: &[usize]| dims.iter().product::<usize>();
    let mut live: Vec<Vec<bool>> = nodes
        .iter()
        .map(|node| vec![false; len(&node.output_dims)])
        .collect();
    for output in model.outputs() {
        live[output.index()].fill(true);
    }
    for (idx, node) in nodes.iter().enumerate().rev() {
        let node_live = live[idx].clone();
        let any = node_live.iter().any(|&cell| cell);
        let elementwise = matches!(
            node.op,
            Op::Add(_)
                | Op::Sub(_)
                | Op::Mul(_)
                | Op::Neg(_)
                | Op::Identity(_)
                | Op::ReLU(_)
                | Op::SmoothRelu(_)
                | Op::Clamp(_)
                | Op::Square(_)
                | Op::Cube(_)
                | Op::Div(_)
                | Op::ScalarConstDiv(_)
                | Op::Rsqrt(_)
                | Op::Tanh(_)
                | Op::Erf(_)
                | Op::Sigmoid(_)
                | Op::Sin(_)
                | Op::Cos(_)
                | Op::And(_)
                | Op::Iff(_)
                | Op::IsNan(_)
        );
        for input in &node.inputs {
            let input = input.index();
            let input_dims = nodes[input].output_dims.clone();
            match &node.op {
                Op::Slice(slice) => {
                    for (cell, _) in node_live.iter().enumerate().filter(|(_, &cell)| cell) {
                        let mut coord =
                            crate::utils::dims::linear_to_coord(cell, &node.output_dims);
                        coord[slice.axis] += slice.start;
                        let source = crate::utils::dims::coord_to_linear(&coord, &input_dims)
                            .expect("a slice's cells are cells of its operand");
                        live[input][source] = true;
                    }
                }
                _ if elementwise && input_dims == node.output_dims => {
                    for (source, &cell) in live[input].iter_mut().zip(&node_live) {
                        *source |= cell;
                    }
                }
                _ if any => live[input].fill(true),
                _ => {}
            }
        }
    }
    live
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
        let live = live_cells(&model);
        let sites = witness_sites(&model, &honest_trace, all);
        let mut rows: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
        let mut neutral = 0usize;
        for site in &sites {
            let (tampered, res) =
                prove_tampered::<PCS>(&h.prover_pp, &h.inputs, &mut |t| apply_witness(t, site));
            digit_tamper::set(None);
            let divergences = probe::take_divergences();
            let class: &'static str = match site {
                WitnessSite::NodeOutput { .. } => "NodeOutput",
                WitnessSite::FusedRemainder { .. } => "FusedRemainder",
                WitnessSite::Softmax { .. } => "Softmax",
                WitnessSite::Advice { .. } => "Advice",
                WitnessSite::BucketDigit { .. } => "BucketDigit",
            };
            let row = rows.entry(class).or_default();
            row.0 += 1;
            let node = match site {
                WitnessSite::NodeOutput { node, .. }
                | WitnessSite::FusedRemainder { node, .. }
                | WitnessSite::Softmax { node, .. }
                | WitnessSite::Advice { node, .. }
                | WitnessSite::BucketDigit { node, .. } => *node,
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
                    // 3. the tampered cell is one no model output depends on
                    //    ([`live_cells`]): a layout tie opens its operand only on
                    //    the cells it selects, so the cells a `Slice` discards
                    //    are not part of the statement.
                    // An input pushed out of its domain (a non-boolean mask)
                    // does not execute at all; that is reported, not judged.
                    let same_proof = io.inputs == h.io.inputs
                        && io.outputs == h.io.outputs
                        && serialize_proof(&proof).expect("serializes") == honest_bytes;
                    if matches!(site, WitnessSite::BucketDigit { .. }) {
                        // The trace is untouched, so neither rule applies; an unchanged proof
                        // means the tamper never reached the digits.
                        match (same_proof, run_verify(&proof, &h.verifier_pp, &io)) {
                            (false, Outcome::Rejected(_)) => row.1 += 1,
                            (true, _) => violations.push(format!(
                                "{label} -> the digit tamper did not change the proof"
                            )),
                            (false, other) => violations.push(format!(
                                "{label} -> {other:?} (prover self-checks that would have fired: {divergences:?})"
                            )),
                        }
                        continue;
                    }
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
                    let dead_cell = match site {
                        WitnessSite::NodeOutput { node, elem }
                        | WitnessSite::FusedRemainder { node, elem }
                        | WitnessSite::Advice { node, elem, .. } => {
                            live[*node].get(*elem).is_some_and(|&cell| !cell)
                        }
                        WitnessSite::Softmax { node, .. } => !live[*node].iter().any(|&cell| cell),
                        WitnessSite::BucketDigit { .. } => false,
                    };
                    match run_verify(&proof, &h.verifier_pp, &io) {
                        Outcome::Rejected(_) => row.1 += 1,
                        Outcome::Accepted if dead_cell => {
                            neutral += 1;
                            row.1 += 1;
                        }
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
        let prover_pp =
            AtlasProverPreprocessing::<<Pcs as CommitmentScheme>::Field, Pcs>::new(shared);
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

// ── fuzz regressions ─────────────────────────────────────────────────────

/// Replay the `verify` fuzz target's driver (`onnx_proof::fuzz`) on the honest input, on fixed
/// truncations, byte flips and IO edits of it, and on every checked-in regression input
/// (`fuzz/regressions/`). The verifier returning at all is the property: a panic fails the test.
#[test]
fn fuzz_regressions_replay() {
    use crate::onnx_proof::fuzz::{VerifyTarget, HEADER_LEN};
    use atlas_algebra::field::fp128::Fp128;
    use atlas_pcs::akita::AkitaScheme;

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = VerifyTarget::<Fp128, AkitaScheme>::new(&manifest_dir.join(".."));
    let seed = target.seed().to_vec();
    assert!(
        matches!(target.run(&seed), Some(Ok(()))),
        "the honest input must verify"
    );

    // Every IO edit of the honest proof is rejected.
    for kind in 1..=7u8 {
        let mut input = seed.clone();
        input[..5].copy_from_slice(&[kind, 0, 1, 0, 1]);
        assert!(
            !matches!(target.run(&input), Some(Ok(()))),
            "IO edit {kind} was accepted"
        );
    }
    // Truncations and byte flips spread over the proof: any verdict, no panic.
    let len = seed.len();
    for cut in [HEADER_LEN, HEADER_LEN + 1, len / 4, len / 2, len - 1] {
        let _ = target.run(&seed[..cut]);
    }
    for step in 0..64 {
        let mut input = seed.clone();
        let at = HEADER_LEN + step * (len - HEADER_LEN) / 64;
        input[at] ^= 0x80;
        let _ = target.run(&input);
    }

    let regressions = manifest_dir.join("fuzz/regressions");
    for entry in std::fs::read_dir(&regressions).expect("fuzz/regressions exists") {
        let path = entry.expect("readable entry").path();
        if path.file_name().is_some_and(|name| name == ".gitkeep") {
            continue;
        }
        let _ = target.run(&std::fs::read(&path).expect("readable regression input"));
    }
}

// ── remainder bounds ─────────────────────────────────────────────────────
//
// Each test below moves a quotient by one and its remainder by the bound in the opposite
// direction, so the op's own identity still holds and only the remainder's range claim in the
// packed remainder buckets can reject. The op is the model output, so no consumer relation
// sees the forged quotient.

/// An edit of the honest trace.
type Forgery = Box<dyn FnMut(&mut Trace)>;

/// Prove `model` on `inputs` from a trace edited by `tamper`, and verify. A prover that
/// refuses is reported as a rejection: no proof was accepted.
fn bound_outcome(
    model: &Model,
    inputs: &[Tensor<i32>],
    tamper: &mut dyn FnMut(&mut Trace),
) -> Outcome {
    let prover_pp = AtlasProverPreprocessing::<<Pcs as CommitmentScheme>::Field, Pcs>::new(
        AtlasSharedPreprocessing::preprocess(model.clone()),
    );
    let verifier_pp = AtlasVerifierPreprocessing::from(&prover_pp);
    match prove_tampered::<Pcs>(&prover_pp, inputs, tamper).1 {
        Ok((proof, io)) => run_verify(&proof, &verifier_pp, &io),
        Err(panic) => Outcome::Rejected(format!("prover refused: {panic}")),
    }
}

/// The honest trace verifies and every forged one is rejected with an `Err`.
fn assert_bound_holds(
    what: &str,
    model: &Model,
    inputs: &[Tensor<i32>],
    forgeries: Vec<(&str, Forgery)>,
) {
    let _probe = ProbeGuard::enter();
    assert!(
        matches!(bound_outcome(model, inputs, &mut |_| {}), Outcome::Accepted),
        "{what}: the honest trace must verify"
    );
    for (name, mut forge) in forgeries {
        let outcome = bound_outcome(model, inputs, &mut *forge);
        assert!(
            matches!(outcome, Outcome::Rejected(_)),
            "{what}, {name}: {outcome:?}"
        );
    }
}

/// The node's `Remainder` advice.
fn remainder_mut(trace: &mut Trace, node: usize) -> &mut [i32] {
    match trace.aux.get_mut(&NodeId(node as u32)) {
        Some(atlas_trace::trace::NodeAux::Remainder(r)) => r.data_mut(),
        _ => panic!("node {node} has no remainder advice"),
    }
}

/// A quotient that is the node's output moved by `step`, its remainder by `−step·bound`.
fn forge_output_quotient(node: usize, elem: usize, step: i32, bound: i32) -> Forgery {
    Box::new(move |trace| {
        trace.node_outputs[node].data_mut()[elem] += step;
        remainder_mut(trace, node)[elem] -= step * bound;
    })
}

#[test]
#[serial(soundness_probe)]
fn div_remainder_is_bounded() {
    let mut b = ModelBuilder::new();
    let x = b.input(vec![8]);
    let divisor = b.constant(Tensor::from_slice(&[7; 8], &[8]).unwrap());
    let out = b.div(x, divisor);
    b.mark_output(out);
    let inputs = vec![Tensor::from_slice(&[100, -100, 3, 46, 1 << 20, -5, 13, 6], &[8]).unwrap()];
    assert_bound_holds(
        "Div",
        &b.build(),
        &inputs,
        vec![
            ("quotient − 1", forge_output_quotient(out, 3, -1, 7)),
            ("quotient + 1", forge_output_quotient(out, 3, 1, 7)),
        ],
    );
}

#[test]
#[serial(soundness_probe)]
fn scalar_const_div_remainder_is_bounded() {
    // A divisor that is not a power of two (the two-sided claim), and `2^14` (one claim).
    for divisor in [7, 1 << 14] {
        let mut b = ModelBuilder::new();
        let x = b.input(vec![8]);
        let out = b.scalar_const_div(x, divisor);
        b.mark_output(out);
        let inputs =
            vec![Tensor::from_slice(&[100, -100, 3, 46, 1 << 20, -5, 13, 99_999], &[8]).unwrap()];
        assert_bound_holds(
            &format!("ScalarConstDiv by {divisor}"),
            &b.build(),
            &inputs,
            vec![
                ("quotient − 1", forge_output_quotient(out, 4, -1, divisor)),
                ("quotient + 1", forge_output_quotient(out, 4, 1, divisor)),
            ],
        );
    }
}

/// The node's `Rsqrt` advice.
fn rsqrt_mut(trace: &mut Trace, node: usize) -> &mut atlas_trace::RsqrtAdvice {
    match trace.aux.get_mut(&NodeId(node as u32)) {
        Some(atlas_trace::trace::NodeAux::Rsqrt(a)) => a,
        _ => panic!("node {node} has no rsqrt advice"),
    }
}

#[test]
#[serial(soundness_probe)]
fn rsqrt_remainders_are_bounded() {
    let mut b = ModelBuilder::new();
    let x = b.input(vec![8]);
    let out = b.rsqrt(x);
    b.mark_output(out);
    let model = b.build();
    let values = [1000, 3000, 7777, 12_345, 50_000, 99_991, 250_007, 1_000_003];
    let inputs = vec![Tensor::from_slice(&values, &[8]).unwrap()];

    // An element whose square-root remainder leaves room both ways: 1 ≤ r_sqrt < 2·out.
    let honest = model.trace(&inputs).expect("honest execution");
    let advice = honest.rsqrt(NodeId(out as u32)).expect("rsqrt advice");
    let elem = (0..values.len())
        .find(|&i| {
            let r = i64::from(advice.sqrt_remainder.data()[i]);
            1 <= r && r < 2 * i64::from(honest.node_outputs[out].data()[i])
        })
        .expect("an element with a square-root remainder strictly inside its range");
    let x_elem = values[elem];

    // Division: S³ = q·x + r_div and q = out² + r_sqrt both still hold.
    let forge_division = |step: i64| -> Forgery {
        Box::new(move |trace| {
            let advice = rsqrt_mut(trace, out);
            advice.quotient.data_mut()[elem] += step;
            advice.div_remainder.data_mut()[elem] -= (step as i32) * x_elem;
            advice.sqrt_remainder.data_mut()[elem] += step as i32;
        })
    };
    // Square root: q = out² + r_sqrt still holds with the output moved.
    let forge_sqrt = |step: i32| -> Forgery {
        Box::new(move |trace| {
            let v = trace.node_outputs[out].data()[elem];
            trace.node_outputs[out].data_mut()[elem] = v + step;
            rsqrt_mut(trace, out).sqrt_remainder.data_mut()[elem] -= step * (2 * v + step);
        })
    };
    assert_bound_holds(
        "Rsqrt",
        &model,
        &inputs,
        vec![
            ("division quotient − 1", forge_division(-1)),
            ("division quotient + 1", forge_division(1)),
            ("output − 1", forge_sqrt(-1)),
            ("output + 1", forge_sqrt(1)),
        ],
    );
}

#[test]
#[serial(soundness_probe)]
fn mean_of_squares_remainder_is_bounded() {
    let (rows, count) = (8usize, 24usize);
    let mut b = ModelBuilder::new();
    let x = b.input(vec![rows, count]);
    let out = b.mean_of_squares(x, vec![1], vec![rows, 1]);
    b.mark_output(out);
    let mut rng = StdRng::seed_from_u64(0xB0D5);
    let inputs = vec![Tensor::random_range(
        &mut rng,
        &[rows, count],
        -(1 << 15)..(1 << 15),
    )];
    let divisor = (count as i32) << MODEL_SCALE;
    let forge = |step: i32| -> Forgery {
        Box::new(move |trace| {
            trace.node_outputs[out].data_mut()[2] += step;
            let Some(atlas_trace::trace::NodeAux::Fused(f)) =
                trace.aux.get_mut(&NodeId(out as u32))
            else {
                panic!("MeanOfSquares has fused intermediates");
            };
            f.remainder.data_mut()[2] -= step * divisor;
        })
    };
    assert_bound_holds(
        "MeanOfSquares",
        &b.build(),
        &inputs,
        vec![("quotient − 1", forge(-1)), ("quotient + 1", forge(1))],
    );
}

// The tests below claim an execution whose exact result at one cell lies outside `i32`: the
// operands are set so that the op's integer identity gives `i32::MAX + 1` or less than
// `i32::MIN`, and the node's output (an `i32` tensor, like every witness column) holds that
// result reduced mod 2^32, with the true remainder. No execution exists, so each must be
// rejected. (An output cell that is itself out of range is the digit tamper's case, in
// [`witness_tamper_sweep`].)

/// One claimed overflow: operand cells to set, as `(node, element, value)`, and the op's
/// exact accumulation at output cell 0 with its divisor (1 for an op that does not rescale).
struct Overflow {
    label: &'static str,
    operands: Vec<(usize, usize, i32)>,
    acc: i128,
    divisor: i128,
}

/// `node`'s output cell 0 is the exact result of `case` wrapped to `i32`, forwarded by
/// `consumer` (an identity).
fn forge_overflow(node: usize, consumer: usize, case: &Overflow) -> Forgery {
    let exact = case.acc.div_euclid(case.divisor);
    let remainder = case.acc.rem_euclid(case.divisor) as i32;
    assert!(
        exact > i128::from(i32::MAX) || exact < i128::from(i32::MIN),
        "{}: {exact} is inside i32",
        case.label
    );
    let wrapped = exact as i32;
    let operands = case.operands.clone();
    Box::new(move |trace| {
        for &(operand, elem, value) in &operands {
            trace.node_outputs[operand].data_mut()[elem] = value;
        }
        trace.node_outputs[node].data_mut()[0] = wrapped;
        trace.node_outputs[consumer].data_mut()[0] = wrapped;
        if let Some(atlas_trace::trace::NodeAux::Fused(f)) = trace.aux.get_mut(&NodeId(node as u32))
        {
            f.remainder.data_mut()[0] = remainder;
        }
    })
}

/// Build a model with `build` (returning the checked node), give it an identity consumer as
/// its output, and assert every case of `cases` is rejected.
fn assert_overflow_rejected(
    what: &str,
    build: impl FnOnce(&mut ModelBuilder) -> usize,
    inputs: Vec<Tensor<i32>>,
    cases: Vec<Overflow>,
) {
    let mut b = ModelBuilder::new();
    let node = build(&mut b);
    let consumer = b.identity(node);
    b.mark_output(consumer);
    let forgeries = cases
        .iter()
        .map(|case| (case.label, forge_overflow(node, consumer, case)))
        .collect();
    assert_bound_holds(what, &b.build(), &inputs, forgeries);
}

const I32_MAX: i128 = i32::MAX as i128;
const I32_MIN: i128 = i32::MIN as i128;
const RESCALE: i128 = 1 << MODEL_SCALE;

fn ones(dims: &[usize]) -> Tensor<i32> {
    Tensor::from_slice(&vec![1; dims.iter().product()], dims).unwrap()
}

#[test]
#[serial(soundness_probe)]
fn add_and_sub_overflow_have_no_proof() {
    for (what, sign) in [("Add", 1i128), ("Sub", -1i128)] {
        let case = |label, x: i32, y: i32| Overflow {
            label,
            operands: vec![(0, 0, x), (1, 0, y)],
            acc: i128::from(x) + sign * i128::from(y),
            divisor: 1,
        };
        let step = sign as i32;
        assert_overflow_rejected(
            what,
            |b| {
                let x = b.input(vec![4]);
                let y = b.input(vec![4]);
                if sign == 1 {
                    b.add(x, y)
                } else {
                    b.sub(x, y)
                }
            },
            vec![ones(&[4]), ones(&[4])],
            vec![
                case("i32::MAX + 1", i32::MAX, step),
                case("i32::MIN − 1", i32::MIN, -step),
            ],
        );
    }
}

#[test]
#[serial(soundness_probe)]
fn sum_overflow_has_no_proof() {
    let case = |label, x: i32, y: i32| Overflow {
        label,
        operands: vec![(0, 0, x), (0, 1, y)],
        // The other two cells of the row are the honest ones.
        acc: i128::from(x) + i128::from(y) + 2,
        divisor: 1,
    };
    assert_overflow_rejected(
        "Sum",
        |b| {
            let x = b.input(vec![2, 4]);
            b.sum(x, vec![1], vec![2])
        },
        vec![ones(&[2, 4])],
        vec![
            case("above i32::MAX", i32::MAX, 1),
            case("below i32::MIN", i32::MIN, -4),
        ],
    );
}

#[test]
#[serial(soundness_probe)]
fn fused_product_overflow_has_no_proof() {
    // (−715827883)·(3·2^14) = (i32::MIN − 1)·2^14.
    let mul = |label, x: i32, y: i32| Overflow {
        label,
        operands: vec![(0, 0, x), (1, 0, y)],
        acc: i128::from(x) * i128::from(y),
        divisor: RESCALE,
    };
    assert_eq!(mul("", 1 << 30, 1 << 15).acc, (I32_MAX + 1) * RESCALE);
    assert_eq!(mul("", -715_827_883, 3 << 14).acc, (I32_MIN - 1) * RESCALE);
    assert_overflow_rejected(
        "Mul",
        |b| {
            let x = b.input(vec![4]);
            let y = b.input(vec![4]);
            b.mul(x, y)
        },
        vec![ones(&[4]), ones(&[4])],
        vec![
            mul("i32::MAX + 1", 1 << 30, 1 << 15),
            mul("i32::MIN − 1", -715_827_883, 3 << 14),
        ],
    );

    assert_overflow_rejected(
        "Square",
        |b| {
            let x = b.input(vec![4]);
            b.square(x)
        },
        vec![ones(&[4])],
        vec![Overflow {
            label: "above i32::MAX",
            operands: vec![(0, 0, 1 << 23)],
            acc: 1 << 46,
            divisor: RESCALE,
        }],
    );

    let cube = |label, x: i32| Overflow {
        label,
        operands: vec![(0, 0, x)],
        acc: i128::from(x).pow(3),
        divisor: RESCALE * RESCALE,
    };
    assert_overflow_rejected(
        "Cube",
        |b| {
            let x = b.input(vec![4]);
            b.cube(x, MODEL_SCALE as u8)
        },
        vec![ones(&[4])],
        vec![
            cube("above i32::MAX", 1 << 20),
            cube("below i32::MIN", -(1 << 20)),
        ],
    );
}

#[test]
#[serial(soundness_probe)]
fn reduction_overflow_has_no_proof() {
    // Row 0 of the input is four cells of 2^24: Σ x² = 2^50 over a divisor of 4·2^14.
    assert_overflow_rejected(
        "MeanOfSquares",
        |b| {
            let x = b.input(vec![2, 4]);
            b.mean_of_squares(x, vec![1], vec![2, 1])
        },
        vec![ones(&[2, 4])],
        vec![Overflow {
            label: "above i32::MAX",
            operands: (0..4).map(|j| (0, j, 1 << 24)).collect(),
            acc: 1 << 50,
            divisor: 4 * RESCALE,
        }],
    );

    // `out[0][0] = Σ_k x[0][k]·w[k][0]`, with `w[0][0] = 2^15` and the rest of the column zero.
    let mut weights = vec![1i32; 16];
    for k in 0..4 {
        weights[4 * k] = 0;
    }
    weights[0] = 1 << 15;
    let einsum = |label, x: i32| Overflow {
        label,
        operands: vec![(0, 0, x)],
        acc: i128::from(x) << 15,
        divisor: RESCALE,
    };
    assert_overflow_rejected(
        "Einsum",
        |b| {
            let x = b.input(vec![4, 4]);
            let w = b.constant(Tensor::from_slice(&weights, &[4, 4]).unwrap());
            b.einsum("mk,kn->mn", vec![x, w], vec![4, 4])
        },
        vec![ones(&[4, 4])],
        vec![
            einsum("i32::MAX + 1", 1 << 30),
            einsum("below i32::MIN", -(1 << 30) - 1),
        ],
    );
}

/// A bound that is not positive has no proof: a negative divisor (floor division puts its
/// remainder in `(B, 0]`) and an `Rsqrt` of a negative input (output 0).
#[test]
#[serial(soundness_probe)]
fn non_positive_bounds_have_no_proof() {
    let _probe = ProbeGuard::enter();

    let mut b = ModelBuilder::new();
    let x = b.input(vec![8]);
    let divisor = b.constant(Tensor::from_slice(&[-7; 8], &[8]).unwrap());
    let out = b.div(x, divisor);
    b.mark_output(out);
    let inputs = vec![Tensor::from_slice(&[100, -100, 3, 46, 1 << 20, -5, 13, 6], &[8]).unwrap()];
    let outcome = bound_outcome(&b.build(), &inputs, &mut |_| {});
    assert!(
        matches!(outcome, Outcome::Rejected(_)),
        "Div by −7: {outcome:?}"
    );

    let mut b = ModelBuilder::new();
    let x = b.input(vec![8]);
    let out = b.rsqrt(x);
    b.mark_output(out);
    let inputs =
        vec![Tensor::from_slice(&[1000, 3000, -5, 12_345, 50_000, 7, 9, 11], &[8]).unwrap()];
    let outcome = bound_outcome(&b.build(), &inputs, &mut |_| {});
    assert!(
        matches!(outcome, Outcome::Rejected(_)),
        "Rsqrt of −5: {outcome:?}"
    );
}

/// Rewrite row `row` of softmax node `node` as an honest-looking witness at shift `c`: the
/// addresses, the live lanes' outputs and the two row gaps all follow from `c`.
fn forge_softmax_shift(node: usize, row: usize, c: i32) -> Forgery {
    use atlas_trace::semantics::softmax::softmax_tables;
    Box::new(move |trace| {
        let tables = softmax_tables(
            atlas_trace::Scale::new(MODEL_SCALE as u8).unwrap(),
            atlas_trace::SemVersion::V2,
        );
        let sum_prev = tables.sum_prev();
        let Some(atlas_trace::trace::NodeAux::Softmax(w)) = trace.aux.get_mut(&NodeId(node as u32))
        else {
            panic!("node {node} has no softmax witness")
        };
        let (n, live) = (w.last_dim, w.active_len);
        let (mut sum, mut prev) = (0i64, 0i64);
        let mut outputs = Vec::new();
        for j in 0..n {
            let cell = row * n + j;
            // The logit, from the honest address.
            let x = w.shift[row] - w.address[cell];
            w.address[cell] = c - x;
            if j < live {
                let a = (w.address[cell] as i64).clamp(0, tables.top()) as usize;
                outputs.push((cell, tables.out[a]));
                sum += tables.sum[a] as i64;
                prev += sum_prev[a] as i64;
            }
        }
        w.shift[row] = c;
        w.sum_gap[row] = tables.sum_scale - sum;
        w.prev_sum_gap[row] = prev - tables.sum_scale - 1;
        for (cell, y) in outputs {
            trace.node_outputs[node].data_mut()[cell] = y;
        }
    })
}

/// The softmax shift is pinned: a witness rebuilt around any other shift is rejected, except
/// below a lane that dominates its row, where it proves the same output.
#[test]
#[serial(soundness_probe)]
fn softmax_shift_is_pinned() {
    let model = |rows: usize| {
        let mut b = ModelBuilder::new();
        let x = b.input(vec![rows, 4]);
        let out = b.softmax_last_axis(x);
        b.mark_output(out);
        (b.build(), out)
    };
    // Row 0 has two comparable lanes; row 1 has one lane far above the rest.
    let inputs =
        vec![Tensor::from_slice(&[1000, 990, -3000, 250, 400_000, 0, -5, 7], &[2, 4]).unwrap()];
    let (m, out) = model(2);
    let honest = m.trace(&inputs).expect("honest execution");
    let shift = |row: usize| match honest.aux.get(&NodeId(out as u32)) {
        Some(atlas_trace::trace::NodeAux::Softmax(w)) => w.shift[row],
        _ => panic!("no softmax witness"),
    };
    let (c0, c1) = (shift(0), shift(1));
    assert!(c0 > 1001, "row 0's shift is above its maximum");
    assert_eq!(c1, 400_000, "row 1's shift is its dominant logit");
    assert_bound_holds(
        "SoftmaxLastAxis",
        &m,
        &inputs,
        vec![
            ("shift + 1", forge_softmax_shift(out, 0, c0 + 1)),
            ("shift − 1", forge_softmax_shift(out, 0, c0 - 1)),
            // Below lane 0's logit with lane 1 still reading a nonzero value.
            ("shift below a lane", forge_softmax_shift(out, 0, 995)),
            ("dominant shift + 1", forge_softmax_shift(out, 1, c1 + 1)),
        ],
    );
    // Below the dominant lane's logit and far above every other: the clamp reads address zero
    // for the dominant lane, the others read zero, and the output is the honest one.
    let _probe = ProbeGuard::enter();
    let outcome = bound_outcome(&m, &inputs, &mut *forge_softmax_shift(out, 1, c1 - 10));
    assert!(matches!(outcome, Outcome::Accepted), "{outcome:?}");
    let mut forged = honest.clone();
    forge_softmax_shift(out, 1, c1 - 10)(&mut forged);
    assert_eq!(forged.node_outputs[out], honest.node_outputs[out]);
}
