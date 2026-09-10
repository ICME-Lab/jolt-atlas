//! `tab:committed-polys` + `tab:clamp-contributions`: GPT-2, deterministic counter dump from
//! preprocessing/witness generation — no proving, no repetitions. See
//! `../../paper-jolt-atlas2/BENCHMARKS.md` benchmark 3 / `PQCrypto.tex` section 6 for the
//! operator taxonomy this mirrors.
//!
//! Table A: one row per committed-polynomial source category. Everything proven via the post-IOP
//! batched `deferred_lookups::prove_all` (see [`is_deferred`]) — pooled buckets and per-node
//! polys alike — lands in one "Deferred" row instead of its owning node's category. Table B
//! breaks that row down exactly by contributing operator group (checked by a runtime assertion).
//!
//! `cargo run --release -p jolt-atlas-core --example bench_census` (requires `MODEL_SCALE ==
//! 12`). Writes `bench/results/tab_committed_polys_gpt2.json`.

mod harness;
mod models;
mod report;

use atlas_onnx_tracer::{
    model::{trace::Trace, Model, RunArgs},
    ops::Operator,
};
use common::CommittedPoly;
use jolt_atlas_core::onnx_proof::{global_clamp, Blake2bTranscript, DoryScheme, Fr, ONNXProof};
use joltworks::poly::multilinear_polynomial::MultilinearPolynomial;
use report::{upsert_row, BenchModel, CommittedPolyRow, Meta, Row};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

const SEQ_LEN: usize = 16;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct CensusTableMeta {
    table: String,
    model: BenchModel,
    seq_len: usize,
    scale: usize,
}

impl Meta for CensusTableMeta {}

/// `clamp_contributions` (Table B) sums exactly to `committed_polys`'s "Deferred" row —
/// pooled-bucket polys are split by real per-node entry ranges, not estimated.
#[derive(Serialize, Deserialize)]
struct CensusRow {
    total_committed_polys: usize,
    total_one_hot_entries: usize,
    committed_polys: Vec<CommittedPolyRow>,
    clamp_contributions: Vec<CommittedPolyRow>,
}

impl Row for CensusRow {
    fn name(&self) -> String {
        "default".to_string()
    }
}

/// Table A's source categories, in report order (`Category::iter()` walks this order).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, EnumIter)]
enum Category {
    /// Everything [`is_deferred`] — Table B breaks this down by operator group.
    Deferred,
    ContractionArithmetic,
    LookupRelu,
    LookupTeleported,
    Composite,
    Indexing,
    FreeBoolean,
}

impl Category {
    fn label(self) -> &'static str {
        match self {
            Category::Deferred => "Deferred sumchecks",
            Category::ContractionArithmetic => "Contraction/arithmetic",
            Category::LookupRelu => "Prefix-Suffix Lookup",
            Category::LookupTeleported => "Teleported",
            Category::Composite => "Composite operators",
            Category::Indexing => "Indexing operators",
            Category::FreeBoolean => "Free operators",
        }
    }

    fn operators(self) -> &'static [&'static str] {
        match self {
            Category::Deferred => &[],
            Category::ContractionArithmetic => &[
                "Add",
                "Sub",
                "Mul",
                "Square",
                "Cube",
                "Einsum",
                "Sum",
                "MeanOfSquares",
            ],
            Category::LookupRelu => &["ReLU"],
            Category::LookupTeleported => &["Erf", "Sigmoid", "Tanh", "Cos", "Sin"],
            Category::Composite => &["Rsqrt", "SoftmaxLastAxis", "Div"],
            Category::Indexing => &[
                "Gather",
                "MoveAxis",
                "Reshape",
                "Slice",
                "Concat",
                "Broadcast",
            ],
            Category::FreeBoolean => &[
                "Neg", "Identity", "IsNan", "Input", "Constant", "And", "Iff",
            ],
        }
    }
}

/// Table B's operator groups. [`Self::for_operator`] panics on a non-deferred-eligible operator.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OperatorGroup {
    AddSub,
    MulSquareCube,
    Einsum,
    SumMeanOfSquares,
    Activation,
    Composite,
}

impl OperatorGroup {
    fn label(self) -> &'static str {
        match self {
            OperatorGroup::AddSub => "Add/Sub",
            OperatorGroup::MulSquareCube => "Mul/Square/Cube",
            OperatorGroup::Einsum => "Einsum",
            OperatorGroup::SumMeanOfSquares => "Sum/MeanOfSquares",
            OperatorGroup::Activation => "Activation (Erf/Sigmoid/Tanh)",
            OperatorGroup::Composite => "Composite operators",
        }
    }

    fn operators(self) -> &'static [&'static str] {
        match self {
            OperatorGroup::AddSub => &["Add", "Sub"],
            OperatorGroup::MulSquareCube => &["Mul", "Square", "Cube"],
            OperatorGroup::Einsum => &["Einsum"],
            OperatorGroup::SumMeanOfSquares => &["Sum", "MeanOfSquares"],
            OperatorGroup::Activation => &["Erf", "Sigmoid", "Tanh"],
            OperatorGroup::Composite => &["Rsqrt", "SoftmaxLastAxis"],
        }
    }

    fn for_operator(op: &Operator) -> Self {
        match op {
            Operator::Add(_) | Operator::Sub(_) => OperatorGroup::AddSub,
            Operator::Mul(_) | Operator::Square(_) | Operator::Cube(_) => {
                OperatorGroup::MulSquareCube
            }
            Operator::Einsum(_) => OperatorGroup::Einsum,
            Operator::Sum(_) | Operator::MeanOfSquares(_) => OperatorGroup::SumMeanOfSquares,
            Operator::Erf(_) | Operator::Sigmoid(_) | Operator::Tanh(_) => {
                OperatorGroup::Activation
            }
            Operator::Rsqrt(_) | Operator::SoftmaxLastAxis(_) => OperatorGroup::Composite,
            other => {
                panic!("operator {other:?} owns no deferred poly, should never be routed here")
            }
        }
    }
}

/// Exhaustive over `Operator`'s variants — a new operator forces a compile error here, not a
/// silent default.
fn category_for_operator(op: &Operator) -> Category {
    use Operator::*;
    match op {
        Add(_) | Sub(_) | Mul(_) | Square(_) | Cube(_) | Einsum(_) | Sum(_) | MeanOfSquares(_) => {
            Category::ContractionArithmetic
        }
        ReLU(_) => Category::LookupRelu,
        Erf(_) | Sigmoid(_) | Tanh(_) | Cos(_) | Sin(_) => Category::LookupTeleported,
        Rsqrt(_) | SoftmaxLastAxis(_) | Div(_) | ScalarConstDiv(_) => Category::Composite,
        GatherSmall(_) | GatherLarge(_) | MoveAxis(_) | Reshape(_) | Slice(_) | Concat(_)
        | Broadcast(_) => Category::Indexing,
        Neg(_) | Identity(_) | IsNan(_) | Input(_) | Constant(_) | And(_) | Iff(_) => {
            Category::FreeBoolean
        }
        // `Clamp` is a test-only operator, never present in a real model; falls back to
        // `FreeBoolean` (harmless — always zero-count for GPT-2) rather than a dedicated arm.
        Clamp(_) => Category::FreeBoolean,
    }
}

/// Node index owning a `CommittedPoly`, or `None` for the pooled `Global*` (bucket-indexed)
/// variants.
fn node_idx_of(poly: &CommittedPoly) -> Option<usize> {
    use CommittedPoly::*;
    match *poly {
        NodeOutputRaD(n, _)
        | CosRaD(n, _)
        | SinRaD(n, _)
        | TrigDownscaleRaD(n, _)
        | DivRangeCheckRaD(n, _)
        | SqrtDivRangeCheckRaD(n, _)
        | MeanOfSquaresRangeCheckRaD(n, _)
        | SqrtRangeCheckRaD(n, _)
        | TeleportRangeCheckRaD(n, _)
        | DivNodeQuotient(n)
        | ScalarConstDivNodeRemainder(n)
        | RsqrtQuotient(n)
        | TeleportNodeQuotient(n)
        | GatherRa(n)
        | GatherRaD(n, _)
        | SoftmaxRemainderRaD(n, _)
        | SoftmaxExpRemainderRaD(n, _)
        | SoftmaxZHiRaD(n, _)
        | SoftmaxZLoRaD(n, _)
        | ClampRaD(n, _)
        | RescaleRemainderRaD(n, _)
        | SymmetricClampRaD(n, _)
        | ActivationClampRaD(n, _)
        | ActivationSmallRaD(n, _)
        | SoftmaxClampRaD(n, _) => Some(n),
        GlobalClampRaD(..)
        | GlobalRemainderRaD(..)
        | GlobalClampOutRaD(..)
        | GlobalClampSlackRaD(..) => None,
    }
}

/// Whether a `CommittedPoly` is proven via the post-IOP batched `deferred_lookups::prove_all`
/// rather than inline during the per-node IOP loop — verified against every
/// `prover.defer(...)`/`prover.deferred.push(...)` call site, not guessed.
fn is_deferred(poly: &CommittedPoly) -> bool {
    use CommittedPoly::*;
    match *poly {
        GlobalClampRaD(..)
        | GlobalRemainderRaD(..)
        | GlobalClampOutRaD(..)
        | GlobalClampSlackRaD(..)
        | ActivationClampRaD(..)
        | ActivationSmallRaD(..)
        | MeanOfSquaresRangeCheckRaD(..)
        | SqrtRangeCheckRaD(..)
        | SqrtDivRangeCheckRaD(..)
        | SoftmaxRemainderRaD(..)
        | SoftmaxExpRemainderRaD(..)
        | SoftmaxZHiRaD(..)
        | SoftmaxZLoRaD(..)
        | SoftmaxClampRaD(..) => true,
        NodeOutputRaD(..)
        | CosRaD(..)
        | SinRaD(..)
        | TrigDownscaleRaD(..)
        | DivRangeCheckRaD(..)
        | TeleportRangeCheckRaD(..)
        | DivNodeQuotient(..)
        | ScalarConstDivNodeRemainder(..)
        | RsqrtQuotient(..)
        | TeleportNodeQuotient(..)
        | GatherRa(..)
        | GatherRaD(..)
        | ClampRaD(..)
        | RescaleRemainderRaD(..)
        | SymmetricClampRaD(..) => false,
    }
}

/// A dense poly's padded size, or a one-hot poly's actual nonzero entry count (the sparse
/// commit cost scales with this, not the padded shape).
enum PolySize {
    Dense,
    OneHot(usize),
}

fn poly_size<F: joltworks::field::JoltField>(poly: &MultilinearPolynomial<F>) -> PolySize {
    match poly {
        MultilinearPolynomial::OneHot(one_hot) => PolySize::OneHot(
            one_hot
                .nonzero_indices
                .iter()
                .filter(|x| x.is_some())
                .count(),
        ),
        _ => PolySize::Dense,
    }
}

/// Per-category tally, split by cost dimension — see [`PolySize`].
#[derive(Default, Clone, Copy)]
struct PolyTally {
    dense_polys: usize,
    one_hot_polys: usize,
    one_hot_entries: usize,
}

fn main() {
    assert_eq!(
        common::consts::MODEL_SCALE,
        12,
        "GPT-2 exports at scale 12; currently compiled with MODEL_SCALE = {}. Edit \
         common/src/consts/general.rs and recompile.",
        common::consts::MODEL_SCALE
    );

    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", SEQ_LEN),
        ("past_sequence_length", 0),
    ]);
    let model = Model::load("atlas-onnx-tracer/models/gpt2/network.onnx", &run_args);
    println!(
        "max_num_vars={} (derived, not part of table meta)",
        model.max_num_vars()
    );
    let scale = run_args.scale;

    let input_ids = models::random_input_ids(SEQ_LEN, models::GPT2_VOCAB_SIZE, 42);
    let position_ids: Vec<i32> = (0..SEQ_LEN as i32).collect();
    let attention_mask = vec![1i32 << scale; SEQ_LEN];
    let inputs = [
        atlas_onnx_tracer::tensor::Tensor::new(Some(&input_ids), &[1, SEQ_LEN]).unwrap(),
        atlas_onnx_tracer::tensor::Tensor::new(Some(&position_ids), &[1, SEQ_LEN]).unwrap(),
        atlas_onnx_tracer::tensor::Tensor::new(Some(&attention_mask), &[1, SEQ_LEN]).unwrap(),
    ];
    let trace: Trace = model.trace(&inputs);

    // PCS choice is irrelevant here; Dory picked arbitrarily to instantiate the generic.
    let poly_map = ONNXProof::<Fr, Blake2bTranscript, DoryScheme>::polynomial_map(&model, &trace);

    let mut by_category: BTreeMap<Category, PolyTally> = BTreeMap::new();
    for (poly, mle) in &poly_map {
        let category = if is_deferred(poly) {
            Category::Deferred
        } else {
            let node_idx = node_idx_of(poly)
                .expect("a non-deferred poly is always node-indexed (only Global* has no node)");
            category_for_operator(&model.graph.nodes[&node_idx].operator)
        };
        let tally = by_category.entry(category).or_default();
        match poly_size(mle) {
            PolySize::Dense => tally.dense_polys += 1,
            PolySize::OneHot(entries) => {
                tally.one_hot_polys += 1;
                tally.one_hot_entries += entries;
            }
        }
    }

    // `Deferred` gets no nodes here — it's a proving-time property of polys, not of operators.
    let mut node_counts: BTreeMap<Category, usize> = BTreeMap::new();
    for node in model.graph.nodes.values() {
        *node_counts
            .entry(category_for_operator(&node.operator))
            .or_default() += 1;
    }

    let total_committed_polys = poly_map.len();
    let total_one_hot_entries: usize = by_category.values().map(|t| t.one_hot_entries).sum();

    let committed_polys: Vec<CommittedPolyRow> = Category::iter()
        .map(|cat| {
            let t = by_category.get(&cat).copied().unwrap_or_default();
            CommittedPolyRow {
                category: cat.label().to_string(),
                operators: cat.operators().iter().map(|s| s.to_string()).collect(),
                nodes: node_counts.get(&cat).copied().unwrap_or(0),
                dense_polys: t.dense_polys,
                one_hot_polys: t.one_hot_polys,
                one_hot_entries: t.one_hot_entries,
            }
        })
        .collect();

    println!("=== tab:committed-polys (GPT-2, seq_len={SEQ_LEN}) ===");
    for row in &committed_polys {
        println!(
            "{:<52} nodes={:<5} dense={:<5} one_hot={:<5}/{:<10}",
            row.category, row.nodes, row.dense_polys, row.one_hot_polys, row.one_hot_entries
        );
    }
    println!(
        "TOTAL: {total_committed_polys} committed polys, {total_one_hot_entries} one-hot entries"
    );

    // --- Table B: per-operator contribution to the pooled clamp/remainder buckets ---
    // Each pooled `Global*` poly's member node occupies a disjoint `[offset, offset+2^log_t)`
    // segment of the bucket's shared cycle space, so slicing `nonzero_indices` by that range
    // gives its real (not estimated) contribution.
    let clamp_buckets = global_clamp::clamp_buckets(&model);
    let remainder_buckets = global_clamp::remainder_buckets(&model);
    let clamp_buckets_by_idx: BTreeMap<usize, &global_clamp::ClampBucket> =
        clamp_buckets.iter().map(|b| (b.idx, b)).collect();
    let remainder_buckets_by_idx: BTreeMap<usize, &global_clamp::ClampBucket> =
        remainder_buckets.iter().map(|b| (b.idx, b)).collect();

    let mut instance_nodes: BTreeMap<OperatorGroup, BTreeSet<usize>> = BTreeMap::new();
    let mut group_entries: BTreeMap<OperatorGroup, usize> = BTreeMap::new();
    // A pooled poly shared across a mixed bucket touches more than one group, counted once each
    // (poly identity isn't fractionally splittable the way its entries are).
    let mut group_polys: BTreeMap<OperatorGroup, BTreeSet<CommittedPoly>> = BTreeMap::new();
    // Dummy pooled-bucket tail cycles are still committed (real cost, owned by no operator) —
    // get their own row rather than being silently dropped from the total.
    let mut packing_overhead_entries = 0usize;

    for (poly, mle) in &poly_map {
        if !is_deferred(poly) {
            continue;
        }
        match node_idx_of(poly) {
            Some(node_idx) => {
                let group = OperatorGroup::for_operator(&model.graph.nodes[&node_idx].operator);
                instance_nodes.entry(group).or_default().insert(node_idx);
                group_polys.entry(group).or_default().insert(*poly);
                match poly_size(mle) {
                    PolySize::OneHot(e) => *group_entries.entry(group).or_default() += e,
                    PolySize::Dense => panic!("deferred poly {poly:?} is unexpectedly dense"),
                }
            }
            None => {
                let MultilinearPolynomial::OneHot(one_hot) = mle else {
                    panic!("pooled Global* poly {poly:?} is unexpectedly not one-hot");
                };
                let is_remainder = matches!(poly, CommittedPoly::GlobalRemainderRaD(..));
                let bucket_idx = match *poly {
                    CommittedPoly::GlobalClampRaD(idx, _)
                    | CommittedPoly::GlobalRemainderRaD(idx, _)
                    | CommittedPoly::GlobalClampOutRaD(idx, _)
                    | CommittedPoly::GlobalClampSlackRaD(idx, _) => idx,
                    _ => unreachable!("node_idx_of already filtered to only Global* variants"),
                };
                let bucket = if is_remainder {
                    remainder_buckets_by_idx[&bucket_idx]
                } else {
                    clamp_buckets_by_idx[&bucket_idx]
                };
                let mut covered_entries = 0usize;
                for node in &bucket.nodes {
                    let group = OperatorGroup::for_operator(&model.graph.nodes[&node.idx].operator);
                    instance_nodes.entry(group).or_default().insert(node.idx);
                    group_polys.entry(group).or_default().insert(*poly);
                    let size = 1usize << node.log_t;
                    let node_entries = one_hot.nonzero_indices[node.offset..node.offset + size]
                        .iter()
                        .filter(|x| x.is_some())
                        .count();
                    covered_entries += node_entries;
                    *group_entries.entry(group).or_default() += node_entries;
                }
                let total_entries = one_hot
                    .nonzero_indices
                    .iter()
                    .filter(|x| x.is_some())
                    .count();
                packing_overhead_entries += total_entries - covered_entries;
            }
        }
    }

    let mut clamp_contributions: Vec<CommittedPolyRow> = group_entries
        .iter()
        .map(|(&group, &one_hot_entries)| CommittedPolyRow {
            category: group.label().to_string(),
            operators: group.operators().iter().map(|s| s.to_string()).collect(),
            nodes: instance_nodes[&group].len(),
            dense_polys: 0,
            one_hot_polys: group_polys[&group].len(),
            one_hot_entries,
        })
        .collect();
    clamp_contributions.push(CommittedPolyRow {
        category: "(packing overhead: dummy pooled-bucket cycles)".to_string(),
        operators: Vec::new(),
        nodes: 0,
        dense_polys: 0,
        one_hot_polys: 0,
        one_hot_entries: packing_overhead_entries,
    });

    let deferred_entries_total = by_category
        .get(&Category::Deferred)
        .map(|t| t.one_hot_entries)
        .unwrap_or(0);
    let table_b_total: usize = clamp_contributions.iter().map(|r| r.one_hot_entries).sum();
    assert_eq!(
        table_b_total, deferred_entries_total,
        "tab:clamp-contributions must sum exactly to tab:committed-polys's Deferred row"
    );

    println!("\n=== tab:clamp-contributions ===");
    for row in &clamp_contributions {
        println!(
            "{:<30} nodes={:<5} one_hot={:<5}/{}",
            row.category, row.nodes, row.one_hot_polys, row.one_hot_entries
        );
    }

    let census = CensusRow {
        total_committed_polys,
        total_one_hot_entries,
        committed_polys,
        clamp_contributions,
    };

    let out_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/bench/results"
    ))
    .join("tab_committed_polys_gpt2.json");
    let meta = CensusTableMeta {
        table: "tab:committed-polys".to_string(),
        model: BenchModel::Gpt2,
        seq_len: SEQ_LEN,
        scale: scale as usize,
    };
    upsert_row(&out_path, meta, census);
}
