//! Softmax over the last axis in log-partition form (ONNX Softmax).
//!
//! Semantics ([`atlas_trace::semantics::softmax`]): per row, the shift `c` is the smallest
//! integer at or above the live lanes' maximum whose sum-table row sum does not exceed `S'`,
//! and a live lane's output is the output table read at `min(c − x, top)`.
//!
//! The proof commits `c` per row and two one-hot addresses per entry: the signed 32-bit clamp
//! address `u = c − x` and the table address `ũ = Clamp[u]`. With `v`, `t`, `t⁻` the reads of
//! the output table, the sum table and the sum table shifted by one address at `ũ`, and `a` the
//! public selector of the live lanes:
//!
//! ```text
//! out[k,j]  = a(j)·v[k,j]
//! g[k]      = S' − Σ_j a(j)·t[k,j]            ≥ 0      (the row sum does not exceed S')
//! g⁻[k]     = Σ_j a(j)·t⁻[k,j] − S' − 1       ≥ 0      (no smaller shift is admissible)
//! u[k,j]    = c[k] − x[k,j]
//! ```
//!
//! The first three are one sumcheck ([`SumsRelation`]) that leaves `v`, `t`, `t⁻` at one point;
//! one table read through the table address proves them there, the clamp lookup ties `ũ` to
//! `u`, and the address link ties `u` to the operand. `g` and `g⁻` are range-checked in the
//! remainder buckets.
//!
//! **S ⊆ R.** The one-hot checks make `u` an integer in `[−2^31, 2^31)` and `ũ` an integer in
//! `[0, 2^D)` per entry with `ũ = min(max(u, 0), top)`, and every read a table entry. The
//! address link at a random point gives `u = c − x` on every entry, over the integers since
//! both sides are below `2^33` in magnitude. The sums relation at a random point gives `g` and
//! `g⁻` as above for every row, with all sums far below the modulus, and their range claims
//! give `Σ t ≤ S'` and `Σ t⁻ > S'` over the live lanes. If every live lane has `u ≥ 0`, then
//! `c ≥ x` on the live lanes, `Σ t = f(c)`, and `Σ t⁻` is `f(c − 1)` or exceeds `S'` by the
//! shifted table's first entry when some `u = 0`; `f` is nonincreasing, so `c` is the defined
//! shift and `v` the defined output. If some live lane has `u < 0`, its table address is 0 and
//! it contributes `S'` to the row sum, so every other live lane reads zero in the sum table,
//! hence in the output table: that lane is the row's strict maximum, the defined shift is its
//! logit, and the defined output is `S` there and zero elsewhere, which is what was read.
//!
//! The clamp table's bound is a const generic at [`MODEL_SCALE`], so only that scale is
//! supported.
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::relation::{
    low_to_high_point, not_derived, AtlasRelation, EngineRelation, Produces, Relation,
    RelationChallenge, RelationDerived, RelationExpr, RelationId, RelationOpening, RelationSetup,
    RelationVerifier, SetupRelation,
};
use atlas_sumcheck::relation::DynRelationVerifier;

use crate::{
    onnx_proof::{
        deferred_lookups::DeferredBatch,
        op_lookups::OpLookupProvider,
        ops::softmax_last_axis::significance_clamp::SoftmaxSignificanceClampOperands,
        plan::{OpOpen, OpPlan, OpRelation, OpTie, Plan, Step},
        ProofType,
    },
    utils::opening_access::{AccOpeningAccessor, OpeningIdBuilder, Target},
};
use atlas_lookup_tables::clamp::SoftmaxClampTable;
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator},
    shout::{RaOneHotEncoding, ReadRafProvider},
};

use atlas_trace::{
    op::SoftmaxLastAxis,
    semantics::softmax::{softmax_tables, SoftmaxTables},
    trace::ModelExecutionIO,
    Model, Op, Scale, SemVersion,
};

use crate::node_view::ComputationNode;
use atlas_algebra::field::Ring;
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq::EqPolynomial,
        point::{OpeningPoint, BIG_ENDIAN},
    },
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{challenge, derived, opening, SymbolicSumcheck};
use atlas_sumcheck::shout;
use atlas_transcript::Transcript;
use common::{
    consts::{MODEL_SCALE, SOFTMAX_CLAMP_BOUND, XLEN},
    CommittedPoly, VirtualPoly,
};
use std::sync::OnceLock;

/// The clamp lookup of the address `u = c − x` (`SoftmaxSignificanceClampOperands`).
pub mod significance_clamp;

/// The softmax tables at [`MODEL_SCALE`] with the shifted sum table, generated once.
pub struct ReadTables {
    /// The tables as the trace semantics generate them.
    pub tables: SoftmaxTables,
    /// [`SoftmaxTables::sum_prev`].
    pub sum_prev: Vec<i32>,
}

/// The softmax tables at [`MODEL_SCALE`]. `Err` if their address width is not the clamp
/// table's.
pub fn read_tables() -> Result<&'static ReadTables, ProofVerifyError> {
    static TABLES: OnceLock<Option<ReadTables>> = OnceLock::new();
    TABLES
        .get_or_init(|| {
            let tables = softmax_tables(Scale::new(MODEL_SCALE as u8).ok()?, SemVersion::V2);
            (tables.address_bits as usize == SOFTMAX_CLAMP_BOUND).then(|| ReadTables {
                sum_prev: tables.sum_prev(),
                tables,
            })
        })
        .as_ref()
        .ok_or_else(|| {
            ProofVerifyError::Malformed(
                "softmax: the tables' address width is not the clamp table's".to_string(),
            )
        })
}

impl OpLayout for SoftmaxLastAxis {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        let idx = node.idx;
        let mut polys = vec![CommittedPoly::SoftmaxShift(idx)];
        for (log_k, ctor) in [
            (
                XLEN,
                CommittedPoly::SoftmaxClampRaD as fn(usize, usize) -> _,
            ),
            (SOFTMAX_CLAMP_BOUND, CommittedPoly::SoftmaxAddrRaD),
        ] {
            let d =
                OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k).instruction_d;
            polys.extend((0..d).map(|i| ctor(idx, i)));
        }
        Ok(polys)
    }
}

/// The plan: the evaluation reduction; the row gaps opened at the reduced point's row
/// coordinates; the sums relation, which leaves the three reads at one point `r1`; the shift at
/// `r1`'s row coordinates and the table address at `r1`; the table read and the clamp lookup,
/// queued for the post-loop lookup batch; the address link in the clear; the two addresses'
/// one-hot checks, queued for the one-hot batch.
impl OpPlan for SoftmaxLastAxis {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        // The clamp table's bound is `SOFTMAX_CLAMP_BOUND` at `MODEL_SCALE`.
        if node.scale as usize != MODEL_SCALE {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "SoftmaxLastAxis verifying is only calibrated for MODEL_SCALE={MODEL_SCALE} (got {})",
                node.scale
            )));
        }
        read_tables()?;
        let shape = SoftmaxShape::for_node(node)?;
        let open = |open: fn(SoftmaxShape) -> SoftmaxOpen| {
            Step::Open(OpOpen::Softmax(open(shape.clone())))
        };
        let relation =
            |stage: fn(SoftmaxShape) -> SoftmaxRelation| OpRelation::Softmax(stage(shape.clone()));
        let defer = |batch, stage| Step::Defer {
            batch,
            relation: relation(stage),
        };
        Ok(Plan(vec![
            Step::Reduce,
            open(SoftmaxOpen::Gaps),
            Step::Relation {
                proof: ProofType::SoftmaxSums,
                relation: relation(SoftmaxRelation::Sums),
            },
            open(SoftmaxOpen::Addresses),
            defer(DeferredBatch::Lookups, SoftmaxRelation::Read),
            defer(DeferredBatch::Lookups, SoftmaxRelation::Clamp),
            Step::Tie {
                relation: OpTie::Softmax(SoftmaxTie::AddressLink(shape.clone())),
            },
            defer(DeferredBatch::OneHot, SoftmaxRelation::ReadOneHot),
            defer(DeferredBatch::OneHot, SoftmaxRelation::ClampOneHot),
        ]))
    }
}

/// A softmax node's shape: its rows `F`, its padded row length `N` and its live lanes.
#[derive(Clone)]
pub struct SoftmaxShape {
    /// The softmax node.
    pub node: ComputationNode,
    /// `[F, N]`: the leading dims' product and the softmax axis.
    pub F_N: [usize; 2],
    /// The live lanes per row; lanes at or beyond it are padding.
    pub active_len: usize,
}

impl SoftmaxShape {
    /// The shape of a softmax `node`. `Err` if the node is not a softmax, has rank 0, or its
    /// live width is not in `1..=N`.
    pub fn for_node(node: &ComputationNode) -> Result<Self, ProofVerifyError> {
        let malformed = |what: &str| {
            ProofVerifyError::Malformed(format!("model: softmax node {}: {what}", node.idx))
        };
        let Op::SoftmaxLastAxis(op) = &node.operator else {
            return Err(malformed("not a softmax"));
        };
        let (&n, leading_dims) = node
            .output_dims
            .split_last()
            .ok_or_else(|| malformed("rank 0"))?;
        let active_len = op.active_len.unwrap_or(n);
        if active_len == 0 || active_len > n {
            return Err(malformed("a live width outside its row"));
        }
        Ok(Self {
            node: node.clone(),
            F_N: [
                leading_dims.iter().product::<usize>().next_power_of_two(),
                n.next_power_of_two(),
            ],
            active_len,
        })
    }

    /// The softmax node's index.
    pub fn idx(&self) -> usize {
        self.node.idx
    }

    /// `log₂ F`.
    pub fn log_f(&self) -> usize {
        self.F_N[0].log_2()
    }

    /// `log₂ N`.
    pub fn log_n(&self) -> usize {
        self.F_N[1].log_2()
    }

    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }

    /// The point `r1` the sums relation leaves the reads at.
    fn read_point<F: JoltField>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(VirtualPoly::SoftmaxRead)?
            .0)
    }

    /// The row and lane coordinates of a point of the node's cells.
    fn split<'a, F>(&self, point: &'a [F]) -> Result<(&'a [F], &'a [F]), ProofVerifyError> {
        if point.len() != self.log_f() + self.log_n() {
            return Err(ProofVerifyError::Malformed(format!(
                "softmax: a point of {} coordinates is not a point of the node's cells",
                point.len()
            )));
        }
        Ok(point.split_at(self.log_f()))
    }
}

/// `Σ_{j < active_len} Π_b w_b(j_b)` over the `bits`-bit lanes, most significant bit first, for
/// per-bit weights `w_b(0)`, `w_b(1)`: the multilinear extension of a function of the lane,
/// restricted to the live lanes, in `O(bits)`.
fn live_lane_sum<F: JoltField>(active_len: usize, zero: &[F], one: &[F]) -> F {
    // From the least significant bit up: `below` sums the lanes whose low bits are below
    // `active_len`'s, `all` every lane's. A higher bit `a` of `active_len` extends them by
    // `below' = [a = 1]·w(0)·all + w(a)·below`.
    let (mut below, mut all) = (F::zero(), F::one());
    for (bit, (z, o)) in zero.iter().rev().zip(one.iter().rev()).enumerate() {
        below = if (active_len >> bit) & 1 == 1 {
            *z * all + *o * below
        } else {
            *z * below
        };
        all *= *z + *o;
    }
    if active_len >> zero.len() != 0 {
        all
    } else {
        below
    }
}

/// A softmax advice opening.
pub enum SoftmaxOpen {
    /// The row gaps `g` and `g⁻` at the row coordinates of the reduced output point.
    Gaps(SoftmaxShape),
    /// The shift `c` at the row coordinates of the reads' point, and the table address `ũ` at
    /// the reads' point.
    Addresses(SoftmaxShape),
}

impl SoftmaxOpen {
    /// Append the openings. `Err` if the accumulator lacks a claim or point it reads.
    pub fn verify<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        match self {
            Self::Gaps(shape) => {
                let accessor = AccOpeningAccessor::new(accumulator, &shape.node);
                let r0 = accessor.get_reduced_opening()?.0;
                let (r0_row, _) = shape.split(&r0.r)?;
                let mut provider =
                    accessor.into_provider(transcript, OpeningPoint::new(r0_row.to_vec()));
                provider.append_advice(VirtualPoly::SoftmaxSumGap)?;
                provider.append_advice(VirtualPoly::SoftmaxPrevSumGap)
            }
            Self::Addresses(shape) => {
                let r1 = shape.read_point(&*accumulator)?;
                let (r1_row, _) = shape.split(&r1.r)?;
                let r1_row = OpeningPoint::new(r1_row.to_vec());
                AccOpeningAccessor::new(&mut *accumulator, &shape.node)
                    .into_provider(transcript, r1_row)
                    .append_advice(CommittedPoly::SoftmaxShift)?;
                AccOpeningAccessor::new(accumulator, &shape.node)
                    .into_provider(transcript, r1)
                    .append_advice(VirtualPoly::SoftmaxClampedAddr)
            }
        }
    }
}

/// A softmax check in the clear.
pub enum SoftmaxTie {
    /// The address link: the clamp lookup's operand `u(r1)` is `c(r1_row) − x(r1)`. Appends the
    /// operand's opening at `r1`.
    AddressLink(SoftmaxShape),
}

impl SoftmaxTie {
    /// The operand's opening the link appends.
    pub fn operand(shape: &SoftmaxShape) -> OpeningId {
        shape.ids().nodeio(Target::Input(0))
    }

    /// Append the operand's opening and check the identity. `Err` if a claim is missing or the
    /// identity fails.
    pub fn verify<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        match self {
            Self::AddressLink(shape) => {
                let r1 = shape.read_point(&*accumulator)?;
                accumulator.append_virtual(transcript, Self::operand(shape), r1)?;
                let accessor = AccOpeningAccessor::new(&*accumulator, &shape.node);
                let x = accessor.get_nodeio(Target::Input(0))?.1;
                let c = accessor.get_advice(CommittedPoly::SoftmaxShift)?.1;
                let u = accessor.get_advice(VirtualPoly::SoftmaxClampWitness)?.1;
                if u != c - x {
                    return Err(ProofVerifyError::InvalidOpeningProof(format!(
                        "softmax node {}: the lookup address is not the shift minus the logit",
                        shape.idx()
                    )));
                }
                Ok(())
            }
        }
    }
}

/// The reduction of a softmax node's output claim and its two row sums to the reads at one
/// point, over the node's cells:
///
/// ```text
/// input claim   out(r0) + γ·(S' − g(r0_row)) + γ²·(g⁻(r0_row) + S' + 1)
/// output claim  A(ρ)·v(ρ) + γ·B(ρ)·t(ρ) + γ²·B(ρ)·t⁻(ρ)
/// ```
///
/// with `A[k,j] = a(j)·eq(r0, (k,j))` and `B[k,j] = a(j)·eq(r0_row, k)`, `a` the selector of the
/// live lanes.
///
/// **S ⊆ R.** `out = a·v`, so `out(r0) = Σ A·v`; and `Σ B·t = Σ_k eq(r0_row, k)·Σ_j a(j)·t[k,j]`
/// is the live row sums' extension at `r0_row`, which is `S' − g(r0_row)` when `g` is the gap of
/// every row, likewise for `t⁻` and `g⁻`. `γ` separates the three.
pub struct SumsRelation {
    shape: SoftmaxShape,
    sum_scale: i64,
}

impl SumsRelation {
    /// The relation of a softmax node. `Err` if the softmax tables are not available.
    pub fn new(shape: SoftmaxShape) -> Result<Self, ProofVerifyError> {
        Ok(Self {
            shape,
            sum_scale: read_tables()?.tables.sum_scale,
        })
    }

    /// The reads' openings: `v`, `t`, `t⁻`.
    fn reads(&self) -> [OpeningId; 3] {
        let ids = self.shape.ids();
        [
            ids.advice(VirtualPoly::SoftmaxRead),
            ids.advice(VirtualPoly::SoftmaxSumRead),
            ids.advice(VirtualPoly::SoftmaxPrevSumRead),
        ]
    }

    /// The live-lane selector's values over a row.
    fn selector<F: JoltField>(&self) -> Vec<F> {
        (0..self.shape.F_N[1])
            .map(|j| {
                if j < self.shape.active_len {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect()
    }
}

impl SymbolicSumcheck for SumsRelation {
    crate::atlas_relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::SoftmaxSums
    }

    fn rounds(&self) -> usize {
        self.shape.log_f() + self.shape.log_n()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        let ids = self.shape.ids();
        opening::<G, _, _, _>(RelationOpening::ReducedNodeOutput(self.shape.idx()))
            + challenge::<G, _, _, _>(RelationChallenge::ClaimConstant)
            - challenge::<G, _, _, _>(RelationChallenge::Batching(0))
                * opening(RelationOpening::Id(ids.advice(VirtualPoly::SoftmaxSumGap)))
            + challenge::<G, _, _, _>(RelationChallenge::Batching(1))
                * opening(RelationOpening::Id(
                    ids.advice(VirtualPoly::SoftmaxPrevSumGap),
                ))
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        let [v, t, t_prev] = self
            .reads()
            .map(|id| opening::<G, _, _, _>(RelationOpening::Id(id)));
        derived::<G, _, _, _>(RelationDerived::Table(0)) * v
            + derived::<G, _, _, _>(RelationDerived::Table(1)) * t
            + derived::<G, _, _, _>(RelationDerived::Table(2)) * t_prev
    }
}

impl Produces for SumsRelation {
    fn produced(&self) -> Vec<OpeningId> {
        self.reads().to_vec()
    }
}

impl<F: JoltField> Relation<F> for SumsRelation {
    /// The weight tables over the node's cells: `A`, `γ·B` and `γ²·B`, for the prover.
    fn table(&self, index: usize, setup: &RelationSetup<F>) -> Result<Vec<F>, ProofVerifyError> {
        let (r0_row, _) = self.shape.split(&setup.incoming_point)?;
        let selector = self.selector::<F>();
        let n = self.shape.F_N[1];
        Ok(match index {
            0 => {
                let mut eq = EqPolynomial::evals(&setup.incoming_point);
                for (cell, value) in eq.iter_mut().enumerate() {
                    if cell % n >= self.shape.active_len {
                        *value = F::zero();
                    }
                }
                eq
            }
            1 | 2 => {
                let gamma = setup.challenge(&RelationChallenge::Batching(index - 1));
                EqPolynomial::evals(r0_row)
                    .into_iter()
                    .flat_map(|row| selector.iter().map(move |lane| gamma * row * *lane))
                    .collect()
            }
            _ => {
                return Err(ProofVerifyError::Malformed(format!(
                    "the softmax sums have three weight tables, no table {index}"
                )))
            }
        })
    }

    /// The weight tables' extensions at the bound point, in `O(log)`: the row part is an `eq`,
    /// the lane part a sum over the live lanes.
    fn derive(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<F, ProofVerifyError> {
        let RelationDerived::Table(index) = id else {
            return Err(not_derived(id));
        };
        let point = low_to_high_point(challenges);
        let (rho_row, rho_lane) = self.shape.split(&point)?;
        let (r0_row, r0_lane) = self.shape.split(&setup.incoming_point)?;
        let row = EqPolynomial::mle(r0_row, rho_row);
        let complement = |x: &F| F::one() - *x;
        Ok(match index {
            0 => {
                // Σ_{j live} eq(r0_lane, j)·eq(ρ_lane, j).
                let one: Vec<F> = r0_lane.iter().zip(rho_lane).map(|(a, b)| *a * *b).collect();
                let zero: Vec<F> = r0_lane
                    .iter()
                    .zip(rho_lane)
                    .map(|(a, b)| complement(a) * complement(b))
                    .collect();
                row * live_lane_sum(self.shape.active_len, &zero, &one)
            }
            1 | 2 => {
                // Σ_{j live} eq(ρ_lane, j).
                let zero: Vec<F> = rho_lane.iter().map(complement).collect();
                setup.challenge(&RelationChallenge::Batching(index - 1))
                    * row
                    * live_lane_sum(self.shape.active_len, &zero, rho_lane)
            }
            _ => return Err(not_derived(id)),
        })
    }
}

impl<F: JoltField> SetupRelation<F> for SumsRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        let gamma: F = transcript.challenge_scalar();
        let gamma_sq = gamma * gamma;
        let sum_scale = F::from_i64(self.sum_scale);
        Ok(RelationSetup {
            incoming_point: accumulator.get_node_output_opening(self.shape.idx())?.0.r,
            challenges: vec![
                (RelationChallenge::Batching(0), gamma),
                (RelationChallenge::Batching(1), gamma_sq),
                (
                    RelationChallenge::ClaimConstant,
                    gamma * sum_scale + gamma_sq * (sum_scale + F::one()),
                ),
            ],
        })
    }
}

impl AtlasRelation for SumsRelation {
    type Spec = SoftmaxLastAxis;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
}

impl<F: JoltField> EngineRelation<F> for SumsRelation {}

/// The table read at the reads' point: the output, sum and shifted sum tables through the
/// table address, with the address itself as the read's `raf`.
#[derive(Clone)]
pub struct ReadProvider {
    /// The softmax node's shape.
    pub shape: SoftmaxShape,
}

impl<F: JoltField> ReadRafProvider<F> for ReadProvider {
    fn log_K(&self) -> usize {
        SOFTMAX_CLAMP_BOUND
    }

    fn r(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
        self.shape.read_point(accumulator)
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (
            VirtualPoly::SoftmaxAddrRa(self.shape.idx()),
            SumcheckId::NodeExecution(self.shape.idx()),
        )
    }

    fn raf_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> Result<F, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, &self.shape.node)
            .get_advice(VirtualPoly::SoftmaxClampedAddr)?
            .1)
    }

    fn rv_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> Result<F, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, &self.shape.node)
            .get_advice(VirtualPoly::SoftmaxRead)?
            .1)
    }

    fn rv_claims(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let accessor = AccOpeningAccessor::new(accumulator, &self.shape.node);
        Ok(vec![
            accessor.get_advice(VirtualPoly::SoftmaxRead)?.1,
            accessor.get_advice(VirtualPoly::SoftmaxSumRead)?.1,
            accessor.get_advice(VirtualPoly::SoftmaxPrevSumRead)?.1,
        ])
    }
}

/// One-hot encoding of the softmax table read's addresses.
pub struct ReadRaEncoding {
    /// Index of the softmax node.
    pub node_idx: usize,
}

impl RaOneHotEncoding for ReadRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::SoftmaxAddrRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SoftmaxRead(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SoftmaxAddrRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        SOFTMAX_CLAMP_BOUND
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), SOFTMAX_CLAMP_BOUND)
    }
}

/// A softmax relation, as the plan holds it.
pub enum SoftmaxRelation {
    /// The sums relation ([`SumsRelation`]).
    Sums(SoftmaxShape),
    /// The table read at the reads' point.
    Read(SoftmaxShape),
    /// The clamp lookup `ũ = Clamp[u]` at the reads' point.
    Clamp(SoftmaxShape),
    /// The one-hot checks of the table address.
    ReadOneHot(SoftmaxShape),
    /// The one-hot checks of the clamp address.
    ClampOneHot(SoftmaxShape),
}

impl SoftmaxRelation {
    /// The relation's verifiers.
    /// `Err` if the accumulator lacks a claim a setup reads.
    pub fn verifiers<F: JoltField, T: Transcript>(
        self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<Vec<Box<dyn DynRelationVerifier<F, T>>>, ProofVerifyError> {
        match self {
            Self::Sums(shape) => Ok(vec![Box::new(RelationVerifier::new(
                SumsRelation::new(shape)?,
                &*accumulator,
                transcript,
            )?)]),
            Self::Read(shape) => {
                let read = read_tables()?;
                Ok(vec![shout::read_raf_verifier(
                    &ReadProvider { shape },
                    vec![
                        read.tables.out.clone(),
                        read.tables.sum.clone(),
                        read.sum_prev.clone(),
                    ],
                    &*accumulator,
                    transcript,
                )?])
            }
            Self::Clamp(shape) => {
                let provider: OpLookupProvider<SoftmaxSignificanceClampOperands> =
                    OpLookupProvider::new(shape.node);
                Ok(vec![Box::new(
                    provider.read_raf_verify::<F, T, SoftmaxClampTable<XLEN>, XLEN>(
                        accumulator,
                        transcript,
                    )?,
                )])
            }
            Self::ReadOneHot(shape) => Ok(shout::ra_onehot_verifiers(
                &ReadRaEncoding {
                    node_idx: shape.idx(),
                },
                &*accumulator,
                transcript,
            )?
            .into()),
            Self::ClampOneHot(shape) => {
                let provider: OpLookupProvider<SoftmaxSignificanceClampOperands> =
                    OpLookupProvider::new(shape.node);
                Ok(
                    shout::ra_onehot_verifiers(&provider.encoding(), &*accumulator, transcript)?
                        .into(),
                )
            }
        }
    }
}
