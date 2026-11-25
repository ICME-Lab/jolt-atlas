mod field;

use std::sync::OnceLock;

use jolt_field::One;
use world_common::{BitZParams, LinearClaim, Root, SecurityLevel, Shape};
use world_field::{gf128::smallest_generator, F128};
use world_pcs::{Pcs, ProverData};
use world_transcript::{build_prover, build_verifier, Proof as WorldProof};

use super::*;
use crate::protocol::{
    session::{instance, Column},
    tensor::{eq, weights},
};
use field::{BitzField as F, Fq};

const ROW_VARS: usize = 7;
const ROWS: usize = 1 << ROW_VARS;
const DOMAIN: &[u8] = b"atlas/model-specific/world-bitz";

pub struct BitzScheme;

impl ModelPcs for BitzScheme {
    type Field = F;
    const MIN_FIELD_BITS: u32 = 0;
    type AdviceProverSetup = ();
    type AdviceVerifierSetup = ();
    type AdviceCommitments = [u8; 32];
    type AdviceState = (Vec<F128>, ProverData, world_transcript::ProverState);
    type AdviceOpening = (Messages<F>, WorldProof);

    type AdvicePlan = (Vec<usize>, BitZParams<Fq>, Pcs);

    fn plan(layout: &Layout) -> Result<Self::AdvicePlan> {
        static GENERATOR: OnceLock<F128> = OnceLock::new();
        let (vars, offsets) = layout.place(ROW_VARS, true)?;
        let vars = vars.max(world_common::shape::MIN_LOG_BITS);
        let shape = Shape::new(ROW_VARS, vars - ROW_VARS).map_err(|_| "invalid BitZ shape")?;
        let params = BitZParams::new(shape, *GENERATOR.get_or_init(smallest_generator))
            .map_err(|_| "invalid BitZ parameters")?;
        let pcs = Pcs::new(&shape, SecurityLevel::Bits128).map_err(|_| "unsupported BitZ size")?;
        Ok((offsets, params, pcs))
    }

    fn commit(
        layout: &Layout,
        advice: &mut Advice,
        (offsets, _, pcs): &Self::AdvicePlan,
        _: &(),
    ) -> Result<([u8; 32], Self::AdviceState)> {
        let mut packed = F128::zeroed_vec(pcs.packed_len());
        for &slot in &layout.slots {
            let cells = 1 << slot.cell_vars;
            advice.fill_bits::<F>(slot, |i| {
                let i = offsets[slot.id] + i / cells * cells.max(ROWS) + i % cells;
                let word = &mut packed[i / ROWS];
                *word = F128::from(word.to_u128() | (1 << (i % ROWS)));
            })?;
        }
        let mut transcript = build_prover(DOMAIN, b"" as &[u8]);
        let (root, data) = pcs
            .commit(&packed, &mut transcript)
            .map_err(|_| "BitZ commitment failed")?;
        Ok((root.0, (packed, data, transcript)))
    }

    fn bind(root: &[u8; 32], t: &mut impl Transcript) {
        t.append_bytes(DOMAIN);
        t.append_bytes(root);
    }

    fn open(
        layout: &Layout,
        openings: Vec<Opening<F>>,
        _: &[u8; 32],
        (packed, data, mut transcript): Self::AdviceState,
        (offsets, params, pcs): &Self::AdvicePlan,
        _: &(),
        t: &mut ProverState<Keccak>,
    ) -> Result<Self::AdviceOpening> {
        let mut messages = Messages::default();
        let mut io = Session::prove(F::prover(t), &mut messages);
        let claim = batch(layout, openings, offsets, params, Some(&packed), &mut io)?;
        world_prover::BitZProver::new(*params, 8)
            .prove(&claim, pcs, &data, packed, &mut transcript)
            .map_err(|_| "BitZ opening failed")?;
        Ok((messages, transcript.finish()))
    }

    fn verify(
        layout: &Layout,
        openings: Vec<Opening<F>>,
        root: &[u8; 32],
        (messages, proof): &Self::AdviceOpening,
        (offsets, params, pcs): &Self::AdvicePlan,
        _: &(),
        t: &mut VerifierState<'_, Keccak>,
    ) -> Result<()> {
        let mut io = messages.read(t);
        let claim = batch(layout, openings, offsets, params, None, &mut io)?;
        io.finish()?;
        let mut transcript = build_verifier(DOMAIN, b"" as &[u8], proof);
        let root = pcs
            .receive_commitment(Root(*root), &mut transcript)
            .map_err(|_| "invalid BitZ commitment")?;
        world_verifier::BitZVerifier::new(*params, 8)
            .verify(&claim, pcs, &root, transcript)
            .map_err(|_| "invalid BitZ opening")
    }
}

struct Query {
    column: usize,
    row_point: Vec<F>,
    axes: Vec<F>,
    cells: Vec<F>,
}

impl Query {
    fn new(slot: Slot, offset: usize, opening: Opening<F>, mix: F) -> Self {
        let (axis, cells) = opening
            .claim
            .point
            .split_at(opening.claim.point.len() - slot.cell_vars);
        let (high, low) = cells.split_at(cells.len().saturating_sub(ROW_VARS));
        let mut row_point = vec![F::zero(); ROW_VARS - low.len()];
        row_point.extend_from_slice(low);
        let mut axes = opening.planes.unwrap_or_else(|| weights(axis));
        if let Kind::Bits { planes } = slot.kind {
            axes.truncate(planes);
        }
        for w in &mut axes {
            *w *= mix;
        }
        Self {
            column: offset / ROWS,
            row_point,
            axes,
            cells: weights(high),
        }
    }

    fn column_weights(&self, scale: F) -> impl Iterator<Item = F> + '_ {
        self.axes
            .iter()
            .map(move |&axis| axis * scale)
            .flat_map(|axis| self.cells.iter().map(move |&cell| axis * cell))
    }

    fn project(&self, words: &[F128]) -> Vec<F> {
        let mut rows = vec![F::zero(); ROWS];
        for (word, weight) in words[self.column..]
            .iter()
            .zip(self.column_weights(F::one()))
        {
            let mut bits = word.to_u128();
            while bits != 0 {
                rows[bits.trailing_zeros() as usize] += weight;
                bits &= bits - 1;
            }
        }
        rows
    }
}

// Fold only the low seven coordinates. The terminal column weights need not be an MLE point.
fn batch(
    layout: &Layout,
    openings: Vec<Opening<F>>,
    offsets: &[usize],
    params: &BitZParams<Fq>,
    words: Option<&[F128]>,
    io: &mut Session<'_, '_, F>,
) -> Result<LinearClaim<Fq>> {
    let mix = io.point(openings.len())?;
    let sum = openings
        .iter()
        .zip(&mix)
        .map(|(q, &m)| q.claim.value * m)
        .sum();
    let queries: Vec<_> = openings
        .into_iter()
        .zip(mix)
        .map(|(q, m)| Query::new(layout.slots[q.slot], offsets[q.slot], q, m))
        .collect();
    let (end, []) = io.fold(
        instance(sum, ROW_VARS, 2),
        || {
            queries
                .iter()
                .flat_map(|q| {
                    [
                        Column::owned(q.project(words.expect("prover has packed bits"))),
                        Column::owned(weights(&q.row_point)),
                    ]
                })
                .collect::<Vec<_>>()
        },
        [],
        |v| v.chunks_exact(2).map(|p| p[0] * p[1]).sum(),
    )?;
    let mut columns = vec![Fq::from(0u64); params.shape().columns()];
    for q in queries {
        let scale = eq(&q.row_point, &end.challenges);
        for (dst, weight) in columns[q.column..].iter_mut().zip(q.column_weights(scale)) {
            *dst += weight.0;
        }
    }
    let rows = weights::<F>(&end.challenges)
        .into_iter()
        .map(|v| v.0)
        .collect();
    LinearClaim::new(params, rows, columns, end.output_claim.0).map_err(|_| "invalid BitZ claim")
}
