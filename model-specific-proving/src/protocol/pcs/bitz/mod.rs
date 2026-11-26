pub(super) mod field;
mod layout;

use std::sync::OnceLock;

use jolt_field::Zero;
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
const DOMAIN: &[u8] = b"atlas/model-specific/world-bitz/groups/v1";

use layout::Plan;

type State = Vec<(Vec<F128>, ProverData, world_transcript::ProverState)>;
pub(super) type OpeningProof = (Messages<F>, Vec<WorldProof>);

pub(super) fn plan(layout: &Layout) -> Result<Plan> {
    if F::MODULUS_BITS < layout.integer_bits {
        return Err("field is too small for the model's integer relations");
    }
    Plan::new(layout)
}

pub(super) fn commit(
    layout: &Layout,
    advice: &mut Advice,
    plan: &Plan,
) -> Result<(Vec<[u8; 32]>, State)> {
    let mut packed: Vec<_> = plan
        .groups
        .iter()
        .map(|(_, pcs)| F128::zeroed_vec(pcs.packed_len()))
        .collect();
    let mut slots = layout.slots.iter().copied().peekable();
    while let Some(slot) = slots.next() {
        if let Kind::Lookup { .. } = slot.kind {
            let (addresses, shift, mask) = advice.take_lookup(slot)?;
            let mut digits = vec![(slot, shift, mask)];
            while let Some(&next) = slots.peek() {
                if !matches!(&advice.0[next.id], Some(Part::Lookup { addresses: next_addresses, .. })
                    if Arc::ptr_eq(&addresses, next_addresses))
                {
                    break;
                }
                let (_, shift, mask) = advice.take_lookup(next)?;
                digits.push((next, shift, mask));
                slots.next();
            }
            let mut block = vec![[0u128; 16]; digits.len()];
            let words = addresses.len().div_ceil(128);
            let stride = addresses.len().max(ROWS) / 128;
            for word in 0..words {
                block.fill([0; 16]);
                let start = word * 128;
                for cell in start..(start + 128).min(addresses.len()) {
                    let address = addresses.get(cell);
                    let bit = 1 << (cell - start);
                    for (&(_, shift, mask), dst) in digits.iter().zip(&mut block) {
                        dst[((address >> shift) & mask) as usize] |= bit;
                    }
                }
                for (&(slot, _, mask), src) in digits.iter().zip(&block) {
                    let (group, offset) = plan.slots[slot.id];
                    for (plane, &value) in src.iter().take(mask as usize + 1).enumerate() {
                        packed[group][offset / 128 + plane * stride + word] = F128::from(value);
                    }
                }
            }
        } else {
            let cells = 1 << slot.cell_vars;
            let (group, offset) = plan.slots[slot.id];
            advice.fill_bits(slot, |i| {
                let i = offset + i / cells * cells.max(ROWS) + i % cells;
                let word = &mut packed[group][i / 128];
                *word = F128::from(word.to_u128() | (1 << (i % 128)));
            })?;
        }
    }
    packed
        .into_iter()
        .zip(&plan.groups)
        .map(|(packed, (_, pcs))| {
            let mut transcript = build_prover(DOMAIN, b"" as &[u8]);
            let (root, data) = pcs
                .commit(&packed, &mut transcript)
                .map_err(|_| "BitZ commitment failed")?;
            Ok((root.0, (packed, data, transcript)))
        })
        .collect()
}

pub(super) fn bind(roots: &[[u8; 32]], t: &mut impl Transcript) {
    t.append_bytes(DOMAIN);
    t.append(&(roots.len() as u64));
    for root in roots {
        t.append_bytes(root);
    }
}

pub(super) fn open(
    layout: &Layout,
    openings: Vec<Opening<F>>,
    state: State,
    plan: &Plan,
    t: &mut ProverState<Keccak>,
) -> Result<OpeningProof> {
    let mut messages = Messages::default();
    let mut io = Session::prove(F::prover(t), &mut messages);
    let proofs = plan
        .queries(openings)
        .into_iter()
        .zip(state)
        .zip(&plan.groups)
        .map(
            |((requests, (packed, data, mut transcript)), (params, pcs))| {
                let claim = batch(
                    layout,
                    requests,
                    &plan.slots,
                    params,
                    Some(&packed),
                    &mut io,
                )?;
                world_prover::BitZProver::new(*params, 8)
                    .prove(&claim, pcs, &data, packed, &mut transcript)
                    .map_err(|_| "BitZ opening failed")?;
                Ok(transcript.finish())
            },
        )
        .collect::<Result<_>>()?;
    Ok((messages, proofs))
}

pub(super) fn verify(
    layout: &Layout,
    openings: Vec<Opening<F>>,
    roots: &[[u8; 32]],
    (messages, proofs): &OpeningProof,
    plan: &Plan,
    t: &mut VerifierState<'_, Keccak>,
) -> Result<()> {
    if roots.len() != plan.groups.len() || proofs.len() != plan.groups.len() {
        return Err("invalid BitZ group count");
    }
    let mut io = messages.read(t);
    for (((requests, root), proof), (params, pcs)) in plan
        .queries(openings)
        .into_iter()
        .zip(roots)
        .zip(proofs)
        .zip(&plan.groups)
    {
        let claim = batch(layout, requests, &plan.slots, params, None, &mut io)?;
        let mut transcript = build_verifier(DOMAIN, b"" as &[u8], proof);
        let root = pcs
            .receive_commitment(Root(*root), &mut transcript)
            .map_err(|_| "invalid BitZ commitment")?;
        world_verifier::BitZVerifier::new(*params, 8)
            .verify(&claim, pcs, &root, transcript)
            .map_err(|_| "invalid BitZ opening")?;
    }
    io.finish()?;
    Ok(())
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
}

fn project(queries: &[Query], words: &[F128]) -> Vec<Vec<F>> {
    let mut projected = vec![vec![F::zero(); ROWS]; queries.len()];
    let mut order: Vec<_> = (0..queries.len()).collect();
    let span = |i: usize| {
        let q = &queries[i];
        (q.column, q.axes.len() * q.cells.len())
    };
    order.sort_unstable_by_key(|&i| span(i));
    for indices in order.chunk_by(|&i, &j| span(i) == span(j)) {
        let first = &queries[indices[0]];
        let cols = first.axes.len() * first.cells.len();
        let mut weighted = vec![F::zero(); indices.len()];
        for (column, word) in words[first.column..][..cols].iter().enumerate() {
            let mut bits = word.to_u128();
            if bits == 0 {
                continue;
            }
            for (&i, w) in indices.iter().zip(&mut weighted) {
                let q = &queries[i];
                let cells = q.cells.len();
                *w = q.axes[column / cells] * q.cells[column % cells];
            }
            while bits != 0 {
                let row = bits.trailing_zeros() as usize;
                for (&i, &weight) in indices.iter().zip(&weighted) {
                    projected[i][row] += weight;
                }
                bits &= bits - 1;
            }
        }
    }
    projected
}

// Fold only the low seven coordinates. The terminal column weights need not be an MLE point.
fn batch(
    layout: &Layout,
    openings: Vec<Opening<F>>,
    slots: &[(usize, usize)],
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
        .map(|(q, m)| Query::new(layout.slots[q.slot], slots[q.slot].1, q, m))
        .collect();
    let (end, []) = io.fold(
        instance(sum, ROW_VARS, 2),
        || {
            let rows = project(&queries, words.expect("prover has packed bits"));
            queries
                .iter()
                .zip(rows)
                .flat_map(|(q, rows)| [Column::owned(rows), Column::owned(weights(&q.row_point))])
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
