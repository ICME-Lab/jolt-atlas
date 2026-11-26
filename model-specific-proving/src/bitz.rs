//! Commit integer columns and authenticate their multilinear evaluations with BitZ.

use crate::field::{Scalar, F};
use crate::fold::{agreement, basis, InnerProducts};
use crate::norm::{Claim, Width};
use jolt_field::{Ring, Zero};
use jolt_sumcheck::single::{
    prove_single, verify_single, SingleFormat, SingleInstance, SingleReduction, SingleSumcheckProof,
};
use jolt_transcript::{FieldTranscript, Keccak, ProverState, Transcript, VerifierState};
use std::{ops::Range, sync::OnceLock};
use world_common::{shape::MIN_LOG_BITS, BitZParams, LinearClaim, Root, SecurityLevel, Shape};
use world_field::{gf128::smallest_generator, F128};
use world_pcs::Pcs;
use world_transcript::{build_prover, build_verifier};

type Result<T> = std::result::Result<T, &'static str>;
const ROWS: usize = 512;
const ROW_BITS: usize = 9;
const WORDS: usize = ROWS / 128;
const DOMAIN: &[u8] = b"atlas/model/bitz/ranges/v2";

#[derive(Clone, Copy)]
pub struct RangeColumn {
    pub vars: usize,
    pub width: Width,
}

#[derive(Clone, Copy)]
struct Placement {
    range: RangeColumn,
    start: usize,
    columns: usize,
}

impl RangeColumn {
    fn offset(self) -> i128 {
        match self.width {
            Width::Signed(n) => 1 << (n - 1),
            Width::Unsigned(_) => 0,
        }
    }

    fn bits(self) -> usize {
        match self.width {
            Width::Signed(n) | Width::Unsigned(n) => n.into(),
        }
    }
}

pub struct Group {
    placements: Vec<Placement>,
    parameters: BitZParams<Scalar>,
    pcs: Pcs,
}

pub struct Query {
    pub slot: usize,
    pub claim: Claim<F>,
}

pub struct Proof {
    rows: SingleSumcheckProof<F>,
    bits: world_transcript::Proof,
}

type RootBytes = [u8; 32];
pub struct Committed {
    pub root: RootBytes,
    packed: Vec<F128>,
    data: world_pcs::ProverData,
    channel: world_transcript::ProverState,
}

impl Group {
    pub fn new(columns: &[RangeColumn]) -> Result<Self> {
        if columns.is_empty()
            || columns
                .iter()
                .any(|r| r.vars > 28 || r.bits() > 65 || matches!(r.width, Width::Signed(0)))
        {
            return Err("invalid range layout");
        }
        let mut used = 0usize;
        let placements = columns
            .iter()
            .map(|&range| {
                let columns = 1 << range.vars.saturating_sub(ROW_BITS);
                let start = used;
                used += columns * range.bits();
                Placement {
                    range,
                    start,
                    columns,
                }
            })
            .collect();
        let size = used.next_power_of_two().ilog2() as usize;
        let shape = Shape::new(ROW_BITS, size.max(MIN_LOG_BITS - ROW_BITS))
            .map_err(|_| "unsupported BitZ shape")?;
        static GENERATOR: OnceLock<F128> = OnceLock::new();
        let parameters = BitZParams::new(shape, *GENERATOR.get_or_init(smallest_generator))
            .map_err(|_| "invalid BitZ parameters")?;
        let pcs = Pcs::new(&shape, SecurityLevel::Bits128).map_err(|_| "invalid BitZ PCS")?;
        Ok(Self {
            placements,
            parameters,
            pcs,
        })
    }

    /// A slot includes the full Boolean domain. Callers pad tensor holes with zero.
    pub fn commit(
        &self,
        source: impl Fn(usize, usize) -> i128,
        outer: &mut ProverState<Keccak>,
    ) -> Result<Committed> {
        let mut packed = vec![F128::from(0u128); self.pcs.packed_len()];
        for (slot, placement) in self.placements.iter().enumerate() {
            let bit_count = placement.range.bits();
            let offset = placement.range.offset();
            let cells = 1 << placement.range.vars;
            for col in 0..placement.columns {
                for group in 0..WORDS {
                    let mut planes = [0u128; 65];
                    let start = col * ROWS + group * 128;
                    for index in start..(start + 128).min(cells) {
                        let value = source(slot, index)
                            .checked_add(offset)
                            .ok_or("range overflow")?;
                        if !(0..1i128 << bit_count).contains(&value) {
                            return Err("integer outside committed range");
                        }
                        let mut value = value as u128;
                        while value != 0 {
                            planes[value.trailing_zeros() as usize] |= 1 << (index - start);
                            value &= value - 1;
                        }
                    }
                    for (bit, &encoded) in planes[..bit_count].iter().enumerate() {
                        let pos =
                            ((placement.start + bit * placement.columns + col) * WORDS) + group;
                        packed[pos] = F128::from(encoded);
                    }
                }
            }
        }
        let mut channel = build_prover(DOMAIN, b"");
        let (root, data) = self
            .pcs
            .commit(&packed, &mut channel)
            .map_err(|_| "BitZ commitment failed")?;
        outer.append_bytes(&root.0);
        Ok(Committed {
            root: root.0,
            packed,
            data,
            channel,
        })
    }

    pub fn bind_root(&self, root: RootBytes, outer: &mut VerifierState<'_, Keccak>) {
        outer.append_bytes(&root);
    }

    fn prepare(
        &self,
        queries: &[Query],
        outer: &mut impl FieldTranscript<F>,
    ) -> Result<(SingleInstance<F>, Vec<WeightedQuery<'_>>)> {
        if queries.is_empty() {
            return Err("missing BitZ evaluation");
        }
        outer.append(&(queries.len() as u64));
        for q in queries {
            let placement = self.placements.get(q.slot).ok_or("invalid BitZ slot")?;
            if q.claim.point.len() != placement.range.vars {
                return Err("invalid BitZ point");
            }
            outer.append(&(q.slot as u64));
            outer.append_values(&q.claim.point);
            outer.append_values(&[q.claim.value]);
        }
        let mix = outer
            .challenge_vector(queries.len())
            .map_err(|_| "invalid BitZ challenges")?;
        let mut total = F::zero();
        let queries = queries
            .iter()
            .zip(mix)
            .map(|(q, scale)| {
                let placement = &self.placements[q.slot];
                let split = placement.range.vars.saturating_sub(ROW_BITS);
                let column = basis(&q.claim.point[..split]);
                let mut row = vec![F::zero(); ROW_BITS - (q.claim.point.len() - split)];
                row.extend_from_slice(&q.claim.point[split..]);
                total += scale * (q.claim.value + F::from_i128(placement.range.offset()));
                WeightedQuery {
                    placement,
                    column,
                    row,
                    scale,
                }
            })
            .collect();
        let instance = SingleInstance {
            input_claim: total,
            rounds: ROW_BITS,
            degree: 2,
            format: SingleFormat::Compressed,
        };
        Ok((instance, queries))
    }

    pub fn prove(
        &self,
        committed: Committed,
        queries: &[Query],
        outer: &mut ProverState<Keccak>,
    ) -> Result<Proof> {
        let (instance, queries) = self.prepare(queries, outer)?;
        let pairs = queries
            .iter()
            .map(|q| {
                let mut values = vec![F::zero(); ROWS];
                for (bit, columns) in q.columns().enumerate() {
                    let scale = q.scale * F::from_u128(1 << bit);
                    for (col, &weight) in columns.zip(&q.column) {
                        for (word, output) in values.chunks_exact_mut(128).enumerate() {
                            let mut mask = u128::from_le_bytes(
                                committed.packed[col * WORDS + word].to_bytes(),
                            );
                            while mask != 0 {
                                output[mask.trailing_zeros() as usize] += scale * weight;
                                mask &= mask - 1;
                            }
                        }
                    }
                }
                (basis(&q.row), values)
            })
            .collect();
        let mut witness = InnerProducts {
            pairs,
            rounds: ROW_BITS,
        };
        let (rows, end) =
            prove_single(&mut witness, &instance, outer).map_err(|_| "BitZ row Sumcheck failed")?;
        let linear = self.linear_claim(&queries, end)?;
        let Committed {
            data,
            packed,
            mut channel,
            ..
        } = committed;
        world_prover::BitZProver::new(self.parameters, 8)
            .prove(&linear, &self.pcs, &data, packed, &mut channel)
            .map_err(|_| "BitZ evaluation proof failed")?;
        Ok(Proof {
            rows,
            bits: channel.finish(),
        })
    }

    pub fn verify(
        &self,
        root: RootBytes,
        queries: &[Query],
        proof: &Proof,
        outer: &mut VerifierState<'_, Keccak>,
    ) -> Result<()> {
        let (instance, queries) = self.prepare(queries, outer)?;
        let end = verify_single(&instance, &proof.rows, outer)
            .map_err(|_| "invalid BitZ row Sumcheck")?;
        let linear = self.linear_claim(&queries, end)?;
        let mut channel = build_verifier(DOMAIN, b"", &proof.bits);
        let committed = self
            .pcs
            .receive_commitment(Root(root), &mut channel)
            .map_err(|_| "invalid BitZ commitment")?;
        world_verifier::BitZVerifier::new(self.parameters, 8)
            .verify(&linear, &self.pcs, &committed, channel)
            .map_err(|_| "invalid BitZ evaluation proof")?;
        Ok(())
    }

    fn linear_claim(
        &self,
        queries: &[WeightedQuery<'_>],
        end: SingleReduction<F>,
    ) -> Result<LinearClaim<Scalar>> {
        let mut column_weights = vec![F::zero(); self.parameters.shape().columns()];
        for q in queries {
            let scale = q.scale * agreement(&q.row, &end.challenges);
            for (bit, columns) in q.columns().enumerate() {
                let scale = scale * F::from_u128(1 << bit);
                for (col, &weight) in columns.zip(&q.column) {
                    column_weights[col] += weight * scale;
                }
            }
        }
        LinearClaim::new(
            &self.parameters,
            basis(&end.challenges).into_iter().map(|x| x.0).collect(),
            column_weights.into_iter().map(|x| x.0).collect(),
            end.output_claim.0,
        )
        .map_err(|_| "invalid BitZ linear claim")
    }
}

struct WeightedQuery<'a> {
    placement: &'a Placement,
    column: Vec<F>,
    row: Vec<F>,
    scale: F,
}

impl WeightedQuery<'_> {
    fn columns(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        (0..self.placement.range.bits()).map(|i| {
            let start = self.placement.start + i * self.placement.columns;
            start..start + self.placement.columns
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_transcript::{prover_transcript, verifier_transcript};

    #[test]
    fn opens_signed_and_unsigned_values_at_different_points() {
        let data = [[-5, 4, 8, 0], [3, 1, 7, 2]];
        let group = Group::new(&[
            RangeColumn {
                vars: 2,
                width: Width::Signed(5),
            },
            RangeColumn {
                vars: 2,
                width: Width::Unsigned(3),
            },
        ])
        .unwrap();
        let points = [[13, 19], [31, 37]].map(|p| p.map(F::from_u64));
        let mut queries: Vec<_> = points
            .into_iter()
            .enumerate()
            .map(|(slot, point)| {
                let value = basis(&point)
                    .into_iter()
                    .zip(data[slot])
                    .map(|(w, x)| w * F::from_i128(x))
                    .sum();
                Query {
                    slot,
                    claim: Claim {
                        point: point.to_vec(),
                        value,
                    },
                }
            })
            .collect();
        let tag = [0x63; 32];
        let mut prover = prover_transcript::<Keccak>(&tag, b"");
        let committed = group.commit(|slot, i| data[slot][i], &mut prover).unwrap();
        let root = committed.root;
        let proof = group.prove(committed, &queries, &mut prover).unwrap();
        let mut verifier = verifier_transcript::<Keccak>(&tag, b"", prover.narg_string());
        group.bind_root(root, &mut verifier);
        group.verify(root, &queries, &proof, &mut verifier).unwrap();
        verifier.check_eof().unwrap();

        queries[0].claim.value += F::from_u64(1);
        let mut verifier = verifier_transcript::<Keccak>(&tag, b"", prover.narg_string());
        group.bind_root(root, &mut verifier);
        assert!(group.verify(root, &queries, &proof, &mut verifier).is_err());
    }
}
