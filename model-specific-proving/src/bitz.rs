//! Commit integer columns and authenticate their multilinear evaluations with BitZ.

use crate::field::{Mixing, Scalar, F};
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
    pub planes: Option<Vec<F>>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Proof {
    rows: SingleSumcheckProof<F>,
    #[serde(with = "BitProof")]
    bits: world_transcript::Proof,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(remote = "world_transcript::Proof")]
struct BitProof {
    #[serde(with = "bytes")]
    narg_string: Vec<u8>,
    #[serde(with = "bytes")]
    hints: Vec<u8>,
}

pub(crate) mod bytes {
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(data: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(data)
    }

    pub fn deserialize<'d, D: Deserializer<'d>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        struct Bytes;
        impl<'d> serde::de::Visitor<'d> for Bytes {
            type Value = Vec<u8>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("bytes")
            }
            fn visit_bytes<E: serde::de::Error>(self, bytes: &[u8]) -> Result<Vec<u8>, E> {
                Ok(bytes.to_vec())
            }
            fn visit_byte_buf<E: serde::de::Error>(self, bytes: Vec<u8>) -> Result<Vec<u8>, E> {
                Ok(bytes)
            }
        }
        deserializer.deserialize_byte_buf(Bytes)
    }
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
        let result = self.commit_detached(source)?;
        outer.append_bytes(&result.root);
        Ok(result)
    }

    pub(crate) fn commit_detached(
        &self,
        source: impl Fn(usize, usize) -> i128,
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
        outer: &mut impl Mixing<F>,
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
            if let Some(planes) = &q.planes {
                if planes.len() != placement.range.bits() || placement.range.offset() != 0 {
                    return Err("invalid BitZ plane weights");
                }
                outer.append_values(planes);
            }
            outer.append(&(q.slot as u64));
            outer.append_values(&q.claim.point);
            outer.append_values(&[q.claim.value]);
        }
        let mix = outer.mix(queries.len(), queries.len())?;
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
                    slot: q.slot,
                    binary: q.planes.is_some(),
                    column,
                    row,
                    scale,
                    planes: q.planes.clone().unwrap_or_else(|| {
                        (0..placement.range.bits())
                            .map(|i| F::from_u128(1 << i))
                            .collect()
                    }),
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
        self.reduce(committed, queries, outer, None::<fn(usize, usize) -> i128>)?
            .open(&mut std::collections::HashMap::new())
    }

    pub(crate) fn reduce(
        &self,
        committed: Committed,
        queries: &[Query],
        outer: &mut ProverState<Keccak>,
        source: Option<impl Fn(usize, usize) -> i128 + Sync>,
    ) -> Result<Pending<'_>> {
        use rayon::prelude::*;
        let (instance, queries) = self.prepare(queries, outer)?;
        let pairs = queries
            .par_iter()
            .map(|q| {
                let mut values = vec![F::zero(); ROWS];
                if let Some(source) = &source {
                    let layout = q.placement.range;
                    let count = (1 << layout.vars).min(ROWS);
                    for (col, &weight) in q.column.iter().enumerate() {
                        let w = weight * q.scale;
                        for (row, v) in values[..count].iter_mut().enumerate() {
                            let x = (source(q.slot, col * ROWS + row) + layout.offset()) as u128;
                            if q.binary {
                                let mut bits = x;
                                while bits != 0 {
                                    *v += w * q.planes[bits.trailing_zeros() as usize];
                                    bits &= bits - 1;
                                }
                            } else {
                                *v += w.mul_integer(x as i128);
                            }
                        }
                    }
                } else {
                    for (bit, columns) in q.columns().enumerate() {
                        let scale = q.scale * q.planes[bit];
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
        Ok(Pending {
            setup: self,
            committed,
            rows,
            linear,
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
                let scale = scale * q.planes[bit];
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

pub(crate) struct Pending<'a> {
    setup: &'a Group,
    committed: Committed,
    rows: SingleSumcheckProof<F>,
    linear: LinearClaim<Scalar>,
}

impl Pending<'_> {
    pub(crate) fn open(
        self,
        cache: &mut std::collections::HashMap<usize, world_prover::BitZProver<Scalar>>,
    ) -> Result<Proof> {
        let Committed {
            data,
            packed,
            mut channel,
            ..
        } = self.committed;
        cache
            .entry(self.setup.parameters.shape().columns())
            .or_insert_with(|| world_prover::BitZProver::new(self.setup.parameters, 8))
            .prove(&self.linear, &self.setup.pcs, &data, packed, &mut channel)
            .map_err(|_| "BitZ evaluation proof failed")?;
        Ok(Proof {
            rows: self.rows,
            bits: channel.finish(),
        })
    }
}

struct WeightedQuery<'a> {
    placement: &'a Placement,
    slot: usize,
    binary: bool,
    column: Vec<F>,
    row: Vec<F>,
    scale: F,
    planes: Vec<F>,
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
                    planes: None,
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
