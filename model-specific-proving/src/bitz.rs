//! Commit integer columns and authenticate their multilinear evaluations with BitZ.

use crate::field::{Mixing, Scalar, F};
use crate::fold::{agreement, basis, InnerProducts};
use crate::norm::{Claim, Width};
use jolt_field::{One, Ring, Zero};
use jolt_sumcheck::single::{
    prove_single, verify_single, SingleFormat, SingleInstance, SingleReduction, SingleSumcheckProof,
};
use jolt_transcript::{Keccak, ProverState, Transcript, VerifierState};
use rayon::prelude::*;
use std::{collections::HashMap, ops::Range, sync::OnceLock};
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

#[derive(Clone, Copy)]
pub(crate) enum Source {
    Range,
    Digit {
        address: usize,
        shift: usize,
        mask: usize,
    },
}

struct PackSlot<'a> {
    place: Placement,
    source: Source,
    words: &'a mut [F128],
    projections: Vec<(&'a WeightedQuery<'a>, Vec<F>)>,
}

impl PackSlot<'_> {
    fn column(&mut self, col: usize, raw: &[u128], values: &mut [u128; ROWS]) -> Result<()> {
        let width = self.place.range.bits();
        let values = match self.source {
            Source::Digit { shift, mask, .. } => {
                let decoded = &mut values[..raw.len()];
                for (out, &address) in decoded.iter_mut().zip(raw) {
                    *out = 1 << ((address as usize >> shift) & mask);
                }
                decoded
            }
            Source::Range => raw,
        };
        for (word, chunk) in values.chunks(128).enumerate() {
            let mut planes = [0u128; 65];
            for (row, &value) in chunk.iter().enumerate() {
                if value >= 1 << width {
                    return Err("integer outside committed range");
                }
                let mut bits = value;
                while bits != 0 {
                    planes[bits.trailing_zeros() as usize] |= 1 << row;
                    bits &= bits - 1;
                }
            }
            for (bit, &encoded) in planes[..width].iter().enumerate() {
                self.words[(bit * self.place.columns + col) * WORDS + word] = F128::from(encoded);
            }
        }
        for (query, output) in &mut self.projections {
            let weight = query.column[col] * query.scale;
            if weight.is_zero() {
                continue;
            }
            if query.binary {
                let mut coefficients = [F::zero(); 65];
                for (out, &plane) in coefficients.iter_mut().zip(&query.planes) {
                    *out = weight * plane;
                }
                for (out, &value) in output.iter_mut().zip(values) {
                    let mut bits = value;
                    while bits != 0 {
                        *out += coefficients[bits.trailing_zeros() as usize];
                        bits &= bits - 1;
                    }
                }
            } else {
                for (out, &value) in output.iter_mut().zip(values) {
                    *out += weight.mul_integer(value as i128);
                }
            }
        }
        Ok(())
    }
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

    // The reader returns the stored unsigned code before any one-hot expansion.
    fn pack(
        &self,
        sources: &[Source],
        read: impl Fn(usize, usize) -> u128 + Sync,
        queries: &[WeightedQuery<'_>],
    ) -> Result<(Vec<F128>, Vec<(Vec<F>, Vec<F>)>)> {
        let mut packed = vec![F128::from(0u128); self.pcs.packed_len()];
        let mut words = packed.as_mut_slice();
        let mut jobs = Vec::<(usize, Vec<PackSlot<'_>>)>::new();
        let mut addresses = HashMap::new();
        let mut by_slot = vec![Vec::new(); sources.len()];
        let mut rows = HashMap::new();
        let mut pairs = Vec::<(Vec<F>, Vec<F>)>::new();
        for q in queries {
            let row = *rows.entry(&q.row).or_insert_with(|| {
                pairs.push((basis(&q.row), vec![F::zero(); ROWS]));
                pairs.len() - 1
            });
            if q.binary && q.planes.iter().all(|&x| x == F::one()) {
                // Lookup advice is one-hot. The verifier still checks the committed bits.
                let count = 1 << q.placement.range.vars.min(ROW_BITS);
                for value in &mut pairs[row].1[..count] {
                    *value += q.scale;
                }
            } else {
                by_slot[q.slot].push((q, vec![F::zero(); ROWS]));
            }
        }
        for (slot, (&place, &source)) in self.placements.iter().zip(sources).enumerate() {
            let (here, rest) = words.split_at_mut(place.range.bits() * place.columns * WORDS);
            words = rest;
            let pack = PackSlot {
                place,
                source,
                words: here,
                projections: std::mem::take(&mut by_slot[slot]),
            };
            if place.range.bits() == 0 {
                continue;
            }
            if let Source::Digit { address, .. } = source {
                let job = *addresses
                    .entry((address, place.range.vars))
                    .or_insert(jobs.len());
                if job < jobs.len() {
                    jobs[job].1.push(pack);
                    continue;
                }
            }
            jobs.push((slot, vec![pack]));
        }
        let jobs = jobs
            .into_par_iter()
            .map(|(slot, mut outputs)| -> Result<_> {
                let place = outputs[0].place;
                let count = 1 << place.range.vars.min(ROW_BITS);
                let mut raw = [0; ROWS];
                let mut values = [0; ROWS];
                for col in 0..place.columns {
                    for (row, value) in raw[..count].iter_mut().enumerate() {
                        *value = read(slot, col * ROWS + row);
                    }
                    for output in &mut outputs {
                        output.column(col, &raw[..count], &mut values)?;
                    }
                }
                Ok(outputs)
            })
            .collect::<Result<Vec<_>>>()?;
        for output in jobs.into_iter().flatten() {
            for (query, values) in output.projections {
                for (out, value) in pairs[rows[&query.row]].1.iter_mut().zip(values) {
                    *out += value;
                }
            }
        }
        Ok((packed, pairs))
    }

    pub(crate) fn commit_detached(
        &self,
        sources: &[Source],
        read: impl Fn(usize, usize) -> u128 + Sync,
    ) -> Result<Committed> {
        let (packed, _) = self.pack(sources, read, &[])?;
        self.commit_packed(packed)
    }

    fn commit_packed(&self, packed: Vec<F128>) -> Result<Committed> {
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

    pub(crate) fn reduce(
        &self,
        root: RootBytes,
        queries: &[Query],
        outer: &mut ProverState<Keccak>,
        sources: &[Source],
        read: impl Fn(usize, usize) -> u128 + Sync,
    ) -> Result<Pending<'_>> {
        let (instance, queries) = self.prepare(queries, outer)?;
        let (packed, pairs) = self.pack(sources, read, &queries)?;
        let committed = self.commit_packed(packed)?;
        if committed.root != root {
            return Err("advice commitment changed");
        }
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
            basis(&end.challenges).into_iter().map(F::scalar).collect(),
            column_weights.into_iter().map(F::scalar).collect(),
            end.output_claim.scalar(),
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
        let sources = [Source::Range; 2];
        let read =
            |slot: usize, i: usize| (data[slot][i] + group.placements[slot].range.offset()) as u128;
        let committed = group.commit_detached(&sources, read).unwrap();
        prover.append_bytes(&committed.root);
        let root = committed.root;
        let proof = group
            .reduce(root, &queries, &mut prover, &sources, read)
            .unwrap()
            .open(&mut std::collections::HashMap::new())
            .unwrap();
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
