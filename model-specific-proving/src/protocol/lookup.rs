use std::{collections::BTreeMap, sync::Arc};

use jolt_field::{One, Ring, Zero};
use jolt_poly::TensorEqTable;
use rayon::prelude::*;

use super::{
    field::F,
    opening::{Layout, Packed, Request, Slot},
    sumcheck::{Column, Session},
    tensor::{equality, inner, weights, Claim},
    Result,
};

// This limits the simultaneously materialized digit columns, in bytes.
const LOOKUP_LINK_BYTES: usize = 192 * 1024 * 1024;

pub(crate) struct Addresses {
    slots: Vec<u8>,
    bits: usize,
    len: usize,
}

impl Addresses {
    pub(crate) fn new(len: usize, bits: usize) -> Self {
        Self {
            slots: vec![0; (len * bits).div_ceil(8) + 4],
            len,
            bits,
        }
    }

    pub(crate) fn get(&self, i: usize) -> usize {
        let bit = i * self.bits;
        let offset = bit / 8;
        let mut word = [0; 8];
        word[..5].copy_from_slice(&self.slots[offset..offset + 5]);
        ((u64::from_le_bytes(word) >> (bit % 8)) & ((1 << self.bits) - 1)) as usize
    }

    pub(crate) fn set(&mut self, i: usize, value: usize) {
        let bit = i * self.bits;
        let offset = bit / 8;
        let mask = ((1u64 << self.bits) - 1) << (bit % 8);
        let mut word = [0; 8];
        word[..5].copy_from_slice(&self.slots[offset..offset + 5]);
        let word = (u64::from_le_bytes(word) & !mask) | (value as u64) << (bit % 8);
        self.slots[offset..offset + 5].copy_from_slice(&word.to_le_bytes()[..5]);
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Table<'a> {
    Direct(&'a [i32]),
    Softmax {
        sums: &'a [i32],
        limit: i32,
        radix: u128,
    },
}

impl Table<'_> {
    fn size(self) -> usize {
        match self {
            Self::Direct(values) | Self::Softmax { sums: values, .. } => values.len(),
        }
    }

    fn key(self) -> (usize, i32, u128) {
        match self {
            Self::Direct(values) => (values.as_ptr() as usize, 0, 0),
            Self::Softmax { sums, limit, radix } => (sums.as_ptr() as usize, limit, radix + 2),
        }
    }

    pub(crate) fn value(self, index: usize) -> F {
        F::from_i128(match self {
            Self::Direct(v) => i128::from(v[index]),
            Self::Softmax { sums, limit, radix } => {
                let previous = if index == 0 {
                    i128::from(limit) + 1
                } else {
                    i128::from(sums[index - 1])
                };
                i128::from(sums[index]) - radix as i128 * previous
            }
        })
    }
}

#[derive(Clone)]
pub(crate) struct Lookup {
    digits: Vec<(Slot, usize, usize)>,
    pub cells: usize,
    pub bits: usize,
}

impl Lookup {
    pub(crate) fn new(layout: &mut Layout, cells: usize, table_size: usize) -> Result<Self> {
        if table_size < 2 || !table_size.is_power_of_two() || table_size > 1 << 32 {
            return Err("invalid lookup table size");
        }
        let bits = table_size.ilog2() as usize;
        let preferred = if (17..=24).contains(&bits) { 3 } else { 4 };
        let first = (bits - 1) % preferred + 1;
        let mut shift = bits;
        let digits = std::iter::once(first)
            .chain(std::iter::repeat_n(preferred, (bits - first) / preferred))
            .map(|n| {
                shift -= n;
                (layout.reserve(cells, 1 << n), shift, n)
            })
            .collect();
        Ok(Self {
            digits,
            cells,
            bits,
        })
    }

    pub(crate) fn record(&self, packed: &mut Packed<'_>, addresses: &Addresses) -> Result<()> {
        if addresses.len != 1 << self.cells || addresses.bits != self.bits {
            return Err("invalid lookup witness");
        }
        packed.one_hot(&self.digits, addresses.len, |i| addresses.get(i));
        Ok(())
    }
}

struct Read {
    plan: Lookup,
    point: Vec<F>,
    address: F,
    outputs: Vec<F>,
    witness: Option<Arc<Addresses>>,
}

#[derive(Default)]
pub(crate) struct Lookups<'a>(Vec<(Vec<Table<'a>>, usize, Vec<Read>)>);

impl<'a> Lookups<'a> {
    pub(crate) fn read(
        &mut self,
        plan: &Lookup,
        tables: &[Table<'a>],
        point: Vec<F>,
        address: F,
        outputs: Vec<F>,
        witness: Option<Arc<Addresses>>,
    ) {
        let read = Read {
            plan: plan.clone(),
            point,
            address,
            outputs,
            witness,
        };
        if let Some((_, _, reads)) = self.0.iter_mut().find(|(current, size, _)| {
            *size == plan.cells
                && current
                    .iter()
                    .map(|t| t.key())
                    .eq(tables.iter().map(|t| t.key()))
        }) {
            reads.push(read);
        } else {
            self.0.push((tables.to_vec(), plan.cells, vec![read]));
        }
    }

    pub(crate) fn finish(
        self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
    ) -> Result<()> {
        self.finish_with_budget(io, requests, LOOKUP_LINK_BYTES)
    }

    fn finish_with_budget(
        self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        budget: usize,
    ) -> Result<()> {
        for (tables, cells, reads) in self.0 {
            let size = tables[0].size();
            let bits = size.ilog2() as usize;
            let mixing = io.random(
                reads.len() + 1 + tables.len(),
                reads.len() + 1 + tables.len(),
            )?;
            let (mix, coefficients) = mixing.split_at(reads.len());
            let table = |i| {
                coefficients[0] * F::from_u64(i as u64)
                    + inner(
                        coefficients[1..].iter().copied(),
                        tables.iter().map(|t| t.value(i)),
                    )
            };
            let sum = inner(
                mix.iter().copied(),
                reads.iter().map(|q| {
                    coefficients[0] * q.address
                        + inner(coefficients[1..].iter().copied(), q.outputs.iter().copied())
                }),
            );
            let (end, [selected]) = io.fold(
                sum,
                bits,
                2,
                || {
                    let scan = |update: &mut dyn FnMut(usize, F)| {
                        for (q, &scale) in reads.iter().zip(mix) {
                            let addresses = q.witness.as_ref().expect("lookup trace");
                            let weights = TensorEqTable::new_with_scaling(&q.point, scale);
                            let inner = weights.e_in();
                            for (block, &outer) in weights.e_out().iter().enumerate() {
                                for (i, &weight) in inner.iter().enumerate() {
                                    update(addresses.get(block * inner.len() + i), outer * weight);
                                }
                            }
                        }
                    };
                    let histogram = if size <= 1 << 19 {
                        let mut bins = vec![F::zero(); size];
                        scan(&mut |i, weight| bins[i] += weight);
                        Column::Dense(bins)
                    } else {
                        let mut bins = BTreeMap::new();
                        scan(&mut |i, weight| *bins.entry(i).or_insert_with(F::zero) += weight);
                        Column::source(size, move |i| bins.get(&i).copied().unwrap_or_else(F::zero))
                    };
                    vec![histogram, Column::source(size, table)]
                },
                [0],
                |v| v[0] * v[1],
            )?;
            io.expect(&end, || {
                selected * inner(weights(&end.challenges), (0..size).map(table))
            })?;

            let digits = reads[0].plan.digits.len();
            let selectors: Vec<_> = reads[0]
                .plan
                .digits
                .iter()
                .map(|&(_, shift, n)| {
                    let start = bits - shift - n;
                    weights(&end.challenges[start..start + n])
                })
                .collect();

            let len = 1usize << cells;
            let bytes_per_read = (len / 8)
                .saturating_mul(digits * std::mem::size_of::<F>())
                .max(1);
            let chunk_size = (budget / bytes_per_read).max(1);
            let chunks = reads.chunks(chunk_size).zip(mix.chunks(chunk_size));
            let partials = if reads.len() <= chunk_size {
                vec![selected]
            } else {
                // Bind every partial claim before any of the link challenges.
                let parts = io.fields(chunks.len(), || {
                    let address = TensorEqTable::new(&end.challenges);
                    let dense = (size <= 1 << 19).then(|| weights(&end.challenges));
                    reads
                        .par_chunks(chunk_size)
                        .zip(mix.par_chunks(chunk_size))
                        .map(|(reads, mix)| {
                            reads
                                .iter()
                                .zip(mix)
                                .map(|(read, &scale)| {
                                    let witness = read.witness.as_ref().expect("lookup trace");
                                    let cells = TensorEqTable::new_with_scaling(&read.point, scale);
                                    (0..len)
                                        .map(|i| {
                                            let addr = witness.get(i);
                                            let weight = dense.as_ref().map_or_else(
                                                || address.evaluate_index(addr),
                                                |table| table[addr],
                                            );
                                            cells.evaluate_index(i) * weight
                                        })
                                        .sum::<F>()
                                })
                                .sum()
                        })
                        .collect()
                })?;
                if parts.iter().copied().sum::<F>() != selected {
                    return Err("invalid lookup link partial sums");
                }
                parts
            };
            for ((reads, mix), partial) in chunks.zip(partials) {
                Self::link(io, requests, reads, mix, &selectors, partial)?;
            }
        }
        Ok(())
    }

    fn link(
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        reads: &[Read],
        mix: &[F],
        selectors: &[Vec<F>],
        partial: F,
    ) -> Result<()> {
        let cells = reads[0].plan.cells;
        let digits = selectors.len();
        let len = 1 << cells;
        let endpoints: Vec<_> = (0..reads.len() * (digits + 1))
            .filter(|i| i % (digits + 1) != digits)
            .collect();
        let (link, values) = io.fold_many(
            partial,
            cells,
            digits + 1,
            || {
                let mut columns = Vec::new();
                for q in reads {
                    for (choice, &(_, shift, n)) in selectors.iter().zip(&q.plan.digits) {
                        let addresses = q.witness.as_ref().expect("lookup trace");
                        columns.push(Column::lookup(len, choice, move |i| {
                            (addresses.get(i) >> shift) & ((1 << n) - 1)
                        }));
                    }
                    columns.push(Column::split_equality(&q.point));
                }
                columns
            },
            &endpoints,
            |row| {
                inner(
                    mix.iter().copied(),
                    row.chunks_exact(digits + 1)
                        .map(|part| part.iter().copied().product()),
                )
            },
        )?;

        let mut terminal = F::zero();
        for ((q, &scale), values) in reads.iter().zip(mix).zip(values.chunks_exact(digits)) {
            let mut product = scale * equality(&q.point, &link.challenges);
            for ((selection, &(slot, _, width)), &value) in
                selectors.iter().zip(&q.plan.digits).zip(values)
            {
                product *= value;
                requests.push(Request {
                    slot,
                    claim: Claim::new(link.challenges.clone(), value),
                    planes: selection.clone(),
                    one_hot_sum: false,
                });
                requests.push(Request {
                    slot,
                    claim: Claim::new(q.point.clone(), F::one()),
                    planes: vec![F::one(); 1 << width],
                    one_hot_sum: true,
                });
            }
            terminal += product;
        }
        io.expect(&link, || terminal)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifies_lookup_link_split_across_multiple_chunks() {
        let table = [3, 5, 7, 11];
        let mut layout = Layout::default();
        let reads: Vec<_> = (0..3)
            .map(|_| Lookup::new(&mut layout, 3, table.len()).unwrap())
            .collect();
        let mut packed = layout.packed();
        let point = [2, 3, 4].map(F::from_u64);
        let points = weights(&point);
        let witnesses: Vec<_> = (0..reads.len())
            .map(|j| {
                let mut witness = Addresses::new(8, 2);
                for i in 0..8 {
                    witness.set(i, ((i + j) ^ (i >> 1)) & 3);
                }
                Arc::new(witness)
            })
            .collect();
        for (read, witness) in reads.iter().zip(&witnesses) {
            read.record(&mut packed, witness).unwrap();
        }

        let check = |io: &mut Session<'_, '_>| {
            let mut lookups = Lookups::default();
            for (read, witness) in reads.iter().zip(&witnesses) {
                let indices = (0..8).map(|i| witness.get(i));
                lookups.read(
                    read,
                    &[Table::Direct(&table)],
                    point.to_vec(),
                    inner(
                        points.iter().copied(),
                        indices.clone().map(|i| F::from_u64(i as u64)),
                    ),
                    vec![inner(
                        points.iter().copied(),
                        indices.map(|i| F::from_i32(table[i])),
                    )],
                    io.proving().then(|| witness.clone()),
                );
            }
            let mut requests = Vec::new();
            lookups.finish_with_budget(io, &mut requests, 16)?;
            assert_ne!(requests[0].claim.point, requests[2].claim.point);
            Ok(requests)
        };
        let domain = [7; 32];
        let plan = layout.finish(&packed.widths).unwrap();
        let proof = plan.prove(&domain, packed, |_| {}, check).unwrap();
        plan.verify(&domain, &proof, |_| {}, check).unwrap();
    }
}
