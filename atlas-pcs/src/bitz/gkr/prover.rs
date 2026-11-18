//! The grand-product prover.
//!
//! The four layers nearest the leaves are never stored: a node there is a product of at most
//! eight adjacent rows of one column, so it is read from a table indexed by the node and the
//! column's bits under it. A sumcheck on such a layer runs its first rounds on those tables and
//! stores the layer once enough row variables are bound for it to be a sixteenth of the bit
//! count. Layers above are stored.
//!
//! Everything is laid out for sequential passes. The bits are copied once into bytes ordered
//! by row, then column, so a pass over one byte position reads one table row and the columns'
//! bytes in order. A stored layer holds, for each row index, its first child's run over the
//! columns and then its second child's, so binding a row variable combines adjacent runs, and
//! the `eq` weight of an entry is the product of a row weight and a column weight, never a
//! table of its own.
#![allow(clippy::indexing_slicing)]

use super::{draw_combine, draw_output_point, round_pow_bits, LayerProof, Proof};
use crate::bitz::gf128::{
    add_products, fold_into_low, inverses, mul_pairs, product_sums, scale_pairs, Gf128, WideSum,
};
use crate::bitz::multilinear::{eq_table, fold_first};
use atlas_transcript::Transcript;
use rayon::prelude::*;

/// Layers within this many levels of the leaves are read from tables.
const TABLE_DEPTH: usize = 4;

/// A table round takes the columns this many at a time, so the children it accumulates for
/// them stay in the first-level cache.
const COLUMN_RUN: usize = 256;

/// A table round sums each column's products over [`REST_RUN`] row indices for this many
/// columns at a time: the unreduced sums of a tile stay in the second-level cache.
const COLUMN_TILE: usize = 4096;
const REST_RUN: usize = 64;

/// The live columns of the bit matrix as bytes: the byte at `position · columns + i` holds
/// the eight rows `8·position ..` of the `i`-th live column, low row in the low bit. A column
/// with no bit set is not live: every node of its product tree is one, so it is never read.
struct ByteMatrix {
    bytes: Vec<u8>,
    log_rows: usize,
    columns: usize,
}

impl ByteMatrix {
    /// From the word-packed bits, bit `(row, column)` at index `row + 2^log_rows · column`,
    /// keeping the columns `live`.
    fn new(words: &[u64], log_rows: usize, live: &[usize]) -> Self {
        let columns = live.len();
        let words_per_column = (1usize << log_rows) / 64;
        let mut bytes = vec![0u8; words_per_column * 8 * columns];
        bytes
            .par_chunks_mut(8 * columns)
            .enumerate()
            .for_each(|(word_index, rows)| {
                for (at, &column) in live.iter().enumerate() {
                    let word = words[column * words_per_column + word_index];
                    for (k, byte) in word.to_le_bytes().into_iter().enumerate() {
                        rows[k * columns + at] = byte;
                    }
                }
            });
        Self {
            bytes,
            log_rows,
            columns,
        }
    }

    fn columns(&self) -> usize {
        self.columns
    }

    /// The number of byte positions in a column.
    fn positions(&self) -> usize {
        (1 << self.log_rows) / 8
    }

    /// Every live column's byte at `position`.
    fn row(&self, position: usize) -> &[u8] {
        &self.bytes[position * self.columns..(position + 1) * self.columns]
    }
}

/// For the layer `depth` levels above the leaves, the value of each node under each pattern of
/// its `2^depth` bits: entry `node << 2^depth | pattern` is the product of the factors of the
/// rows whose bit is set.
fn node_table(factors: &[Gf128], depth: usize) -> Vec<Gf128> {
    let leaves = 1usize << depth;
    let patterns = 1usize << leaves;
    factors
        .par_chunks(leaves)
        .flat_map_iter(|rows| {
            let mut values = vec![Gf128::ONE; patterns];
            for pattern in 1..patterns {
                let low = pattern.trailing_zeros() as usize;
                values[pattern] = values[pattern & (pattern - 1)] * rows[low];
            }
            values
        })
        .collect()
}

/// The two sums a round computes: `t(0)` and the leading coefficient of the quadratic `t`.
/// `t(1)` follows from the running claim.
type RoundSums = (Gf128, Gf128);

const NO_SUMS: RoundSums = (Gf128::ZERO, Gf128::ZERO);

fn add_sums(a: RoundSums, b: RoundSums) -> RoundSums {
    (a.0 + b.0, a.1 + b.1)
}

fn scale_sums(weight: Gf128, sums: RoundSums) -> RoundSums {
    (weight * sums.0, weight * sums.1)
}

/// The two products of a term, from the two factors at the round variable's two values.
#[inline]
fn term(p: [Gf128; 2], q: [Gf128; 2]) -> RoundSums {
    (p[0] * q[0], (p[0] + p[1]) * (q[0] + q[1]))
}

/// The running state of one sumcheck of `scalar · eq(point, ·) · P · Q`.
struct Sumcheck<'t, T: Transcript> {
    claim: Gf128,
    /// `eq` of the point's coordinates already bound, at their challenges.
    scalar: Gf128,
    rounds: Vec<[Gf128; 3]>,
    challenges: Vec<Gf128>,
    target_bits: u32,
    transcript: &'t mut T,
}

impl<'t, T: Transcript> Sumcheck<'t, T> {
    fn new(claim: Gf128, target_bits: u32, transcript: &'t mut T) -> Self {
        Self {
            claim,
            scalar: Gf128::ONE,
            rounds: Vec::new(),
            challenges: Vec::new(),
            target_bits,
            transcript,
        }
    }

    /// Sends a round given the sums of the quadratic
    /// `t(X) = Σ_x eq(rest, x)·P(X, x)·Q(X, x)`, for the variable whose `eq` coordinate is
    /// `coordinate`. The round polynomial is `h = scalar · eq(coordinate, X) · t(X)`, with
    /// `eq(coordinate, X) = 1 + coordinate + X`, so `h(0) + h(1) = claim` gives `t(1)` from
    /// `t(0)`: `scalar · coordinate · t(1) = claim + scalar · (1 + coordinate) · t(0)`.
    fn round(&mut self, coordinate: Gf128, (at_zero, leading): RoundSums) -> Gf128 {
        let shifted = self.scalar * (Gf128::ONE + coordinate);
        #[allow(clippy::expect_used)]
        let divisor = (self.scalar * coordinate)
            .inverse()
            .expect("the point's challenges are nonzero");
        let at_one = (self.claim + shifted * at_zero) * divisor;
        let linear = at_zero + at_one + leading;
        let message = [
            shifted * at_zero,
            shifted * leading + self.scalar * linear,
            self.scalar * leading,
        ];
        let [constant, quadratic, cubic] = message;
        let message_linear = shifted * linear + self.scalar * at_zero;
        debug_assert_eq!(
            message_linear + quadratic + cubic,
            self.claim,
            "h(0) + h(1) is the claim"
        );
        self.transcript.append_serializable(&message);
        #[allow(clippy::expect_used)]
        self.transcript
            .pow(round_pow_bits(self.target_bits))
            .expect("the prover's proof of work cannot fail");
        let r = Gf128(self.transcript.challenge_u128());
        self.claim = constant + r * (message_linear + r * (quadratic + r * cubic));
        self.scalar *= Gf128::ONE + coordinate + r;
        self.rounds.push(message);
        self.challenges.push(r);
        r
    }

    /// Runs the rounds of the row variables on a stored layer: for each row index, the run of
    /// `P` over the live columns, then the run of `Q`. A round pairs two adjacent row indices
    /// and folds the pair into the first of the two, so the layer is folded where it lies:
    /// after `j` rounds the rows left are every `2^j`-th run. The first pass gives every `P`
    /// its column's weight, which the rounds, linear in `P`, carry: no term multiplies by a
    /// weight, and the weights are divided out of what is returned.
    ///
    /// `idle` is the weight of the columns that are not live: their `P` and `Q` are one, so
    /// they add `idle` to every round's `t(0)` and nothing to its leading coefficient.
    /// Returns `P` and `Q` over the live columns alone.
    fn bind_rows(
        &mut self,
        layer: &mut [Gf128],
        row_point: &[Gf128],
        column_weights: &[Gf128],
        idle: Gf128,
    ) -> (Vec<Gf128>, Vec<Gf128>) {
        let columns = column_weights.len();
        let run = 2 * columns;
        debug_assert_eq!(layer.len(), run << row_point.len());
        let children = |row: &[Gf128]| (row[..columns].to_vec(), row[columns..run].to_vec());
        let Some((&first, later)) = row_point.split_first() else {
            return children(layer);
        };
        // The sums of a pair of rows, each `[P, Q]`.
        let pair_sums = |low: &[Gf128], high: &[Gf128]| -> RoundSums {
            let ((p0, q0), (p1, q1)) = (low.split_at(columns), high.split_at(columns));
            let [at_zero, leading] = product_sums(p0, p1, q0, q1);
            (at_zero, leading)
        };
        #[allow(clippy::expect_used)]
        let unweight = inverses(column_weights).expect("the point's challenges are not bits");
        let mut coordinate = first;
        let mut sums = eq_table(later)
            .par_iter()
            .zip(layer.par_chunks_exact_mut(2 * run))
            .map(|(&weight, pair)| {
                let (low, high) = pair.split_at_mut(run);
                scale_pairs(&mut low[..columns], column_weights);
                scale_pairs(&mut high[..columns], column_weights);
                scale_sums(weight, pair_sums(low, high))
            })
            .reduce(|| NO_SUMS, add_sums);
        sums.0 += idle;
        let mut rest = later;
        // The rows left are the runs at multiples of `stride`.
        let mut stride = 1usize;
        loop {
            let r = self.round(coordinate, sums);
            let fold = |pair: &mut [Gf128]| {
                let (low, high) = pair.split_at_mut(stride * run);
                fold_into_low(&mut low[..run], &high[..run], r);
            };
            let Some((&next, later)) = rest.split_first() else {
                fold(layer);
                let (mut p, q) = children(layer);
                scale_pairs(&mut p, &unweight);
                return (p, q);
            };
            // Fold two pairs and take the next round's sums of the result in one pass.
            sums = eq_table(later)
                .par_iter()
                .zip(layer.par_chunks_exact_mut(4 * stride * run))
                .map(|(&weight, pairs)| {
                    let (low, high) = pairs.split_at_mut(2 * stride * run);
                    fold(low);
                    fold(high);
                    scale_sums(weight, pair_sums(&low[..run], &high[..run]))
                })
                .reduce(|| NO_SUMS, add_sums);
            sums.0 += idle;
            stride *= 2;
            coordinate = next;
            rest = later;
        }
    }

    /// Runs the remaining rounds on stored `P` and `Q` whose variables, first to last, have
    /// the `eq` coordinates `point`. Returns the values the two end on.
    fn finish(&mut self, mut p: Vec<Gf128>, mut q: Vec<Gf128>, point: &[Gf128]) -> (Gf128, Gf128) {
        debug_assert_eq!(p.len(), 1 << point.len());
        for (j, &coordinate) in point.iter().enumerate() {
            let rest = eq_table(&point[j + 1..]);
            let sums = rest
                .par_iter()
                .zip(p.par_chunks_exact(2).zip(q.par_chunks_exact(2)))
                .map(|(&w, (p, q))| scale_sums(w, term([p[0], p[1]], [q[0], q[1]])))
                .reduce(|| NO_SUMS, add_sums);
            let r = self.round(coordinate, sums);
            p = fold_first(&p, r);
            q = fold_first(&q, r);
        }
        (p[0], q[0])
    }
}

/// A layer read from tables, during a sumcheck with `bound` of its row variables fixed.
///
/// A node of the layer is indexed by its child bit, then the bound row variables, then the
/// rest. With the bound variables summed out against `eq` of their challenges, the two
/// children at a remaining row index `rest` and round variable `τ` are sums over a block of
/// `2^(depth + bound + 2)` adjacent rows of one column, the half of the block with the low
/// rows belonging to `τ = 0`.
struct TableLayer<'a> {
    bytes: &'a ByteMatrix,
    depth: usize,
    bound: usize,
    /// The layer's node table with each node's entries scaled by `eq` of the bound
    /// challenges at the node's bound bits.
    scaled: Vec<Gf128>,
}

impl<'a> TableLayer<'a> {
    fn new(bytes: &'a ByteMatrix, table: &[Gf128], depth: usize, challenges: &[Gf128]) -> Self {
        let patterns = 1usize << (1 << depth);
        let bound = challenges.len();
        let weights = eq_table(challenges);
        let mask = weights.len() - 1;
        let scaled = table
            .par_chunks(patterns)
            .enumerate()
            .flat_map_iter(|(node, values)| {
                let weight = weights[(node >> 1) & mask];
                values.iter().map(move |&value| weight * value)
            })
            .collect();
        Self {
            bytes,
            depth,
            bound,
            scaled,
        }
    }

    /// `log2` of the rows in a round's block.
    fn log_block(&self) -> usize {
        self.depth + self.bound + 2
    }

    /// For every byte position and every value of the byte, the sum of the byte's nodes of
    /// each child: entry `position << 8 | value` holds `[child 0, child 1]`. A node lies
    /// inside one byte since `depth ≤ 3`.
    /// Written over `table`, whose buffer the rounds of a proof share.
    fn byte_sums(&self, table: &mut Vec<[Gf128; 2]>) {
        let nodes_per_byte = 8usize >> self.depth;
        let pattern_bits = 1usize << self.depth;
        let pattern_mask = (1usize << pattern_bits) - 1;
        table.resize(self.bytes.positions() << 8, [Gf128::ZERO; 2]);
        table.par_iter_mut().enumerate().for_each(|(index, entry)| {
            let (position, value) = (index >> 8, index & 0xff);
            let mut sums = [Gf128::ZERO; 2];
            for j in 0..nodes_per_byte {
                let node = position * nodes_per_byte + j;
                let pattern = (value >> (j * pattern_bits)) & pattern_mask;
                sums[node & 1] += self.scaled[node << pattern_bits | pattern];
            }
            *entry = sums;
        });
    }

    /// A round whose block is at most a byte: the two products of a term depend only on
    /// the block's bits, so the column weights are summed per block and pattern before any
    /// multiplication.
    fn bucketed_round(&self, rest_weights: &[Gf128], column_weights: &[Gf128]) -> RoundSums {
        let block_bits = 1usize << self.log_block();
        let patterns = 1usize << block_bits;
        let blocks_per_byte = 8 / block_bits;
        let pattern_bits = 1usize << self.depth;
        let pattern_mask = (1usize << pattern_bits) - 1;
        (0..self.bytes.positions())
            .into_par_iter()
            .map(|position| {
                let mut buckets = vec![Gf128::ZERO; blocks_per_byte * patterns];
                for (&value, &weight) in self.bytes.row(position).iter().zip(column_weights) {
                    for block in 0..blocks_per_byte {
                        let pattern = (value as usize >> (block * block_bits)) & (patterns - 1);
                        buckets[block * patterns + pattern] += weight;
                    }
                }
                let mut sums = NO_SUMS;
                for block in 0..blocks_per_byte {
                    let rest = position * blocks_per_byte + block;
                    let mut block_sums = NO_SUMS;
                    for pattern in 0..patterns {
                        let bucket = buckets[block * patterns + pattern];
                        if bucket.is_zero() {
                            continue;
                        }
                        // The children `[τ = 0, τ = 1]` of each child bit under this pattern.
                        let mut children = [[Gf128::ZERO; 2]; 2];
                        for local in 0..1usize << (self.bound + 2) {
                            let node = rest << (self.bound + 2) | local;
                            let sub = (pattern >> (local * pattern_bits)) & pattern_mask;
                            children[local & 1][local >> (self.bound + 1)] +=
                                self.scaled[node << pattern_bits | sub];
                        }
                        let products = term(children[0], children[1]);
                        block_sums = add_sums(block_sums, scale_sums(bucket, products));
                    }
                    sums = add_sums(sums, scale_sums(rest_weights[rest], block_sums));
                }
                sums
            })
            .reduce(|| NO_SUMS, add_sums)
    }

    /// A round whose half-block is whole bytes: each child is a sum of per-byte table
    /// entries, accumulated for every column one byte position at a time.
    ///
    /// The first child's entries carry their row index's weight, so a term is two products
    /// with no weight in them. The products of a column are summed unreduced over a run of
    /// row indices and meet the column's weight once per run.
    fn byte_round(
        &self,
        rest_weights: &[Gf128],
        column_weights: &[Gf128],
        sums_table: &mut Vec<[Gf128; 2]>,
    ) -> RoundSums {
        let bytes_per_half = 1usize << (self.log_block() - 4);
        self.byte_sums(sums_table);
        sums_table
            .par_chunks_mut((2 * bytes_per_half) << 8)
            .zip(rest_weights)
            .for_each(|(entries, &weight)| entries.iter_mut().for_each(|entry| entry[0] *= weight));
        let sums_table = &*sums_table;
        let rests: Vec<usize> = (0..rest_weights.len()).collect();
        column_weights
            .par_chunks(COLUMN_TILE)
            .enumerate()
            .flat_map(|(tile, weights)| {
                rests
                    .par_chunks(REST_RUN)
                    .map(move |rests| (tile * COLUMN_TILE, weights, rests))
            })
            .map(|(first_column, weights, rests)| {
                let mut products = vec![[WideSum::ZERO; 2]; weights.len()];
                // The children at the round variable's two values, `P` then `Q`.
                let mut children = [[Gf128::ZERO; COLUMN_RUN]; 4];
                for &rest in rests {
                    for (run, products) in products.chunks_mut(COLUMN_RUN).enumerate() {
                        let (start, len) = (first_column + run * COLUMN_RUN, products.len());
                        children
                            .iter_mut()
                            .for_each(|child| child.fill(Gf128::ZERO));
                        let [p0, p1, q0, q1] = &mut children;
                        for (tau, (p, q)) in [(&mut *p0, &mut *q0), (&mut *p1, &mut *q1)]
                            .into_iter()
                            .enumerate()
                        {
                            for k in 0..bytes_per_half {
                                let position = (2 * rest + tau) * bytes_per_half + k;
                                let entries = &sums_table[position << 8..(position + 1) << 8];
                                let values = &self.bytes.row(position)[start..start + len];
                                for ((p, q), &value) in p.iter_mut().zip(q.iter_mut()).zip(values) {
                                    let entry = entries[value as usize];
                                    *p += entry[0];
                                    *q += entry[1];
                                }
                            }
                        }
                        add_products(products, &p0[..len], &p1[..len], &q0[..len], &q1[..len]);
                    }
                }
                products
                    .iter()
                    .zip(weights)
                    .fold(NO_SUMS, |acc, (sums, &weight)| {
                        add_sums(
                            acc,
                            scale_sums(weight, (sums[0].reduce(), sums[1].reduce())),
                        )
                    })
            })
            .reduce(|| NO_SUMS, add_sums)
    }

    /// The round's two sums for the next row variable, whose remaining row indices have the
    /// weights `rest_weights`.
    fn round(
        &self,
        rest_weights: &[Gf128],
        column_weights: &[Gf128],
        sums_table: &mut Vec<[Gf128; 2]>,
    ) -> RoundSums {
        if self.log_block() <= 3 {
            self.bucketed_round(rest_weights, column_weights)
        } else {
            self.byte_round(rest_weights, column_weights, sums_table)
        }
    }

    /// The two children with the bound variables fixed, stored at the front of `arena`: for
    /// each remaining row index, the run of the first child over the columns, then the run of
    /// the second. A node's rows are then `2^(depth + bound + 1)`, whole bytes. Returns the
    /// length written; the arena grows only if it is shorter.
    fn materialize(
        &self,
        rest_vars: usize,
        arena: &mut Vec<Gf128>,
        sums_table: &mut Vec<[Gf128; 2]>,
    ) -> usize {
        self.byte_sums(sums_table);
        let sums_table = &*sums_table;
        let bytes_per_node = 1usize << (self.depth + self.bound + 1 - 3);
        let columns = self.bytes.columns();
        let len = (2 * columns) << rest_vars;
        if arena.len() < len {
            arena.resize(len, Gf128::ZERO);
        }
        arena[..len]
            .par_chunks_mut(2 * columns)
            .enumerate()
            .for_each(|(rest, children)| {
                let (low, high) = children.split_at_mut(columns);
                for k in 0..bytes_per_node {
                    let position = rest * bytes_per_node + k;
                    let entries = &sums_table[position << 8..(position + 1) << 8];
                    let values = self.bytes.row(position);
                    for ((p, q), &value) in low.iter_mut().zip(high.iter_mut()).zip(values) {
                        let entry = entries[value as usize];
                        // The first byte overwrites what the arena held.
                        if k == 0 {
                            (*p, *q) = (entry[0], entry[1]);
                        } else {
                            *p += entry[0];
                            *q += entry[1];
                        }
                    }
                }
            });
        len
    }
}

/// Proves the columns' products for row factors `factors`. Returns the proof, the products,
/// and the point and value of the claim left on the multilinear extension of the bits.
pub fn prove<T: Transcript>(
    words: &[u64],
    factors: &[Gf128],
    log_columns: usize,
    target_bits: u32,
    transcript: &mut T,
) -> (Proof, Vec<Gf128>, Vec<Gf128>, Gf128) {
    let log_rows = factors.len().trailing_zeros() as usize;
    assert_eq!(factors.len(), 1 << log_rows, "a power of two rows");
    assert!(
        log_rows >= TABLE_DEPTH + 3,
        "a column is whole bytes of whole table nodes"
    );
    assert_eq!(words.len() * 64, 1 << (log_rows + log_columns), "bit count");
    // The columns with a bit set. With none, the first column stands in: its tree is all ones
    // like any other's.
    let words_per_column = (1usize << log_rows) / 64;
    let mut live: Vec<usize> = (0..1usize << log_columns)
        .into_par_iter()
        .filter(|&column| {
            words[column * words_per_column..(column + 1) * words_per_column]
                .iter()
                .any(|&word| word != 0)
        })
        .collect();
    if live.is_empty() {
        live.push(0);
    }
    let live = &live;
    let columns = live.len();
    // `values` over the live columns as a vector over all columns, one elsewhere.
    let spread = |values: &[Gf128]| -> Vec<Gf128> {
        let mut all = vec![Gf128::ONE; 1 << log_columns];
        for (&column, &value) in live.iter().zip(values) {
            all[column] = value;
        }
        all
    };
    // The weights of the live columns, and the summed weight of the rest: an `eq` table sums
    // to one.
    let live_weights = |column_point: &[Gf128]| -> (Vec<Gf128>, Gf128) {
        let all = eq_table(column_point);
        let weights: Vec<Gf128> = live.iter().map(|&column| all[column]).collect();
        let idle = weights.iter().fold(Gf128::ONE, |acc, &w| acc + w);
        (weights, idle)
    };
    let bytes = ByteMatrix::new(words, log_rows, live);

    // The stored layers, `stored[i] = L_i` for `i ≤ log_rows − TABLE_DEPTH`, each indexed by
    // column, then node.
    let span = tracing::info_span!("gkr_stored_layers").entered();
    let top = log_rows - TABLE_DEPTH;
    let byte_table = node_table(factors, 3);
    let mut stored: Vec<Vec<Gf128>> = vec![Vec::new(); top + 1];
    let mut top_layer = Gf128::zeros(columns << top);
    top_layer
        .par_chunks_mut(columns)
        .enumerate()
        .for_each(|(node, values)| {
            let (low, high) = (2 * node, 2 * node + 1);
            let low_values = &byte_table[low << 8..(low + 1) << 8];
            let high_values = &byte_table[high << 8..(high + 1) << 8];
            for ((value, &a), &b) in values.iter_mut().zip(bytes.row(low)).zip(bytes.row(high)) {
                *value = low_values[a as usize] * high_values[b as usize];
            }
        });
    stored[top] = top_layer;
    for i in (0..top).rev() {
        let mut above = Gf128::zeros(columns << i);
        above
            .par_chunks_mut(columns)
            .zip(stored[i + 1].par_chunks_exact(2 * columns))
            .for_each(|(out, pair)| {
                let (low, high) = pair.split_at(columns);
                mul_pairs(out, low, high);
            });
        stored[i] = above;
    }
    let outputs = spread(&stored[0]);
    drop(span);

    #[allow(clippy::expect_used)]
    let mut point = draw_output_point(log_columns, target_bits, transcript)
        .expect("the prover's proof of work cannot fail");
    let mut claim = eq_table(&point)
        .iter()
        .zip(&outputs)
        .fold(Gf128::ZERO, |acc, (&w, &y)| acc + w * y);

    let mut layers = Vec::with_capacity(log_rows);
    // The buffer the table layers are stored in (the largest stored layer's, once that layer
    // is folded), and the one their per-byte tables are built in.
    let mut arena: Vec<Gf128> = Vec::new();
    let mut sums_table: Vec<[Gf128; 2]> = Vec::new();
    for row_vars in 0..log_rows {
        // The layer below has `row_vars` row variables after its child bit.
        let depth = log_rows - row_vars - 1;
        let _span = tracing::info_span!("gkr_layer", depth).entered();
        let (row_point, column_point) = point.split_at(row_vars);
        let (column_weights, idle) = live_weights(column_point);
        let mut sumcheck = Sumcheck::new(claim, target_bits, transcript);
        let (p, q) = if depth >= TABLE_DEPTH {
            // A stored layer is its two children's runs, alternating.
            let mut below = std::mem::take(&mut stored[row_vars + 1]);
            let children = sumcheck.bind_rows(&mut below, row_point, &column_weights, idle);
            if below.len() > arena.len() {
                arena = below;
            }
            children
        } else {
            let table = node_table(factors, depth);
            // Rounds on the tables until the layer is a sixteenth of the bit count: the size
            // of the largest stored layer, whose buffer it is then stored in.
            let streamed = (TABLE_DEPTH - depth).min(row_vars);
            for bound in 0..streamed {
                let layer = TableLayer::new(&bytes, &table, depth, &sumcheck.challenges);
                let rest_weights = eq_table(&row_point[bound + 1..]);
                let (at_zero, leading) =
                    layer.round(&rest_weights, &column_weights, &mut sums_table);
                sumcheck.round(row_point[bound], (at_zero + idle, leading));
            }
            let layer = TableLayer::new(&bytes, &table, depth, &sumcheck.challenges);
            let len = layer.materialize(row_vars - streamed, &mut arena, &mut sums_table);
            sumcheck.bind_rows(
                &mut arena[..len],
                &row_point[streamed..],
                &column_weights,
                idle,
            )
        };
        let (low, high) = sumcheck.finish(spread(&p), spread(&q), column_point);
        let Sumcheck {
            rounds, challenges, ..
        } = sumcheck;
        let children = [low, high];
        transcript.append_serializable(&children);
        #[allow(clippy::expect_used)]
        let combine =
            draw_combine(target_bits, transcript).expect("the prover's proof of work cannot fail");
        claim = low + combine * (low + high);
        point = std::iter::once(combine).chain(challenges).collect();
        layers.push(LayerProof { rounds, children });
    }

    // The leaves: 1 + (y_b + 1)·bit(b, c). Fix the columns at the point and run a sumcheck
    // over the rows of eq · (y + 1) · bits(·, column point).
    let _span = tracing::info_span!("gkr_leaves").entered();
    let (row_point, column_point) = point.split_at(log_rows);
    let (column_weights, _) = live_weights(column_point);
    let bits_at_columns: Vec<Gf128> = (0..bytes.positions())
        .into_par_iter()
        .flat_map_iter(|position| {
            // The column weights summed per byte value, then per row of the byte.
            let mut buckets = [Gf128::ZERO; 256];
            for (&value, &weight) in bytes.row(position).iter().zip(&column_weights) {
                buckets[value as usize] += weight;
            }
            let mut rows = [Gf128::ZERO; 8];
            for (value, &bucket) in buckets.iter().enumerate() {
                for (row, sum) in rows.iter_mut().enumerate() {
                    if (value >> row) & 1 == 1 {
                        *sum += bucket;
                    }
                }
            }
            rows
        })
        .collect();
    let factors_plus_one: Vec<Gf128> = factors.par_iter().map(|&y| y + Gf128::ONE).collect();
    let mut sumcheck = Sumcheck::new(claim + Gf128::ONE, target_bits, transcript);
    let (_, bits_value) = sumcheck.finish(factors_plus_one, bits_at_columns, row_point);
    let Sumcheck {
        rounds: leaf_rounds,
        challenges,
        ..
    } = sumcheck;
    transcript.append_serializable(&bits_value);
    let bits_point = challenges.iter().chain(column_point).copied().collect();
    (
        Proof {
            layers,
            leaf_rounds,
            bits_value,
        },
        outputs,
        bits_point,
        bits_value,
    )
}
