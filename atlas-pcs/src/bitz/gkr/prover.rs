//! The grand-product prover.
//!
//! The four layers nearest the leaves are never stored: a node there is a product of at most
//! eight adjacent rows of one column, so it is read from a table indexed by the node and the
//! column's bits under it. A sumcheck on such a layer runs its first rounds on those tables and
//! stores the layer only once enough row variables are bound for it to be a sixteenth of the
//! bit count. Layers above are stored.
#![allow(clippy::indexing_slicing)]

use super::{draw_combine, draw_output_point, round_pow_bits, LayerProof, Proof};
use crate::bitz::gf128::Gf128;
use crate::bitz::multilinear::{eq_table, fold_first};
use atlas_transcript::Transcript;
use rayon::prelude::*;

/// Layers within this many levels of the leaves are read from tables.
const TABLE_DEPTH: usize = 4;

/// The bit matrix: bit `(row, column)` at index `row + 2^log_rows · column`.
struct Bits<'a> {
    words: &'a [u64],
    log_rows: usize,
}

impl Bits<'_> {
    /// The `2^depth` bits of `column` under node `node` of the layer `depth` levels above the
    /// leaves, as an integer.
    #[inline]
    fn pattern(&self, column: usize, node: usize, depth: usize) -> usize {
        let start = (column << self.log_rows) + (node << depth);
        let mask = (1usize << (1 << depth)) - 1;
        (self.words[start / 64] >> (start % 64)) as usize & mask
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

    /// Sends a round given `t(0)`, `t(1)` and the leading coefficient of the quadratic
    /// `t(X) = Σ_x eq(rest, x)·P(X, x)·Q(X, x)`, for the variable whose `eq` coordinate is
    /// `coordinate`. The round polynomial is `scalar · eq(coordinate, X) · t(X)`, and
    /// `eq(coordinate, X) = 1 + coordinate + X`.
    fn round(&mut self, coordinate: Gf128, at_zero: Gf128, at_one: Gf128, leading: Gf128) -> Gf128 {
        let linear = at_zero + at_one + leading;
        let shifted = self.scalar * (Gf128::ONE + coordinate);
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

    /// Runs the remaining rounds on stored `P` and `Q`, whose variables have the `eq`
    /// coordinates `point`. Returns the values the two end on.
    fn finish(&mut self, mut p: Vec<Gf128>, mut q: Vec<Gf128>, point: &[Gf128]) -> (Gf128, Gf128) {
        debug_assert_eq!(p.len(), 1 << point.len());
        for (j, &coordinate) in point.iter().enumerate() {
            let rest = eq_table(&point[j + 1..]);
            let (at_zero, at_one, leading) = rest
                .par_iter()
                .zip(p.par_chunks_exact(2).zip(q.par_chunks_exact(2)))
                .map(|(&w, (p, q))| {
                    (
                        w * (p[0] * q[0]),
                        w * (p[1] * q[1]),
                        w * ((p[0] + p[1]) * (q[0] + q[1])),
                    )
                })
                .reduce(
                    || (Gf128::ZERO, Gf128::ZERO, Gf128::ZERO),
                    |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
                );
            let r = self.round(coordinate, at_zero, at_one, leading);
            p = fold_first(&p, r);
            q = fold_first(&q, r);
        }
        (p[0], q[0])
    }
}

/// A layer read from tables, during a sumcheck with `bound` row variables fixed.
struct TableLayer<'a> {
    bits: &'a Bits<'a>,
    depth: usize,
    /// The layer's node table with each node's entries scaled by `eq` of the bound
    /// challenges at the node's bound bits.
    scaled: Vec<Gf128>,
    bound: usize,
}

impl<'a> TableLayer<'a> {
    fn new(bits: &'a Bits<'a>, table: &[Gf128], depth: usize, challenges: &[Gf128]) -> Self {
        let patterns = 1usize << (1 << depth);
        let bound = challenges.len();
        let weights = eq_table(challenges);
        let mask = weights.len() - 1;
        // Node index: child bit, then the bound row variables, then the rest.
        let scaled = table
            .par_chunks(patterns)
            .enumerate()
            .flat_map_iter(|(node, values)| {
                let weight = weights[(node >> 1) & mask];
                values.iter().map(move |&value| weight * value)
            })
            .collect();
        Self {
            bits,
            depth,
            scaled,
            bound,
        }
    }

    /// The child `child` of the layer, with its bound variables fixed, at the remaining row
    /// index `rest` of `column`: the sum over the bound variables' cube.
    #[inline]
    fn value(&self, column: usize, rest: usize, child: usize) -> Gf128 {
        let patterns_log = 1 << self.depth;
        let mut acc = Gf128::ZERO;
        for beta in 0..1usize << self.bound {
            let node = rest << (self.bound + 1) | beta << 1 | child;
            let pattern = self.bits.pattern(column, node, self.depth);
            acc += self.scaled[node << patterns_log | pattern];
        }
        acc
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
    let bits = Bits { words, log_rows };
    let columns = 1usize << log_columns;

    // The stored layers, `stored[i] = L_i` for `i ≤ log_rows − TABLE_DEPTH`, each indexed by
    // node, then column.
    let top = log_rows - TABLE_DEPTH;
    let byte_table = node_table(factors, 3);
    let mut stored: Vec<Vec<Gf128>> = vec![Vec::new(); top + 1];
    stored[top] = (0..columns << top)
        .into_par_iter()
        .map(|index| {
            let (node, column) = (index & ((1 << top) - 1), index >> top);
            let low = byte_table[(2 * node) << 8 | bits.pattern(column, 2 * node, 3)];
            let high = byte_table[(2 * node + 1) << 8 | bits.pattern(column, 2 * node + 1, 3)];
            low * high
        })
        .collect();
    for i in (0..top).rev() {
        stored[i] = stored[i + 1]
            .par_chunks_exact(2)
            .map(|pair| pair[0] * pair[1])
            .collect();
    }
    let outputs = stored[0].clone();

    #[allow(clippy::expect_used)]
    let mut point = draw_output_point(log_columns, target_bits, transcript)
        .expect("the prover's proof of work cannot fail");
    let mut claim = eq_table(&point)
        .iter()
        .zip(&outputs)
        .fold(Gf128::ZERO, |acc, (&w, &y)| acc + w * y);

    let mut layers = Vec::with_capacity(log_rows);
    for row_vars in 0..log_rows {
        // The layer below has `row_vars` row variables after its child bit.
        let depth = log_rows - row_vars - 1;
        let mut sumcheck = Sumcheck::new(claim, target_bits, transcript);
        let (low, high) = if depth >= TABLE_DEPTH {
            let below = std::mem::take(&mut stored[row_vars + 1]);
            let p = below.par_iter().step_by(2).copied().collect();
            let q = below.par_iter().skip(1).step_by(2).copied().collect();
            sumcheck.finish(p, q, &point)
        } else {
            let table = node_table(factors, depth);
            let streamed = TABLE_DEPTH - depth;
            let (row_point, column_point) = point.split_at(row_vars);
            let column_weights = eq_table(column_point);
            for bound in 0..streamed {
                let layer = TableLayer::new(&bits, &table, depth, &sumcheck.challenges);
                let rest_weights = eq_table(&row_point[bound + 1..]);
                let (at_zero, at_one, leading) = (0..columns)
                    .into_par_iter()
                    .map(|column| {
                        let mut acc = (Gf128::ZERO, Gf128::ZERO, Gf128::ZERO);
                        for (rest, &w) in rest_weights.iter().enumerate() {
                            let p = [
                                layer.value(column, 2 * rest, 0),
                                layer.value(column, 2 * rest + 1, 0),
                            ];
                            let q = [
                                layer.value(column, 2 * rest, 1),
                                layer.value(column, 2 * rest + 1, 1),
                            ];
                            acc.0 += w * (p[0] * q[0]);
                            acc.1 += w * (p[1] * q[1]);
                            acc.2 += w * ((p[0] + p[1]) * (q[0] + q[1]));
                        }
                        let w = column_weights[column];
                        (w * acc.0, w * acc.1, w * acc.2)
                    })
                    .reduce(
                        || (Gf128::ZERO, Gf128::ZERO, Gf128::ZERO),
                        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
                    );
                sumcheck.round(row_point[bound], at_zero, at_one, leading);
            }
            let layer = TableLayer::new(&bits, &table, depth, &sumcheck.challenges);
            let rest_vars = row_vars - streamed;
            let materialize = |child: usize| -> Vec<Gf128> {
                (0..columns << rest_vars)
                    .into_par_iter()
                    .map(|index| {
                        layer.value(index >> rest_vars, index & ((1 << rest_vars) - 1), child)
                    })
                    .collect()
            };
            let (p, q) = (materialize(0), materialize(1));
            sumcheck.finish(p, q, &point[streamed..])
        };
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
    let (row_point, column_point) = point.split_at(log_rows);
    let column_weights = eq_table(column_point);
    let row_words = (1usize << log_rows) / 64;
    let bits_at_columns = words
        .par_chunks(row_words)
        .zip(column_weights.par_iter())
        .fold(
            || vec![Gf128::ZERO; 1 << log_rows],
            |mut acc, (column, &w)| {
                for (i, &word) in column.iter().enumerate() {
                    let mut rest = word;
                    while rest != 0 {
                        acc[64 * i + rest.trailing_zeros() as usize] += w;
                        rest &= rest - 1;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![Gf128::ZERO; 1 << log_rows],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += *y;
                }
                a
            },
        );
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
