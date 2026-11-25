use common::{BitMatrix, BitTable, SecurityLevel};
use field::{F128, Wide256};
use num_traits::{ConstOne, ConstZero};
use rayon::prelude::*;
use std::ops::Range;
use transcript::{ProverState, VerifierState};

mod bounded;
mod packed_round;

pub type Field = F128;

const CUBIC_GRINDING_LABEL: &[u8] = b"gkr/cubic/v1";

/// Proves the layer-by-layer sumcheck reduction from a claim at `point`
/// (an evaluation point on the output layer) down to a claim on the leaves.
///
/// Points are little-endian, as in `poly`'s tables: `point[i]` is the
/// coordinate of index bit `i`. Every layer adds one coordinate, at the top.
///
/// # Panics
///
/// If `point` does not have a coordinate per tree bit: the output layer has
/// one value per tree.
#[must_use]
#[tracing::instrument(name = "Prove GKR", skip_all)]
pub fn gpgkr_prove(
    ps: &mut ProverState,
    point: &[Field],
    // Retained upper layers and the packed source for the remaining layers.
    witnesses: LayerWitnesses,
    security: SecurityLevel,
) -> (Vec<Field>, Field) {
    assert_eq!(
        point.len(),
        witnesses.lut_layers.log_trees,
        "the output has one value per tree, so the point has a coordinate per tree bit"
    );
    if witnesses.lut_layers.nodes::<LAYER1>() > bounded::DENSE_NODES {
        return bounded::prove(
            ps,
            point.to_vec(),
            witnesses,
            security,
            bounded::DENSE_NODES,
        );
    }
    let mut point = point.to_vec();
    let LayerWitnesses {
        lut_layers,
        upper_nodes,
        ..
    } = witnesses;
    // The leaves' sumcheck has the largest point, a coordinate per index bit
    // of its parent, layer 1.
    let mut eq_memory = EqTables::alloc_storage(lut_layers.nodes::<LAYER1>());
    for children in upper_nodes.into_iter().rev() {
        (point, _) = prove_layer(ps, &mut eq_memory, &point, children, security);
    }

    // Layers 2, 1 and 0 stay in their lookup tables. Round 1 of each writes the
    // children it has bound, `[left′ | right′]`, into one allocation sized
    // for the leaves': round 1 halves them to as many as layer 1 has nodes.
    let mut bound_children = Field::zeroed_vec(lut_layers.nodes::<LAYER1>());
    let (point, _) = prove_lut_layer::<LAYER2>(
        ps,
        &mut eq_memory,
        &point,
        &lut_layers,
        &mut bound_children,
        security,
    );
    let (point, _) = prove_lut_layer::<LAYER1>(
        ps,
        &mut eq_memory,
        &point,
        &lut_layers,
        &mut bound_children,
        security,
    );
    prove_lut_layer::<LEAVES>(
        ps,
        &mut eq_memory,
        &point,
        &lut_layers,
        &mut bound_children,
        security,
    )
}

/// Proves `c = Σ_x eq(z, x) · L(x) · R(x)` for the halves `L`, `R` of
/// `children`, one round per coordinate of `point`, top first. Round `i`
/// sends `sum_i(b)` and its leading coefficient `sum_i(∞)`, where `b` is the
/// boolean point that [`sends_one`] picks, `s_i(X) = eq(z_i, X) · sum_i(X)`,
/// and `sum_i` carries `eq_bound = Π_{j > i} eq(z_j, r_j)` from the rounds
/// already bound. Returns the point `(r, t)` on `children` and the claim
/// `W(r, t) = L(r) + t · (R(r) - L(r))`.
fn prove_layer(
    ps: &mut ProverState,
    eq_memory: &mut [Field],
    point: &[Field],
    mut children: Vec<Field>,
    security: SecurityLevel,
) -> (Vec<Field>, Field) {
    let eq_tables = EqTables::new(eq_memory, point);
    // The top index bit of `children` picks the half: `L`, then `R`.
    let mid = children.len() / 2;
    let (left, right) = children.split_at_mut(mid);
    prove_rounds(ps, &eq_tables, point, None, left, right, security)
}

/// The dense rounds of a layer, then its last message. A LUT-backed layer
/// has already proved round 1, which binds the top coordinate of `point`, and
/// passes that round's challenge as `round_one`; its `left` and `right` are
/// then the bound halves. A dense layer passes `None` and proves every round
/// here.
fn prove_rounds(
    ps: &mut ProverState,
    eq_tables: &EqTables,
    point: &[Field],
    round_one: Option<Field>,
    mut left: &mut [Field],
    mut right: &mut [Field],
    security: SecurityLevel,
) -> (Vec<Field>, Field) {
    let n = point.len();
    let mut next_point = vec![Field::ZERO; n + 1];
    // The coordinates whose rounds run here, and `eq` of those already bound.
    let (remaining, mut eq_bound) = match round_one {
        Some(r) => {
            let top = n - 1;
            next_point[top] = r;
            (&point[..top], eq_factor(r, point[top]))
        }
        None => (point, Field::ONE),
    };

    for (i, &z) in remaining.iter().enumerate().rev() {
        // `x_i = 0` in the low half of each, `x_i = 1` in the high half.
        let half = 1 << i;
        let (left0, left1) = left.split_at_mut(half);
        let (right0, right1) = right.split_at_mut(half);

        // Branch once per round: `round_sums` takes `b` as a const generic,
        // so the hot fold has no per-element check.
        let eq = eq_tables.below(i);
        let (sum_b, sum_inf) = if sends_one(z) {
            round_sums::<true>(left0, left1, right0, right1, eq)
        } else {
            round_sums::<false>(left0, left1, right0, right1, eq)
        };
        ps.prover_message(&[eq_bound * sum_b.reduce(), eq_bound * sum_inf.reduce()]);

        ps.grind(CUBIC_GRINDING_LABEL, security.grinding_bits(3));
        let r = ps.verifier_message();
        bind(r, left0, left1, right0, right1);
        left = &mut left[..half];
        right = &mut right[..half];

        next_point[i] = r;
        eq_bound *= eq_factor(r, z);
    }

    let (left_r, right_r) = (left[0], right[0]);
    ps.prover_message(&[left_r, right_r]);
    let t = ps.verifier_message();
    next_point[n] = t;
    (next_point, interpolate(left_r, right_r, t))
}

/// `Σ eq · L(b) · R(b)` and `Σ eq · ΔL · ΔR` over the free variables: the
/// round's sum at the boolean point `b` (`1` if `AT_ONE`) and its leading
/// coefficient, without `eq_bound` and left unreduced.
fn round_sums<const AT_ONE: bool>(
    left0: &[Field],
    left1: &[Field],
    right0: &[Field],
    right1: &[Field],
    eq: &[Field],
) -> (Wide256, Wide256) {
    let at = |v0: Field, v1: Field| if AT_ONE { v1 } else { v0 };

    // The last round has a single summand, and `eq` there is the empty
    // product `1`: no multiplication by it, and too few for two lanes.
    if let [_] = eq {
        let (l0, l1, r0, r1) = (left0[0], left1[0], right0[0], right1[0]);
        return (
            Wide256::mul(at(l0, l1), at(r0, r1)),
            Wide256::mul(l1 - l0, r1 - r0),
        );
    }

    // Two lanes per step: independent multiplication chains. Chunking also
    // reduces variance in benchmarking.
    left0
        .par_chunks_exact(2)
        .zip(left1.par_chunks_exact(2))
        .zip(right0.par_chunks_exact(2))
        .zip(right1.par_chunks_exact(2))
        .zip(eq.par_chunks_exact(2))
        // Chunked by 2, so the element-count threshold below is halved in
        // terms of chunks.
        .with_min_len(PARALLEL_MIN_LANES / 2)
        .fold(
            || (Wide256::zero(), Wide256::zero()),
            |(mut sum_b, mut sum_inf), ((((l0, l1), r0), r1), e)| {
                // `e · L(b) · R(b)` into `sum_b` and `e · ΔL · ΔR` into
                // `sum_inf`, with `e` joining the left factor through a reduced
                // multiply so that the right one can stay wide.
                //
                // Steps, products, then sums, each for both lanes: this order
                // puts every load of the step ahead of the first multiply.
                // Lane by lane is 1.1-1.4% slower single-threaded.
                let (d_l0, d_r0) = (l1[0] - l0[0], r1[0] - r0[0]);
                let (d_l1, d_r1) = (l1[1] - l0[1], r1[1] - r0[1]);
                let (a0, b0) = (e[0] * at(l0[0], l1[0]), e[0] * d_l0);
                let (a1, b1) = (e[1] * at(l0[1], l1[1]), e[1] * d_l1);
                sum_b += Wide256::mul(a0, at(r0[0], r1[0]));
                sum_inf += Wide256::mul(b0, d_r0);
                sum_b += Wide256::mul(a1, at(r0[1], r1[1]));
                sum_inf += Wide256::mul(b1, d_r1);
                (sum_b, sum_inf)
            },
        )
        .reduce(
            || (Wide256::zero(), Wide256::zero()),
            |(x_b, x_inf), (y_b, y_inf)| (x_b + y_b, x_inf + y_inf),
        )
}

/// Binds the variable that splits each pair of halves to `r`, writing the
/// result over the low halves.
fn bind(r: Field, left0: &mut [Field], left1: &[Field], right0: &mut [Field], right1: &[Field]) {
    left0
        .par_iter_mut()
        .zip(right0.par_iter_mut())
        .zip(left1.par_iter())
        .zip(right1.par_iter())
        .with_min_len(PARALLEL_MIN_LANES)
        .for_each(|(((l0, r0), &l1), &r1)| {
            // Recalculating the step rather than writing it in the high half
            // in `round_sums` to save on writes. No benchmarking has been done
            // to check the difference.
            let (d_l, d_r) = (l1 - *l0, r1 - *r0);

            // `interpolate`, spelled out: as `*r0 = interpolate(*r0, r1, r)`
            // the compiler loads `*r0` as a vector and moves it back to
            // general registers for the subtraction, one extra instruction
            // per pair on the way into the multiply (~3.7% single-threaded).
            *l0 += r * d_l;
            *r0 += r * d_r;
        });
}

/// The LUT-backed layers, by their index in the tree. Each names the
/// sumcheck whose children it is.
const LEAVES: usize = 0;
const LAYER1: usize = 1;
const LAYER2: usize = 2;

/// Where round 1 of a LUT-backed sumcheck reads its operands `l0, l1, r0,
/// r1`, by the layer of its children: operand `o` at position `k` is nibble
/// `k + STREAM[o] · stride`, masked by `MASK[o]`, in that nibble's lookup
/// table. [`LutLayers::pairs`] says why.
const STREAM: [[usize; 4]; 3] = [[0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 2, 3]];
const MASK: [[usize; 4]; 3] = [
    [0b0001, 0b0010, 0b0100, 0b1000],
    [0b0101, 0b0101, 0b1010, 0b1010],
    [0b1111; 4],
];

/// Layers 0 to 2 in their lookup tables, as round 1 of their sumchecks reads
/// them: layer-2 node `k = (g << s) | c` is nibble `k` of the matrix, looked
/// up in the lookup table of node position `g`, which all `2^s` trees share.
struct LutLayers {
    /// The nibbles of the layer-2 nodes; see [`LutLayers::new`].
    matrix: BitMatrix<'static>,
    luts: Vec<[Field; LUT_ENTRIES]>,
    pairs: Vec<[Field; LUT_ENTRIES * LUT_ENTRIES]>,
    /// `s`: a lookup table's run of nibbles holds one per tree.
    log_trees: usize,
}

impl LutLayers {
    fn new(matrix: BitMatrix<'_>, images: &[Field]) -> Self {
        // Leaf `(b << s) | c` is `images[b]` where `bit(c, b)` is set and one
        // elsewhere: the low `s` index bits pick one of the `2^s` trees and
        // the rest are the leaf's position in it. Each layer folds the top
        // position bit, so tree `c` multiplies the images `bit(c, ·)` selects.
        //
        // Layers 0 to 2 are looked up in 16-entry lookup tables, one per four
        // images, and the four leaves under a layer-2 node are a quarter of
        // the positions apart, so the top two position bits are split off and
        // moved below the tree bits. After the transpose, layer-2 node
        // position `g` is one contiguous run, a nibble per tree in ascending
        // order, and nibble `c` of it holds `bit(c, g + q·n)` at bit `q`, `n`
        // being a quarter of the images: the four leaves under layer-2 node
        // `(g << s) | c`.
        let log_dim2 = matrix.log_dim2() - 2;
        let matrix = matrix.reshape(log_dim2).transpose();

        let log_trees = (matrix.dim2() / NIBBLE_BITS).ilog2() as usize;
        // `pairs` reads whole words of one lookup table's run.
        assert!(
            1 << log_trees >= NIBBLES_IN_WORD,
            "each lookup table needs at least a word of nibbles"
        );
        let luts = lower_luts(images);
        let (left, right) = luts.split_at(luts.len() / 2);
        let pairs = left
            .iter()
            .zip(right)
            .map(|(left, right)| {
                std::array::from_fn(|i| left[i % LUT_ENTRIES] * right[i / LUT_ENTRIES])
            })
            .collect();
        Self {
            matrix,
            luts,
            pairs,
            log_trees,
        }
    }

    /// The nibbles, a word at a time.
    fn words(&self) -> &[u64] {
        // Read as `u64`s: on a little-endian target, bit `i` of the words is
        // bit `i` of the `u64`s too.
        const { assert!(cfg!(target_endian = "little")) };
        bytemuck::cast_slice(&self.matrix[..])
    }

    /// The nodes of layer `LAYER`: one per nibble in layer 2, twice as many
    /// in each layer below it.
    fn nodes<const LAYER: usize>(&self) -> usize {
        const { assert!(LAYER <= LAYER2, "layers 0 to 2 are the LUT-backed ones") };
        (self.words().len() * NIBBLES_IN_WORD) << (LAYER2 - LAYER)
    }

    /// The positions of round 1 handled as one piece: a lookup table's run,
    /// or a part of it, so that a piece never spans two lookup tables.
    fn piece(&self) -> usize {
        (1 << self.log_trees).min(PARALLEL_MIN_LANES)
    }

    /// Round 1's positions `0..half` in pieces, as a parallel iterator
    /// taking at least `PARALLEL_MIN_LANES` positions per task, whatever the
    /// number of trees.
    fn pieces(&self, half: usize) -> impl IndexedParallelIterator<Item = Range<usize>> + use<> {
        let piece = self.piece();
        (0..half / piece)
            .into_par_iter()
            .with_min_len(PARALLEL_MIN_LANES / piece)
            .map(move |p| p * piece..(p + 1) * piece)
    }

    /// The pairs that round 1 binds at `positions`, which lie in one lookup
    /// table's run and start and end on a word: per position, the nibbles of
    /// its operands `l0, l1, r0, r1`, unmasked, and each operand's lookup
    /// table.
    ///
    /// Round 1 binds the top bit of the parent's index, so each node pairs
    /// with its partner `stride` nodes later, where `stride` is the number of
    /// round-1 positions, half the parent's nodes:
    /// - Layer 2: a nibble is one node. The left pair is nibbles `k` and
    ///   `k + stride`, and the right pair the two half the layer on,
    ///   `k + 2·stride` and `k + 3·stride`.
    /// - Layer 1: a nibble holds a left node, bits 0 and 2, and a right node,
    ///   bits 1 and 3. Both of their partners sit in nibble `k + stride`.
    /// - Leaves: a nibble holds both pairs, bits 0 and 1 for the left child
    ///   and bits 2 and 3 for the right, so every operand reads nibble `k`.
    ///
    /// The words each operand reads are zipped, and every tuple of words
    /// gives its 16 positions, one item per word. Callers zip the words with
    /// 16-position chunks of their own data and loop over the positions
    /// inside: flattened here and zipped per position, the loop becomes a
    /// per-position state machine that LLVM does not unroll.
    fn pairs<const LAYER: usize>(
        &self,
        positions: Range<usize>,
    ) -> (
        [&[Field; LUT_ENTRIES]; 4],
        impl Iterator<Item = impl Iterator<Item = [usize; 4]>> + '_,
    ) {
        // Half the parent's nodes, which are half the layer's.
        let stride = self.nodes::<LAYER>() / 4;
        let (words, len) = (self.words(), positions.len() / NIBBLES_IN_WORD);
        let at = STREAM[LAYER].map(|q| positions.start + q * stride);
        let luts = at.map(|k| &self.luts[k >> self.log_trees]);
        let [l0, l1, r0, r1] = at.map(|k| &words[k / NIBBLES_IN_WORD..][..len]);
        let pairs = l0
            .iter()
            .zip(l1)
            .zip(r0)
            .zip(r1)
            .map(|(((&l0, &l1), &r0), &r1)| {
                (0..NIBBLES_IN_WORD).map(move |i| {
                    [l0, l1, r0, r1].map(|word| (word >> (NIBBLE_BITS * i)) as usize & 0xF)
                })
            });
        (luts, pairs)
    }
}

/// The operands `[l0, l1, r0, r1]` of one position of round 1 of layer
/// `LAYER`, from their lookup tables and nibbles.
fn operands<const LAYER: usize>(
    luts: &[&[Field; LUT_ENTRIES]; 4],
    nibbles: [usize; 4],
) -> [Field; 4] {
    std::array::from_fn(|o| luts[o][nibbles[o] & MASK[LAYER][o]])
}

/// Proves the layer whose children are layer `LAYER` of the lookup tables,
/// like [`prove_layer`]. Round 1 reads its operands from the lookup tables
/// and writes only the children it has bound, `[left′ | right′]`, into
/// `bound_children`; the rounds after it are dense.
fn prove_lut_layer<const LAYER: usize>(
    ps: &mut ProverState,
    eq_memory: &mut [Field],
    point: &[Field],
    lut_layers: &LutLayers,
    bound_children: &mut [Field],
    security: SecurityLevel,
) -> (Vec<Field>, Field) {
    let eq_tables = EqTables::new(eq_memory, point);
    let top = point.len() - 1;

    // Nothing is bound before round 1, so its sums go out as they are.
    let (sum_b, sum_inf) = if sends_one(point[top]) {
        round_one_sums::<LAYER, true>(lut_layers, &eq_tables, point)
    } else {
        round_one_sums::<LAYER, false>(lut_layers, &eq_tables, point)
    };
    ps.prover_message(&[sum_b.reduce(), sum_inf.reduce()]);

    ps.grind(CUBIC_GRINDING_LABEL, security.grinding_bits(3));
    let r = ps.verifier_message();
    let half = 1 << top;
    let (left, right) = bound_children[..2 * half].split_at_mut(half);
    lut_bind::<LAYER>(r, lut_layers, left, right);
    prove_rounds(ps, &eq_tables, point, Some(r), left, right, security)
}

/// Round 1's sum at the boolean point `b` (`1` if `AT_ONE`) and its leading
/// coefficient, unreduced, as [`round_sums`] gives them. The leaves and
/// layer 1 sum their eq weights in buckets. Layer 2 pairs whole nibbles of
/// two lookup tables, 256 keys per product, which costs more to finish than
/// a piece saves at small `s`, so it works node by node.
fn round_one_sums<const LAYER: usize, const AT_ONE: bool>(
    lut_layers: &LutLayers,
    eq_tables: &EqTables,
    point: &[Field],
) -> (Wide256, Wide256) {
    match LAYER {
        LEAVES => bucket_sums(lut_layers, eq_tables, point, leaf_buckets::<AT_ONE>),
        LAYER1 => bucket_sums(lut_layers, eq_tables, point, layer1_buckets::<AT_ONE>),
        LAYER2 => layer2_sums::<AT_ONE>(lut_layers, eq_tables.below(point.len() - 1)),
        _ => unreachable!("layers 0 to 2 are the LUT-backed ones"),
    }
}

/// Round 1's sums a piece at a time, with `buckets` summing each piece's eq
/// weights by the nibble bits that `L·R` reads and multiplying each bucket
/// once. The weight splits as `eq(z_{<top}, k) = eq_hi[g] · eq_lo[c]` for
/// `k = (g << s) | c`, so the buckets sum `eq_lo`, which stays in cache, and
/// `eq_hi` joins once per piece.
fn bucket_sums(
    lut_layers: &LutLayers,
    eq_tables: &EqTables,
    point: &[Field],
    buckets: impl Fn(&LutLayers, Range<usize>, &[Field]) -> (Wide256, Wide256) + Sync,
) -> (Wide256, Wide256) {
    let top = point.len() - 1;
    let (half, s) = (1 << top, lut_layers.log_trees);
    let eq_lo = eq_tables.below(s);
    let eq_hi = poly::eq_table(&point[s..top]);
    lut_layers
        .pieces(half)
        .map(|positions| {
            let h = eq_hi[positions.start >> s];
            let eq_lo = &eq_lo[positions.start % (1 << s)..][..positions.len()];
            let (sum_b, sum_inf) = buckets(lut_layers, positions, eq_lo);
            (
                Wide256::mul(h, sum_b.reduce()),
                Wide256::mul(h, sum_inf.reduce()),
            )
        })
        .reduce(
            || (Wide256::zero(), Wide256::zero()),
            |(x_b, x_inf), (y_b, y_inf)| (x_b + y_b, x_inf + y_inf),
        )
}

/// The piece of round 1 for the leaves at `positions`, its weights in `eq`.
/// Every operand is one bit of the same nibble, so both sums are functions of
/// the nibble: each position adds its weight into its nibble's bucket, and
/// each of the 16 buckets is multiplied once.
fn leaf_buckets<const AT_ONE: bool>(
    lut_layers: &LutLayers,
    positions: Range<usize>,
    eq: &[Field],
) -> (Wide256, Wide256) {
    let ([lut, ..], pairs) = lut_layers.pairs::<LEAVES>(positions);
    let mut bucket = [Field::ZERO; LUT_ENTRIES];
    for (word, eq) in pairs.zip(eq.chunks_exact(NIBBLES_IN_WORD)) {
        for ([a, ..], &e) in word.zip(eq) {
            bucket[a] += e;
        }
    }
    // `L(b)·R(b)` is the node of bits 0 and 2, or 1 and 3, and `L(∞)·R(∞)` is
    // `(l1 − l0)(r1 − r0)`.
    let b_mask = if AT_ONE { 0b1010 } else { 0b0101 };
    let at_b = std::array::from_fn(|a| lut[a & b_mask]);
    let at_inf = std::array::from_fn(|a| {
        (lut[a & 0b0010] - lut[a & 0b0001]) * (lut[a & 0b1000] - lut[a & 0b0100])
    });
    (dot(&bucket, &at_b), dot(&bucket, &at_inf))
}

/// The piece of round 1 for layer 1 at `positions`, its weights in `eq`:
/// nibble `n0` and the nibble `n1` half the layer later. In characteristic 2,
/// `(l1 − l0)(r1 − r0)` is `l0·r0 + l1·r1 + l1·r0 + l0·r1`, and each product
/// reads two disjoint 2-bit masks: 16 buckets each, keyed by the bits it
/// reads.
fn layer1_buckets<const AT_ONE: bool>(
    lut_layers: &LutLayers,
    positions: Range<usize>,
    eq: &[Field],
) -> (Wide256, Wide256) {
    // `l0` and `r0` read nibble `n0` in its lookup table, `l1` and `r1`
    // nibble `n1`.
    let ([lut0, lut1, ..], pairs) = lut_layers.pairs::<LAYER1>(positions);
    let mut bucket = [[Field::ZERO; LUT_ENTRIES]; 4];
    // A position's keys in the buckets of `l0·r0`, `l1·r1`, `l1·r0` and
    // `l0·r1`: bits 0 and 2 from the nibble of the `l` factor, bits 1 and 3
    // from that of the `r` factor.
    for (word, eq) in pairs.zip(eq.chunks_exact(NIBBLES_IN_WORD)) {
        for ([n0, n1, ..], &e) in word.zip(eq) {
            bucket[0][n0] += e;
            bucket[1][n1] += e;
            bucket[2][(n1 & 0b0101) | (n0 & 0b1010)] += e;
            bucket[3][(n0 & 0b0101) | (n1 & 0b1010)] += e;
        }
    }
    // `l0·r0`, `l1·r1`, `l1·r0` and `l0·r1` at every key.
    let product: [[Field; LUT_ENTRIES]; 4] = [
        *lut0,
        *lut1,
        std::array::from_fn(|x| lut1[x & 0b0101] * lut0[x & 0b1010]),
        std::array::from_fn(|x| lut0[x & 0b0101] * lut1[x & 0b1010]),
    ];
    // Product `b` is `l_b·r_b`.
    let b = usize::from(AT_ONE);
    let sum_inf = (0..4).fold(Wide256::zero(), |acc, j| acc + dot(&bucket[j], &product[j]));
    (dot(&bucket[b], &product[b]), sum_inf)
}

/// `Σ bucket · value`, unreduced.
fn dot(bucket: &[Field; LUT_ENTRIES], value: &[Field; LUT_ENTRIES]) -> Wide256 {
    bucket
        .iter()
        .zip(value)
        .fold(Wide256::zero(), |acc, (&b, &v)| acc + Wide256::mul(b, v))
}

/// Round 1's sums for layer 2, node by node, as [`round_sums`] forms them,
/// with the operands read from the lookup tables.
fn layer2_sums<const AT_ONE: bool>(lut_layers: &LutLayers, eq: &[Field]) -> (Wide256, Wide256) {
    let at = |v0: Field, v1: Field| if AT_ONE { v1 } else { v0 };
    let half = eq.len();
    lut_layers
        .pieces(half)
        .map(|positions| {
            let (luts, pairs) = lut_layers.pairs::<LAYER2>(positions.clone());
            let (mut sum_b, mut sum_inf) = (Wide256::zero(), Wide256::zero());
            for (word, eq) in pairs.zip(eq[positions].chunks_exact(NIBBLES_IN_WORD)) {
                for (n, &e) in word.zip(eq) {
                    let [l0, l1, r0, r1] = operands::<LAYER2>(&luts, n);
                    sum_b += Wide256::mul(e * at(l0, l1), at(r0, r1));
                    sum_inf += Wide256::mul(e * (l1 - l0), r1 - r0);
                }
            }
            (sum_b, sum_inf)
        })
        .reduce(
            || (Wide256::zero(), Wide256::zero()),
            |(x_b, x_inf), (y_b, y_inf)| (x_b + y_b, x_inf + y_inf),
        )
}

/// Round 1's bind, `left′[k] = l0 + r·(l1 − l0)` and
/// `right′[k] = r0 + r·(r1 − r0)`, with the operands read from the lookup
/// tables.
///
/// What position `k` combines depends on the layer; see [`LutLayers::pairs`]:
/// - Layer 2: four nibbles, one node each, the left pair at `k` and
///   `k + stride` and the right pair half the layer on.
/// - Layer 1: two nibbles, `k` and `k + stride`, each holding a left and a
///   right node, so one scan binds both without jumping ahead.
/// - Leaves: one nibble, which holds both pairs.
fn lut_bind<const LAYER: usize>(
    r: Field,
    lut_layers: &LutLayers,
    left: &mut [Field],
    right: &mut [Field],
) {
    let (half, piece) = (left.len(), lut_layers.piece());
    left.par_chunks_mut(piece)
        .zip(right.par_chunks_mut(piece))
        .zip(lut_layers.pieces(half))
        .for_each(|((left, right), positions)| {
            let (luts, pairs) = lut_layers.pairs::<LAYER>(positions);
            let left = left.chunks_exact_mut(NIBBLES_IN_WORD);
            let right = right.chunks_exact_mut(NIBBLES_IN_WORD);
            for ((left, right), word) in left.zip(right).zip(pairs) {
                for ((l, rt), n) in left.iter_mut().zip(right).zip(word) {
                    let [l0, l1, r0, r1] = operands::<LAYER>(&luts, n);
                    (*l, *rt) = (interpolate(l0, l1, r), interpolate(r0, r1, r));
                }
            }
        });
}

/// The line through `at0` (at `0`) and `at1` (at `1`), at `r`: a multilinear
/// function of one variable.
fn interpolate(at0: Field, at1: Field, r: Field) -> Field {
    // Narrow multiply, as the next round will multiply with this result.
    at0 + r * (at1 - at0)
}

/// The quadratic through `at0` (at `0`) and `at1` (at `1`) with leading
/// coefficient `at_inf`, at `r`: `at0 + r·(at1 - at0) + r·(r - 1)·at_inf`, with
/// `r` factored out.
fn quadratic_at(at0: Field, at1: Field, at_inf: Field, r: Field) -> Field {
    at0 + r * ((at1 - at0) + (r - Field::ONE) * at_inf)
}

/// Whether round `i` sends `sum_i(1)` rather than `sum_i(0)`. The verifier
/// recovers the other one from `c = (1 - z_i) · sum_i(0) + z_i · sum_i(1)`,
/// which divides by `z_i` to get `sum_i(1)`; at `z_i = 0` it gives
/// `sum_i(0) = c` directly instead.
fn sends_one(z: Field) -> bool {
    z == Field::ZERO
}

/// `eq(r, z) = r*z + (1 - r)*(1 - z)`. Expanding gives
/// `1 + r + z + 2*r*z`, and in characteristic 2 `2*r*z = r*z + r*z = 0`, so
/// this is just `1 + r + z` -- no multiplication at all, and so nothing for
/// widemul to help with.
fn eq_factor(r: Field, z: Field) -> Field {
    Field::ONE + r + z
}

/// `eq(z_{<i}, ·)` for every round `i` of a layer, in one buffer.
///
/// The former `SuffixTable` with the point little-endian instead of reversed:
/// the summed coordinates are the prefix `z[..i]`, not a suffix. Same entries.
///
/// Keeping every table takes twice the memory of the largest one. Keeping
/// only the largest and halving it between rounds (`low[k] += high[k]`)
/// would save that half, but it measured 7.5-12% slower, as it adds a pass
/// over memory. Splitting the table makes the trade moot.
// TODO: Split the table. The buckets already do a kind of splitting
struct EqTables<'a> {
    storage: &'a [Field],
}

impl<'a> EqTables<'a> {
    /// `eq(z_{<i}, x)` for `x ∈ {0,1}^i`, in the order of
    /// `poly::eq_table(&z[..i])`. Table `i` sits at
    /// `2^i - 1 .. 2^(i+1) - 1`.
    fn below(&self, i: usize) -> &[Field] {
        let start = (1 << i) - 1;
        let end = (1 << (i + 1)) - 1;
        &self.storage[start..end]
    }

    /// Builds every table of `point` in `storage`, a buffer from
    /// [`EqTables::alloc_storage`]; nothing is allocated here.
    // Inlining or not doesn't improve performance, but it does hide it in
    // profiling therefore it has been annotated with inline never.
    #[inline(never)]
    fn new(storage: &'a mut [Field], point: &[Field]) -> EqTables<'a> {
        // Round `i` reads only the coordinates below `i`, so the top one
        // never enters a table.
        let (_, below_top) = point
            .split_last()
            .expect("a sumcheck has a round per coordinate, so at least one");
        // Establishes the whole table's space budget up front so LLVM can
        // prove the `low`/`hi` writes in the loop below are in bounds
        let needed = (1usize << point.len()) - 1;
        assert!(storage.len() >= needed);

        storage[0] = Field::ONE;
        let (mut prev, mut remaining) = storage.split_at_mut(1);

        // Each coordinate doubles the table, as its new top index bit.
        for &z in below_top {
            let size = prev.len() << 1;
            let (entry, next) = remaining.split_at_mut(size);
            let (low, hi) = entry.split_at_mut(entry.len() >> 1);

            for ((l, h), &e) in low.iter_mut().zip(hi.iter_mut()).zip(prev.iter()) {
                let tmp = z * e;
                // (1-z)*e, z*e
                (*l, *h) = (e - tmp, tmp)
            }

            prev = entry;
            remaining = next;
        }
        EqTables { storage }
    }

    /// Room for every table of a sumcheck whose parent layer has up to
    /// `nodes` nodes, allocated once for all layers. Its point has a
    /// coordinate per index bit of the parent, so the tables take
    /// `nodes - 1` entries. All the tables in one buffer take as much space
    /// as a double buffer would.
    fn alloc_storage(nodes: usize) -> Vec<Field> {
        Field::zeroed_vec(nodes - 1)
    }
}

// TODO replace this with a proper approach
/// Matches `poly`'s `PARALLEL_FOLD_THRESHOLD`, chosen for the same per-round
/// MLE-fold shape.
const PARALLEL_MIN_LANES: usize = 1 << 12;

/// Verifies what [`gpgkr_prove`] proves: `layers` layer sumchecks, from
/// `claim`, the output's MLE at `point`, down to a claim on the leaves.
///
/// Returns the leaves' point, `point.len() + layers` coordinates, and the
/// claim on it; `None` if a check fails or the proof runs short.
#[must_use]
#[tracing::instrument(name = "Verify GKR", skip_all)]
pub fn gpgkr_verify(
    vs: &mut VerifierState,
    mut claim: Field,
    point: &[Field],
    layers: usize,
    security: SecurityLevel,
) -> Option<(Vec<Field>, Field)> {
    let mut point = point.to_owned();
    for _ in 0..layers {
        (point, claim) = verify_layer(vs, claim, &point, security)?
    }
    Some((point, claim))
}

/// Per round, recovers the sum at the other boolean point from
/// `c = (1 - z_i) · sum_i(0) + z_i · sum_i(1)` and moves the claim to
/// `s_i(r_i)`.
/// Then checks `c = eq(z, r) · L(r) · R(r)` and continues at
/// `W(r, t) = L(r) + t · (R(r) - L(r))`.
fn verify_layer(
    vs: &mut VerifierState,
    mut claim: Field,
    point: &[Field],
    security: SecurityLevel,
) -> Option<(Vec<Field>, Field)> {
    let n = point.len();
    let mut next_point = vec![Field::ZERO; n + 1];
    let mut eq_bound = Field::ONE;

    for (i, &z) in point.iter().enumerate().rev() {
        let [sum_b, sum_inf]: [Field; 2] = vs.prover_message().ok()?;
        // `claim = (1 - z) * sum0 + z * sum1`.
        let (sum0, sum1) = if sends_one(z) {
            (claim, sum_b)
        } else {
            (sum_b, (claim - (Field::ONE - z) * sum_b) / z)
        };

        vs.grind(CUBIC_GRINDING_LABEL, security.grinding_bits(3))
            .ok()?;
        let r = vs.verifier_message();
        next_point[i] = r;
        let eq_r = eq_factor(r, z);
        claim = eq_r * quadratic_at(sum0, sum1, sum_inf, r);
        eq_bound *= eq_r;
    }

    // Every coordinate is bound now: `eq_bound = eq(z, r)`.
    let [left_r, right_r]: [Field; 2] = vs.prover_message().ok()?;
    if eq_bound * left_r * right_r != claim {
        return None;
    }
    let t = vs.verifier_message();
    next_point[n] = t;
    Some((next_point, interpolate(left_r, right_r, t)))
}

/// Returns the output layer, the product of each column of the table, and
/// the witnesses of the layers below it.
///
/// Layers 0 to 2 stay as lookup tables, layer 3 is the first one built in
/// full, and the layers above it fold as usual.
///
/// # Panics
///
/// If `row_images` does not have one image per row of `bittable`, or the
/// table has fewer than 16 trees (`s < 4`).
#[tracing::instrument(name = "Evaluate grand-product circuit", level = "debug", skip_all)]
pub fn grand_product(row_images: &[Field], bittable: &BitTable) -> (Vec<Field>, LayerWitnesses) {
    assert_eq!(
        row_images.len(),
        bittable.shape().rows(),
        "one image per row of the table"
    );
    bounded::build(row_images, bittable, bounded::DENSE_NODES)
}

const NIBBLE_BITS: usize = 4;
const NIBBLES_IN_WORD: usize = u64::BITS as usize / NIBBLE_BITS;
/// A lookup table has an entry per nibble value.
const LUT_ENTRIES: usize = 1 << NIBBLE_BITS;

/// Layers 0 to 2 as lookup tables, one per four images: `lut[nibble]` is the
/// layer-2 node whose four leaves are the bits of `nibble`.
///
/// Each lookup table takes one image from each quarter of `images`.
fn lower_luts(images: &[Field]) -> Vec<[Field; LUT_ENTRIES]> {
    /// `lut[nibble]` is the product of the inputs whose bit is set in
    /// `nibble`, `a` on bit 0 through `d` on bit 3 (`0b0000 -> 1`,
    /// `0b0001 -> a`, `0b0011 -> a*b`, ...), so a nibble indexes its
    /// multilinear value directly.
    ///
    /// Built by doubling: entries `2^i..2^(i+1)` are entries `0..2^i`
    /// times the `i`-th input. The first entry of each new half is the
    /// input itself, so it is copied rather than multiplied by one,
    /// leaving 11 multiplications -- one per entry with two or more
    /// factors, which is the minimum.
    fn create_lut(c: [Field; NIBBLE_BITS]) -> [Field; LUT_ENTRIES] {
        let mut lut = [Field::ONE; LUT_ENTRIES];
        for (i, x) in c.into_iter().enumerate() {
            let half = 1 << i;
            lut[half] = x;
            for k in 1..half {
                lut[half + k] = lut[k] * x;
            }
        }
        lut
    }

    let quart = images.len() / 4;
    let (a, rest) = images.split_at(quart);
    let (b, rest) = rest.split_at(quart);
    let (c, d) = rest.split_at(quart);
    a.par_iter()
        .zip(b)
        .zip(c)
        .zip(d)
        // A lookup table writes `LUT_ENTRIES` entries.
        .with_min_len(PARALLEL_MIN_LANES / LUT_ENTRIES)
        .map(|(((&a, &b), &c), &d)| create_lut([a, b, c, d]))
        .collect()
}

/// Layer 3, the first layer built in full: node `k` multiplies layer-2 nodes
/// `k` and `k + n/2`, each a lookup-table entry looked up by its nibble.
/// Those are the two nibbles that round 1 of layer 1 reads at position `k`,
/// here read whole.
fn layer3(lut_layers: &LutLayers) -> Vec<Field> {
    let half = lut_layers.nodes::<LAYER2>() / 2;
    let mut layer3 = Field::zeroed_vec(half);
    layer3
        .par_chunks_mut(lut_layers.piece())
        .zip(lut_layers.pieces(half))
        .for_each(|(nodes, positions)| {
            let table = &lut_layers.pairs[positions.start >> lut_layers.log_trees];
            let (_, pairs) = lut_layers.pairs::<LAYER1>(positions);
            for (nodes, word) in nodes.chunks_exact_mut(NIBBLES_IN_WORD).zip(pairs) {
                for (node, [n0, n1, ..]) in nodes.iter_mut().zip(word) {
                    *node = table[n0 + n1 * LUT_ENTRIES];
                }
            }
        });
    layer3
}

/// Multiplies `layer` pairwise, left half against right half, until one
/// value per group remains. Returns that output layer and every layer before
/// it, `layer` first.
fn fold_layers(mut layer: Vec<Field>, groups: usize) -> (Vec<Field>, Vec<Vec<Field>>) {
    let rounds = (layer.len() / groups).checked_ilog2().unwrap_or(0);
    let mut witnesses = Vec::with_capacity(rounds as usize);

    // Stop when there is one output per group
    while layer.len() > groups {
        // Node `k` multiplies node `k` of the left half and node `k` of the
        // right half below it.
        let (l, r) = layer.split_at(layer.len() / 2);
        let next = l
            .par_iter()
            .zip(r)
            .with_min_len(PARALLEL_MIN_LANES)
            .map(|(&a, &b)| a * b)
            .collect();
        witnesses.push(layer);
        layer = next;
    }

    (layer, witnesses)
}

/// The intermediate layers. Layers 0 to 2 stay in their lookup tables,
/// which is how round 1 of their sumchecks reads them; layer 3 and up are
/// kept in full.
pub struct LayerWitnesses {
    /// Layers 0 to 2, in their lookup tables.
    lut_layers: LutLayers,
    first_level: usize,
    /// Materialized layers, starting at `first_level`.
    upper_nodes: Vec<Vec<Field>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::SecurityLevel::{Bits100, Bits128};
    use common::shape::{MIN_LOG_BITS, PACK_BITS};
    use common::{BitZParams, Shape};
    use field::gf128::smallest_generator;
    use proptest::prelude::*;

    fn field() -> impl Strategy<Value = Field> {
        any::<u128>().prop_map(Field::from)
    }

    /// `leaves` padded with ones to a power of two, so they have an MLE. The
    /// `fold_layers` tests below take any leaves, not only those a bit table
    /// gives.
    fn pad(mut leaves: Vec<Field>) -> Vec<Field> {
        if !leaves.is_empty() {
            leaves.resize(leaves.len().next_power_of_two(), Field::ONE);
        }
        leaves
    }

    proptest! {
        // fold_layers, unbatched (groups = 1): pairwise multiplication is
        // associative, so folding-by-multiply any witness layer (including
        // the leaves, padded with the multiplicative identity 1) must equal
        // folding the original input. Checks every intermediate layer, not
        // just the top.
        #[test]
        fn eval_preserves_product_across_layers(leaves in prop::collection::vec(field(), 0..12)) {
            let expected: Field = leaves.iter().product();

            let (top, witnesses) = fold_layers(pad(leaves), 1);

            prop_assert_eq!(top.iter().product::<Field>(), expected);

            for layer in witnesses {
                prop_assert_eq!(layer.iter().product::<Field>(), expected);
            }
        }

        #[test]
        fn fold_layers_top_layer_is_interleaved_group_products(
            leaves in prop::collection::vec(field(), 1..32),
            groups_log2 in 0usize..4,
        ) {
            let padded = pad(leaves);
            let groups = 1usize << groups_log2.min(padded.len().ilog2() as usize);

            let (top, _witnesses) = fold_layers(padded.clone(), groups);
            prop_assert_eq!(top.len(), groups);

            for (j, &value) in top.iter().enumerate() {
                let expected: Field = padded[j..].iter().step_by(groups).product();
                prop_assert_eq!(value, expected);
            }
        }
    }

    /// The smallest shape the gates accept, `(t, s)`: the narrowest `t` at
    /// the smallest total.
    const SMALLEST: (usize, usize) = (PACK_BITS as usize, MIN_LOG_BITS - PACK_BITS as usize);

    /// How the bits of a test table are set.
    #[derive(Clone, Copy, Debug)]
    enum Bits {
        /// Every leaf is one.
        Clear,
        /// Every leaf is its row's image.
        Set,
        /// Bits drawn from the seed.
        Drawn,
    }

    fn bits() -> impl Strategy<Value = Bits> {
        prop_oneof![Just(Bits::Clear), Just(Bits::Set), Just(Bits::Drawn)]
    }

    /// A splitmix64 stream: test data that is fixed by its seed.
    fn stream(seed: u64) -> impl Iterator<Item = u64> {
        let mut x = seed;
        std::iter::repeat_with(move || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let z = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        })
    }

    fn element(words: &mut impl Iterator<Item = u64>) -> F128 {
        let mut word = || words.next().unwrap();
        F128::from([word(), word()])
    }

    /// The packed bits of a `(t, s)` table and one image per row, from `seed`.
    fn instance((t, s): (usize, usize), bits: Bits, seed: u64) -> (Shape, Vec<F128>, Vec<Field>) {
        let shape = Shape::new(t, s).unwrap();
        let mut words = stream(seed);
        let packed = (0..1usize << shape.log_packed_len())
            .map(|_| match bits {
                Bits::Clear => F128::ZERO,
                Bits::Set => F128::from(u128::MAX),
                Bits::Drawn => element(&mut words),
            })
            .collect();
        let row_images = (0..shape.rows()).map(|_| element(&mut words)).collect();
        (shape, packed, row_images)
    }

    fn table(shape: Shape, packed: &[F128]) -> BitTable<'_> {
        BitZParams::<field::FqDefault>::new(shape, smallest_generator())
            .unwrap()
            .table(packed)
            .unwrap()
    }

    /// Leaf `(b << s) | c` is `row_images[b]` where `bit(c, b)` is set and one
    /// elsewhere, as `LutLayers::new` describes, built one leaf at a
    /// time: the reference the circuit is checked against.
    fn leaves(table: &BitTable, row_images: &[Field]) -> Vec<Field> {
        let shape = table.shape();
        let mut leaves = vec![Field::ONE; 1 << shape.log_bits()];
        for (b, &image) in row_images.iter().enumerate() {
            for c in 0..shape.columns() {
                if table.bit(c, b) {
                    leaves[(b << shape.log_columns()) | c] = image;
                }
            }
        }
        leaves
    }

    proptest! {
        // Every case proves a whole circuit at the smallest shape.
        #![proptest_config(ProptestConfig::with_cases(4))]

        // Full protocol: the verifier accepts, and the output is the product
        // of each tree's leaves.
        #[test]
        fn gpgkr_round_trip(bits in bits(), seed in any::<u64>()) {
            let (shape, packed, row_images) = instance(SMALLEST, bits, seed);
            let table = table(shape, &packed);
            let leaves = leaves(&table, &row_images);

            let (output, proof) = prove(&table, &row_images, Bits100);
            prop_assert!(verify(&leaves, &output, &proof, Bits100));

            let columns = shape.columns();
            let expected: Vec<Field> = (0..columns)
                .map(|c| leaves[c..].iter().step_by(columns).product())
                .collect();
            prop_assert_eq!(output, expected);
        }

        // negative/soundness test
        #[test]
        fn gpgkr_verify_rejects_tampered_output(bits in bits(), seed in any::<u64>()) {
            let (shape, packed, row_images) = instance(SMALLEST, bits, seed);
            let table = table(shape, &packed);
            let (output, proof) = prove(&table, &row_images, Bits100);

            let mut tampered = output;
            tampered[0] += Field::ONE;

            prop_assert!(!verify(&leaves(&table, &row_images), &tampered, &proof, Bits100));
        }
    }

    // At 128 bits every cubic challenge is ground: the proof carries a nonce
    // per sumcheck round, and the verifier rejects a changed one.
    #[test]
    fn gpgkr_security_targets_replay_and_reject_changed_nonces() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 6);
        let table = table(shape, &packed);
        let leaves = leaves(&table, &row_images);

        let (_, unground) = prove(&table, &row_images, Bits100);
        let (output, mut proof) = prove(&table, &row_images, Bits128);
        assert!(verify(&leaves, &output, &proof, Bits128));

        // One cubic challenge per coordinate of each layer's point, from the
        // output's `s` up to the leaves' `log_bits - 1`, each with an
        // eight-byte nonce.
        let cubic_challenges: usize = (shape.log_columns()..shape.log_bits()).sum();
        assert_eq!(
            proof.narg_string.len(),
            unground.narg_string.len() + 8 * cubic_challenges
        );
        // The first cubic message occupies two canonical field elements.
        proof.narg_string[32] ^= 1;
        assert!(!verify(&leaves, &output, &proof, Bits128));
    }

    // Zero and one coordinates in the output point take the `sends_one`
    // branch in the top layer's rounds.
    #[test]
    fn gpgkr_round_trip_with_zero_coordinates() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 1);
        let table = table(shape, &packed);
        let leaves = leaves(&table, &row_images);
        let s = shape.log_columns();
        let layers = shape.log_rows();

        // `value` at coordinate `i`, and a nonzero value at every other.
        let nonzero = Field::from(5u128);
        let with = |i: usize, value: Field| -> Vec<Field> {
            (0..s)
                .map(|j| if j == i { value } else { nonzero })
                .collect()
        };
        for point in [
            vec![Field::ZERO; s],
            with(s - 1, Field::ZERO),
            with(0, Field::ZERO),
            with(s - 1, Field::ONE),
            with(0, Field::ONE),
        ] {
            let (output, witnesses) = grand_product(&row_images, &table);
            let claim = mle(&output, &point);
            let mut prover = transcript::build_prover("gkr-zero", &point);
            let terminal = gpgkr_prove(&mut prover, &point, witnesses, Bits100);
            assert_eq!(terminal.1, mle(&leaves, &terminal.0));
            let proof = prover.finish();

            let mut verifier = transcript::build_verifier("gkr-zero", &point, &proof);
            assert_eq!(
                gpgkr_verify(&mut verifier, claim, &point, layers, Bits100),
                Some(terminal)
            );
            verifier.check_eof().unwrap();

            let mut verifier = transcript::build_verifier("gkr-zero", &point, &proof);
            assert!(
                gpgkr_verify(&mut verifier, claim + Field::ONE, &point, layers, Bits100).is_none()
            );
        }
    }

    #[test]
    fn bounded_prover_matches_dense_transcript() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 7);
        let table = table(shape, &packed);
        let dense_limit = shape.columns() * 4;
        let (expected, dense) = grand_product(&row_images, &table);
        let (output, compact) = bounded::build(&row_images, &table, dense_limit);
        assert_eq!(expected, output);

        let mut point = vec![Field::from(5u128); shape.log_columns()];
        point[0] = Field::ZERO;
        let mut old = transcript::build_prover("bounded-gkr", &point);
        let mut new = transcript::build_prover("bounded-gkr", &point);
        let old_result = gpgkr_prove(&mut old, &point, dense, Bits100);
        let new_result = bounded::prove(&mut new, point.clone(), compact, Bits100, dense_limit);
        assert_eq!(old_result, new_result);
        let new_proof = new.finish();
        assert_eq!(old.finish(), new_proof);
        let mut verifier = transcript::build_verifier("bounded-gkr", &point, &new_proof);
        assert_eq!(
            gpgkr_verify(
                &mut verifier,
                mle(&output, &point),
                &point,
                shape.log_rows(),
                Bits100
            ),
            Some(new_result)
        );
        verifier.check_eof().unwrap();
    }

    /// Proves the smallest shape from a point of `len(s)` coordinates, `s`
    /// being the tree bits. The bits are left clear: the point's length is
    /// checked before any of them is read.
    fn prove_from_point(len: fn(usize) -> usize) {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Clear, 1);
        let table = table(shape, &packed);
        let (_, witnesses) = grand_product(&row_images, &table);
        let point = vec![Field::from(5u128); len(shape.log_columns())];
        let mut prover = transcript::build_prover("gkr-point", &point);
        let _ = gpgkr_prove(&mut prover, &point, witnesses, Bits100);
    }

    #[test]
    #[should_panic(expected = "a coordinate per tree bit")]
    fn gpgkr_prove_rejects_a_short_point() {
        prove_from_point(|s| s - 1);
    }

    #[test]
    #[should_panic(expected = "a coordinate per tree bit")]
    fn gpgkr_prove_rejects_a_long_point() {
        prove_from_point(|s| s + 1);
    }

    /// Each LUT-backed layer proved on its own, from a point whose top
    /// coordinate is zero (round 1's `sends_one` branch) or not: the verifier
    /// accepts it, and the final claim is the MLE of the children.
    #[test]
    fn lut_layers_prove_from_any_top_coordinate() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 4);
        let table = table(shape, &packed);
        // The leaves and every layer above, built one leaf at a time.
        let (_, layers) = fold_layers(leaves(&table, &row_images), shape.columns());
        let (_, witnesses) = grand_product(&row_images, &table);
        let lut_layers = witnesses.lut_layers;
        let mut bound_children = Field::zeroed_vec(lut_layers.nodes::<LAYER1>());
        let mut eq_memory = EqTables::alloc_storage(lut_layers.nodes::<LAYER1>());

        let mut draws = stream(5);
        for layer in [LEAVES, LAYER1, LAYER2] {
            type Prove = fn(
                &mut ProverState,
                &mut [Field],
                &[Field],
                &LutLayers,
                &mut [Field],
                SecurityLevel,
            ) -> (Vec<Field>, Field);
            let prove: Prove = match layer {
                LEAVES => prove_lut_layer::<LEAVES>,
                LAYER1 => prove_lut_layer::<LAYER1>,
                _ => prove_lut_layer::<LAYER2>,
            };
            let (children, parent) = (&layers[layer], &layers[layer + 1]);
            for top in [Field::ZERO, Field::from(7u128)] {
                let mut point: Vec<Field> = (0..parent.len().ilog2())
                    .map(|_| element(&mut draws))
                    .collect();
                *point.last_mut().unwrap() = top;

                let mut prover = transcript::build_prover("lut-layer", &point);
                let terminal = prove(
                    &mut prover,
                    &mut eq_memory,
                    &point,
                    &lut_layers,
                    &mut bound_children,
                    Bits100,
                );
                let proof = prover.finish();

                let mut verifier = transcript::build_verifier("lut-layer", &point, &proof);
                let claim = mle(parent, &point);
                assert_eq!(
                    verify_layer(&mut verifier, claim, &point, Bits100),
                    Some(terminal.clone()),
                    "layer {layer}, top coordinate {top}"
                );
                assert_eq!(terminal.1, mle(children, &terminal.0));
            }
        }
    }

    /// The output and every layer from layer 3 up match the circuit built one
    /// leaf at a time from the same table. Layers 0 to 2 stay in their lookup
    /// tables; `lut_layers_prove_from_any_top_coordinate` checks them through
    /// the sumchecks that read them.
    #[test]
    fn lut_layers_match_the_leaf_construction() {
        // At t = 7 and 8 there are 32 and 64 lookup tables, fewer than the bits
        // of a `u128`, so the transpose in `LutLayers::new` moves
        // `u32` and `u64` blocks; from t = 9 on it moves `u128` ones. At s = 7
        // a lookup table's run is 128 nibbles, 8 words, below
        // `PARALLEL_MIN_LANES`, so a piece of `layer3` is a whole run rather
        // than part of one.
        for (log_rows, log_columns) in [(7, 15), (8, 14), (9, 13), (15, 7)] {
            let (shape, packed, row_images) = instance((log_rows, log_columns), Bits::Drawn, 3);
            let table = table(shape, &packed);
            let (expected_output, expected_layers) =
                fold_layers(leaves(&table, &row_images), shape.columns());

            let (output, witnesses) = grand_product(&row_images, &table);
            // Both start at layer 3: the reference after its first three.
            let (layers, expected_layers) = (&witnesses.upper_nodes, &expected_layers[3..]);

            assert!(output == expected_output, "output, t = {log_rows}");
            assert_eq!(layers.len(), expected_layers.len(), "t = {log_rows}");
            for (i, (layer, expected)) in layers.iter().zip(expected_layers).enumerate() {
                assert!(layer == expected, "layer {}, t = {log_rows}", i + 3);
            }
        }
    }

    #[test]
    fn gpgkr_verify_rejects_every_truncated_prefix() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 2);
        let table = table(shape, &packed);
        let leaves = leaves(&table, &row_images);
        let (output, proof) = prove(&table, &row_images, Bits100);
        assert!(verify(&leaves, &output, &proof, Bits100));

        // Cover partial field elements and missing messages in every layer.
        for len in 0..proof.narg_string.len() {
            let mut truncated = proof.clone();
            truncated.narg_string.truncate(len);
            assert!(
                !verify(&leaves, &output, &truncated, Bits100),
                "accepted {len} bytes"
            );
        }
    }

    /// Proves the circuit of `table` at a point the verifier draws, as
    /// `gkr_reduce` does. Returns the output layer, one product per tree.
    fn prove(
        table: &BitTable,
        row_images: &[Field],
        security: SecurityLevel,
    ) -> (Vec<Field>, transcript::Proof) {
        let (output, witnesses) = grand_product(row_images, table);

        // TODO instance is a bit loose and should be replaced by PCS
        let instance = (output.clone(), 1u128 << table.shape().log_bits());
        let mut prover = transcript::build_prover("gkr", &instance);

        // Mirrors `verify`, since both sides must draw the same number of
        // challenges here.
        let point: Vec<Field> = (0..output.len().ilog2())
            .map(|_| prover.verifier_message())
            .collect();

        // `verify` checks the terminal claim against the leaves itself.
        let _terminal = gpgkr_prove(&mut prover, &point, witnesses, security);
        (output, prover.finish())
    }

    /// Verifies `proof` for `output`, then checks the final claim against the
    /// MLE of `leaves`.
    fn verify(
        leaves: &[Field],
        output: &[Field],
        proof: &transcript::Proof,
        security: SecurityLevel,
    ) -> bool {
        let instance = (output.to_vec(), leaves.len() as u128);
        let mut verifier = transcript::build_verifier("gkr", &instance, proof);

        let point: Vec<Field> = (0..output.len().ilog2())
            .map(|_| verifier.verifier_message())
            .collect();
        let claim = mle(output, &point);
        let layers = (leaves.len().ilog2() - output.len().ilog2()) as usize;

        match gpgkr_verify(&mut verifier, claim, &point, layers, security) {
            Some((point, claim)) => mle(leaves, &point) == claim && verifier.check_eof().is_ok(),
            None => false,
        }
    }

    /// The multilinear extension of `evaluations` at `point`.
    fn mle(evaluations: &[Field], point: &[Field]) -> Field {
        poly::DenseMultilinearExtension::evaluate_exact(evaluations, point)
    }
}
