//! The fused einsum contraction and its parsed equation type.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    error::{KernelError, ShapeError},
    id::Scale,
    tensor::{Tensor, row_major_strides as strides},
};

use super::{Einsum, Eval, EvalCtx, OpSpec, PadBehavior, fused::rebase_eval};

/// A parsed, validated einsum equation (`"mk,kn->mn"`).
///
/// Parsing happens once at construction; kernels and analyses never re-split
/// strings. Constraints: ASCII lowercase index letters, no letter repeated
/// within one operand, and every output letter bound by some operand.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EinsumEq {
    inputs: Vec<Vec<u8>>,
    output: Vec<u8>,
}

impl EinsumEq {
    /// Index letters of each operand.
    pub fn operands(&self) -> &[Vec<u8>] {
        &self.inputs
    }

    /// Index letters of the output.
    pub fn output(&self) -> &[u8] {
        &self.output
    }

    /// The contracted (summed-over) letters, in first-occurrence order.
    pub fn contracted(&self) -> Vec<u8> {
        let mut seen = Vec::new();
        for operand in &self.inputs {
            for &c in operand {
                if !self.output.contains(&c) && !seen.contains(&c) {
                    seen.push(c);
                }
            }
        }
        seen
    }

    /// Resolve each letter's size from the operands' dims.
    fn letter_sizes(&self, dims: &[&[usize]]) -> Result<HashMap<u8, usize>, ShapeError> {
        if dims.len() != self.inputs.len() {
            return Err(ShapeError::WrongArity {
                expected: self.inputs.len(),
                got: dims.len(),
            });
        }
        let mut sizes = HashMap::new();
        for (letters, d) in self.inputs.iter().zip(dims) {
            if letters.len() != d.len() {
                return Err(ShapeError::Invalid(format!(
                    "einsum operand {letters:?} has {} letters but rank {}",
                    letters.len(),
                    d.len()
                )));
            }
            for (&c, &size) in letters.iter().zip(*d) {
                if *sizes.entry(c).or_insert(size) != size {
                    return Err(ShapeError::Invalid(format!(
                        "einsum letter '{}' bound to conflicting sizes",
                        c as char
                    )));
                }
            }
        }
        Ok(sizes)
    }
}

impl FromStr for EinsumEq {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (lhs, output) = s.split_once("->").ok_or("missing '->'")?;
        let parse_letters = |part: &str| -> Result<Vec<u8>, String> {
            part.bytes()
                .map(|c| {
                    c.is_ascii_lowercase()
                        .then_some(c)
                        .ok_or_else(|| format!("invalid index letter {:?}", c as char))
                })
                .collect()
        };
        let inputs: Vec<Vec<u8>> = lhs
            .split(',')
            .map(parse_letters)
            .collect::<Result<_, _>>()?;
        let output = parse_letters(output)?;

        if inputs.is_empty() || inputs.iter().any(|i| i.is_empty() && inputs.len() > 1) {
            return Err("einsum needs at least one non-empty operand".into());
        }
        for operand in &inputs {
            for (i, c) in operand.iter().enumerate() {
                if operand[..i].contains(c) {
                    return Err(format!(
                        "letter '{}' repeated within one operand",
                        *c as char
                    ));
                }
            }
        }
        for (i, c) in output.iter().enumerate() {
            if output[..i].contains(c) {
                return Err(format!("letter '{}' repeated in output", *c as char));
            }
            if !inputs.iter().any(|op| op.contains(c)) {
                return Err(format!("output letter '{}' unbound", *c as char));
            }
        }
        Ok(EinsumEq { inputs, output })
    }
}

impl fmt::Display for EinsumEq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, operand) in self.inputs.iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            for &c in operand {
                write!(f, "{}", c as char)?;
            }
        }
        f.write_str("->")?;
        for &c in &self.output {
            write!(f, "{}", c as char)?;
        }
        Ok(())
    }
}

impl Serialize for EinsumEq {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for EinsumEq {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// `tensor`'s data with its axes reordered: axis `order[i]` of the tensor is axis `i` of the
/// result, row-major. Borrowed when the order is the tensor's own.
fn permuted<'a>(tensor: &'a Tensor<i32>, order: &[usize]) -> Cow<'a, [i32]> {
    let data = tensor.data();
    if order.iter().copied().eq(0..order.len()) {
        return Cow::Borrowed(data);
    }
    let source_strides = strides(tensor.dims());
    let dims: Vec<usize> = order.iter().map(|&axis| tensor.dims()[axis]).collect();
    let steps: Vec<usize> = order.iter().map(|&axis| source_strides[axis]).collect();
    let (&run, outer_dims) = dims.split_last().expect("a reordered tensor has axes");
    let (&run_step, outer_steps) = steps.split_last().expect("a reordered tensor has axes");
    let outer_strides = strides(outer_dims);
    // One run of the last axis: a strided read of the source.
    let fill = |(index, out): (usize, &mut [i32])| {
        let base: usize = outer_strides
            .iter()
            .zip(outer_dims)
            .zip(outer_steps)
            .map(|((&stride, &dim), &step)| (index / stride) % dim * step)
            .sum();
        for (j, value) in out.iter_mut().enumerate() {
            *value = data[base + j * run_step];
        }
    };
    let mut out = vec![0i32; data.len()];
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(run.max(1)).enumerate().for_each(fill);
    }
    #[cfg(not(feature = "rayon"))]
    out.chunks_mut(run.max(1)).enumerate().for_each(fill);
    Cow::Owned(out)
}

/// Row `j` of `values` (rows of `run` entries), from column `start` for `width` columns.
fn columns(values: &[i32], run: usize, start: usize, width: usize, j: usize) -> &[i32] {
    &values[j * run + start..j * run + start + width]
}

/// The bytes of output one task of [`scaled_rows_acc_i64`] holds: a quarter of a megabyte,
/// inside a second-level cache.
const TILE_BYTES: usize = 1 << 18;
/// The fewest columns in a tile, so that a run of the second operand is worth a loop.
const MIN_TILE: usize = 64;
/// The fewest rows of the second operand one task takes.
const MIN_PART: usize = 16;

/// The contraction when the second operand is laid out summed letters first and the output is
/// the first operand's kept letters followed by the second's (`mk,kn->mn`): an output row is
/// the sum of the second operand's rows scaled by the first operand's entries, read in place.
/// `None` if the equation is not of that form.
///
/// A task takes a tile of the output's columns, narrow enough that its outputs stay in
/// cache, and a run of the second operand's rows, which it reads once, in order, adding each
/// to every output row. The tasks of one tile sum into blocks of their own, which are added
/// up at the end. The second operand is read from memory once whatever the number of output
/// rows, and an output is the same `i64` sum of `i32` products in any order.
fn scaled_rows_acc_i64(
    eq: &EinsumEq,
    inputs: &[&Tensor<i32>],
    sum_letters: &[u8],
    sum_total: usize,
) -> Option<Vec<i64>> {
    let kept = |letters: &[u8]| -> Vec<u8> {
        letters
            .iter()
            .copied()
            .filter(|c| !sum_letters.contains(c))
            .collect()
    };
    let (kept_a, kept_b) = (kept(&eq.inputs[0]), kept(&eq.inputs[1]));
    let second_in_place = eq.inputs[1].iter().eq(sum_letters.iter().chain(&kept_b));
    let output_is_rows = eq.output.iter().eq(kept_a.iter().chain(&kept_b));
    if kept_b.is_empty() || !second_in_place || !output_is_rows {
        return None;
    }
    // The first operand with its summed letters first: the scales of one row of the second
    // operand, one per output row, are a contiguous run.
    let position = |c: &u8| eq.inputs[0].iter().position(|l| l == c).expect("carried");
    let order: Vec<usize> = sum_letters.iter().chain(&kept_a).map(position).collect();
    let a = permuted(inputs[0], &order);
    let b = inputs[1].data();
    let run = b.len() / sum_total;
    let rows = a.len() / sum_total;
    let mut out = vec![0i64; rows * run];
    if rows == 0 || run == 0 {
        return Some(out);
    }

    #[cfg(feature = "rayon")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "rayon"))]
    let threads = 1usize;
    let tile = run.min((TILE_BYTES / (rows * size_of::<i64>())).max(MIN_TILE));
    let tiles = run.div_ceil(tile);
    // About two tasks a thread: the second operand's rows are split when the tiles are few.
    let parts = (2 * threads)
        .div_ceil(tiles)
        .clamp(1, sum_total.div_ceil(MIN_PART));
    let part = sum_total.div_ceil(parts).next_multiple_of(4);
    let parts = sum_total.div_ceil(part);

    // Four rows of the second operand at a time: an output run is read and written once for
    // the four, and a group whose scales are all zero (padding, mostly) is skipped.
    let block = |task: usize| -> Vec<i64> {
        let (tile_index, part_index) = (task / parts, task % parts);
        let start = tile_index * tile;
        let width = tile.min(run - start);
        let (first, last) = (part_index * part, ((part_index + 1) * part).min(sum_total));
        let (values, scales) = (&b[first * run..last * run], &a[first * rows..last * rows]);
        let mut block = vec![0i64; rows * width];
        for (values, scales) in values.chunks(4 * run).zip(scales.chunks(4 * rows)) {
            if scales.len() == 4 * rows {
                let (v0, v1) = (
                    columns(values, run, start, width, 0),
                    columns(values, run, start, width, 1),
                );
                let (v2, v3) = (
                    columns(values, run, start, width, 2),
                    columns(values, run, start, width, 3),
                );
                for (row, output) in block.chunks_exact_mut(width).enumerate() {
                    let s = [0, 1, 2, 3].map(|j| scales[j * rows + row]);
                    if s == [0; 4] {
                        continue;
                    }
                    for ((((o, &x0), &x1), &x2), &x3) in
                        output.iter_mut().zip(v0).zip(v1).zip(v2).zip(v3)
                    {
                        *o += s[0] as i64 * x0 as i64
                            + s[1] as i64 * x1 as i64
                            + s[2] as i64 * x2 as i64
                            + s[3] as i64 * x3 as i64;
                    }
                }
            } else {
                for (j, scales) in scales.chunks_exact(rows).enumerate() {
                    let values = columns(values, run, start, width, j);
                    for (output, &scale) in block.chunks_exact_mut(width).zip(scales) {
                        if scale == 0 {
                            continue;
                        }
                        for (o, &value) in output.iter_mut().zip(values) {
                            *o += scale as i64 * value as i64;
                        }
                    }
                }
            }
        }
        block
    };
    // An output row is, tile by tile, the sum of the tile's blocks' rows.
    let gather = |blocks: &[Vec<i64>], (row, out): (usize, &mut [i64])| {
        for (tile_blocks, out) in blocks.chunks(parts).zip(out.chunks_mut(tile)) {
            for block in tile_blocks {
                let width = out.len();
                for (o, &value) in out.iter_mut().zip(&block[row * width..(row + 1) * width]) {
                    *o += value;
                }
            }
        }
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let blocks: Vec<Vec<i64>> = (0..tiles * parts).into_par_iter().map(block).collect();
        out.par_chunks_mut(run)
            .enumerate()
            .for_each(|row| gather(&blocks, row));
    }
    #[cfg(not(feature = "rayon"))]
    {
        let blocks: Vec<Vec<i64>> = (0..tiles * parts).map(block).collect();
        out.chunks_mut(run)
            .enumerate()
            .for_each(|row| gather(&blocks, row));
    }
    Some(out)
}

/// The contraction of two operands that both carry every summed letter: each operand is
/// laid out with its summed letters last, so an output entry is a dot product of two
/// contiguous runs. `None` if the equation is not of that form.
fn pairwise_acc_i64(
    eq: &EinsumEq,
    inputs: &[&Tensor<i32>],
    sizes: &HashMap<u8, usize>,
    out_shape: &[usize],
) -> Option<Vec<i64>> {
    let sum_letters = eq.contracted();
    let carries_sums = |letters: &Vec<u8>| sum_letters.iter().all(|c| letters.contains(c));
    if inputs.len() != 2 || sum_letters.is_empty() || !eq.inputs.iter().all(carries_sums) {
        return None;
    }
    let sum_total: usize = sum_letters.iter().map(|c| sizes[c]).product();
    let out_strides = strides(out_shape);

    // Each operand's data with its summed letters last, and for each of its output letters
    // the output axis's stride and size and the operand's step along it.
    type Kept = Vec<(usize, usize, usize)>;
    let layout = |letters: &Vec<u8>, tensor: &'_ Tensor<i32>| -> (Vec<usize>, Kept) {
        let position = |c: &u8| letters.iter().position(|l| l == c).expect("carried");
        let kept: Vec<u8> = letters
            .iter()
            .copied()
            .filter(|c| !sum_letters.contains(c))
            .collect();
        let order: Vec<usize> = kept.iter().chain(&sum_letters).map(position).collect();
        let kept_dims: Vec<usize> = kept.iter().map(|c| tensor.dims()[position(c)]).collect();
        let terms = kept
            .iter()
            .zip(strides(&kept_dims))
            .map(|(c, stride)| {
                let axis = eq
                    .output
                    .iter()
                    .position(|o| o == c)
                    .expect("an output letter");
                (out_strides[axis], out_shape[axis], stride * sum_total)
            })
            .collect();
        (order, terms)
    };
    if let Some(data) = scaled_rows_acc_i64(eq, inputs, &sum_letters, sum_total) {
        return Some(data);
    }
    let (order_a, terms_a) = layout(&eq.inputs[0], inputs[0]);
    let (order_b, terms_b) = layout(&eq.inputs[1], inputs[1]);
    let a = permuted(inputs[0], &order_a);
    let b = permuted(inputs[1], &order_b);

    let base = |terms: &Kept, out_flat: usize| -> usize {
        terms
            .iter()
            .map(|&(stride, dim, step)| (out_flat / stride) % dim * step)
            .sum()
    };
    let eval_one = |out_flat: usize| -> i64 {
        let (start_a, start_b) = (base(&terms_a, out_flat), base(&terms_b, out_flat));
        a[start_a..start_a + sum_total]
            .iter()
            .zip(&b[start_b..start_b + sum_total])
            .map(|(&x, &y)| x as i64 * y as i64)
            .sum()
    };
    let out_total: usize = out_shape.iter().product();
    #[cfg(feature = "rayon")]
    let data: Vec<i64> = {
        use rayon::prelude::*;
        (0..out_total)
            .into_par_iter()
            .with_min_len(64)
            .map(eval_one)
            .collect()
    };
    #[cfg(not(feature = "rayon"))]
    let data: Vec<i64> = (0..out_total).map(eval_one).collect();
    Some(data)
}

/// Raw i64 contraction `acc[out] = Σ_sum Π_i operand_i[...]`.
fn einsum_acc_i64(eq: &EinsumEq, inputs: &[&Tensor<i32>]) -> Result<Tensor<i64>, ShapeError> {
    let dims: Vec<&[usize]> = inputs.iter().map(|t| t.dims()).collect();
    let sizes = eq.letter_sizes(&dims)?;

    let mut out_shape: Vec<usize> = eq.output.iter().map(|c| sizes[c]).collect();
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    if let Some(data) = pairwise_acc_i64(eq, inputs, &sizes, &out_shape) {
        return Tensor::new(data, &out_shape);
    }
    let sum_letters = eq.contracted();
    let sum_sizes: Vec<usize> = sum_letters.iter().map(|c| sizes[c]).collect();
    let sum_total: usize = sum_sizes.iter().product::<usize>().max(1);
    let out_total: usize = out_shape.iter().product();

    // For each operand: per-letter (position-in-out-or-sum, stride) split.
    struct DimMap {
        out_terms: Vec<(usize, usize)>, // (output-coord position, stride)
        sum_terms: Vec<(usize, usize)>, // (sum-coord position, stride)
    }
    let maps: Vec<DimMap> = eq
        .inputs
        .iter()
        .zip(&dims)
        .map(|(letters, d)| {
            let st = strides(d);
            let mut m = DimMap {
                out_terms: vec![],
                sum_terms: vec![],
            };
            for (i, c) in letters.iter().enumerate() {
                if let Some(p) = eq.output.iter().position(|o| o == c) {
                    m.out_terms.push((p, st[i]));
                } else {
                    let p = sum_letters.iter().position(|s| s == c).expect("contracted");
                    m.sum_terms.push((p, st[i]));
                }
            }
            m
        })
        .collect();

    // Precompute each operand's flat offset for every sum coordinate.
    let sum_strides = strides(&sum_sizes);
    let sum_offsets: Vec<Vec<usize>> = maps
        .iter()
        .map(|m| {
            (0..sum_total)
                .map(|flat| {
                    m.sum_terms
                        .iter()
                        .map(|&(p, stride)| (flat / sum_strides[p]) % sum_sizes[p] * stride)
                        .sum()
                })
                .collect()
        })
        .collect();

    let out_strides = strides(&out_shape);
    let eval_one = |out_flat: usize| -> i64 {
        let out_coord: Vec<usize> = out_strides
            .iter()
            .zip(&out_shape)
            .map(|(&s, &d)| (out_flat / s) % d)
            .collect();
        let bases: Vec<usize> = maps
            .iter()
            .map(|m| {
                m.out_terms
                    .iter()
                    .map(|&(p, stride)| out_coord[p] * stride)
                    .sum()
            })
            .collect();
        let mut acc = 0i64;
        #[allow(clippy::needless_range_loop)]
        for s in 0..sum_total {
            let mut product = 1i64;
            for (i, input) in inputs.iter().enumerate() {
                product *= input.data()[bases[i] + sum_offsets[i][s]] as i64;
            }
            acc += product;
        }
        acc
    };

    #[cfg(feature = "rayon")]
    let data: Vec<i64> = {
        use rayon::prelude::*;
        (0..out_total).into_par_iter().map(eval_one).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let data: Vec<i64> = (0..out_total).map(eval_one).collect();

    Tensor::new(data, &out_shape)
}

impl fmt::Display for Einsum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Einsum({})", self.equation)
    }
}

impl OpSpec for Einsum {
    fn eval(&self, inputs: &[&Tensor<i32>], ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        let acc = einsum_acc_i64(&self.equation, inputs)?;
        rebase_eval(
            acc,
            self.rebase_divisor(ctx.scale).expect("fused"),
            "einsum",
        )
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        let sizes = self.equation.letter_sizes(inputs)?;
        let mut shape: Vec<usize> = self.equation.output().iter().map(|c| sizes[c]).collect();
        if shape.is_empty() {
            shape.push(1);
        }
        Ok(shape)
    }

    fn checks_output(&self) -> bool {
        true
    }

    fn padding(&self) -> PadBehavior {
        // Zero-padded weight lanes contribute zero to every contraction.
        PadBehavior::ZeroPreserving
    }

    fn rebase_divisor(&self, scale: Scale) -> Option<i64> {
        Some(scale.denom_i64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::SemVersion;
    use crate::op::TableCache;

    fn ctx(cache: &TableCache) -> EvalCtx<'_> {
        EvalCtx {
            scale: Scale::new(4).unwrap(),
            version: SemVersion::V1,
            tables: cache,
        }
    }

    fn t(data: &[i32], dims: &[usize]) -> Tensor<i32> {
        Tensor::new(data.to_vec(), dims).unwrap()
    }

    #[test]
    fn parse_display_roundtrip() {
        for s in ["mk,kn->mn", "abmk,abnk->abmn", "k,k->", "mk,k->m"] {
            let eq: EinsumEq = s.parse().unwrap();
            assert_eq!(eq.to_string(), s);
        }
        assert!("mk,kn".parse::<EinsumEq>().is_err()); // no ->
        assert!("mm,mn->mn".parse::<EinsumEq>().is_err()); // repeated in operand
        assert!("mk,kn->mq".parse::<EinsumEq>().is_err()); // unbound output
    }

    #[test]
    fn matmul_matches_reference() {
        // [2x3]·[3x2] at scale 4 (divisor 16).
        let eq: EinsumEq = "mk,kn->mn".parse().unwrap();
        let a = t(&[16, 32, 48, 64, 80, 96], &[2, 3]); // 1,2,3;4,5,6 at S=16
        let b = t(&[16, 0, 0, 16, 16, 16], &[3, 2]);
        let op = Einsum { equation: eq };
        let cache = TableCache::default();
        let out = op.eval(&[&a, &b], &ctx(&cache)).unwrap();
        // real A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]
        // A·B = [[4,5],[10,11]] → quantized ×16.
        assert_eq!(out.output.data(), &[4 * 16, 5 * 16, 10 * 16, 11 * 16]);
        // Fused identity: acc = q·16 + r.
        let crate::trace::NodeAux::Fused(f) = out.aux.unwrap() else {
            panic!()
        };
        assert!(f.remainder.data().iter().all(|&r| (0..16).contains(&r)));
    }

    /// A batched contraction whose summed letter is not last in either operand, against the
    /// sum written out.
    #[test]
    fn reordered_contraction_matches_the_sum() {
        let eq: EinsumEq = "akm,bkn->abmn".parse().unwrap();
        let (da, db, dk, dm, dn) = (2usize, 3, 4, 5, 2);
        let value = |seed: usize| (seed * 37 % 23) as i32 - 11;
        let a: Vec<i32> = (0..da * dk * dm).map(value).collect();
        let b: Vec<i32> = (0..db * dk * dn).map(|i| value(i + 7)).collect();
        let (a, b) = (t(&a, &[da, dk, dm]), t(&b, &[db, dk, dn]));
        let acc = einsum_acc_i64(&eq, &[&a, &b]).unwrap();
        assert_eq!(acc.dims(), &[da, db, dm, dn]);
        for (flat, &got) in acc.data().iter().enumerate() {
            let (n, m) = (flat % dn, flat / dn % dm);
            let (ib, ia) = (flat / (dn * dm) % db, flat / (dn * dm * db));
            let want: i64 = (0..dk)
                .map(|k| {
                    a.data()[(ia * dk + k) * dm + m] as i64
                        * b.data()[(ib * dk + k) * dn + n] as i64
                })
                .sum();
            assert_eq!(got, want, "entry {flat}");
        }
    }

    /// A matrix product with zero entries in both operands, against the sum written out.
    #[test]
    fn matmul_with_zeros_matches_the_sum() {
        let eq: EinsumEq = "mk,kn->mn".parse().unwrap();
        let (dm, dk, dn) = (3usize, 5, 4);
        let value = |seed: usize| (seed * 29 % 7) as i32 - 3;
        let a: Vec<i32> = (0..dm * dk).map(value).collect();
        let b: Vec<i32> = (0..dk * dn).map(|i| value(i + 3)).collect();
        assert!(a.contains(&0) && b.contains(&0));
        let (a, b) = (t(&a, &[dm, dk]), t(&b, &[dk, dn]));
        let acc = einsum_acc_i64(&eq, &[&a, &b]).unwrap();
        assert_eq!(acc.dims(), &[dm, dn]);
        for (flat, &got) in acc.data().iter().enumerate() {
            let (m, n) = (flat / dn, flat % dn);
            let want: i64 = (0..dk)
                .map(|k| a.data()[m * dk + k] as i64 * b.data()[k * dn + n] as i64)
                .sum();
            assert_eq!(got, want, "entry {flat}");
        }
    }

    /// A matrix product against the sum written out, with a row of zero scales: one whose
    /// columns span several tiles, the last one partial (many output rows make a tile
    /// narrow); one whose summed axis splits into several parts, the last one short of four
    /// rows; and the one-column and one-row products.
    #[test]
    fn tiled_matmul_matches_the_sum() {
        let eq: EinsumEq = "mk,kn->mn".parse().unwrap();
        let many_rows = TILE_BYTES / (MIN_TILE * size_of::<i64>()) + 1;
        for (dm, dk, dn) in [
            (many_rows, 6usize, 2 * MIN_TILE + 17),
            (2, 4 * MIN_PART + 3, 9),
            (4, 3, 1),
            (1, 7, MIN_TILE),
        ] {
            let value = |seed: usize| (seed * 7919 % 2003) as i32 - 1001;
            let mut a: Vec<i32> = (0..dm * dk).map(value).collect();
            a[..dk].fill(0);
            let b: Vec<i32> = (0..dk * dn).map(|i| value(i + 11)).collect();
            let (a, b) = (t(&a, &[dm, dk]), t(&b, &[dk, dn]));
            let acc = einsum_acc_i64(&eq, &[&a, &b]).unwrap();
            assert_eq!(acc.dims(), &[dm, dn]);
            for (flat, &got) in acc.data().iter().enumerate() {
                let (m, n) = (flat / dn, flat % dn);
                let want: i64 = (0..dk)
                    .map(|k| a.data()[m * dk + k] as i64 * b.data()[k * dn + n] as i64)
                    .sum();
                assert_eq!(got, want, "{dm}x{dk}x{dn}, entry {flat}");
            }
        }
    }

    #[test]
    fn batched_contraction_shape() {
        let eq: EinsumEq = "abmk,abnk->abmn".parse().unwrap();
        let op = Einsum { equation: eq };
        let shape = op.infer_shape(&[&[2, 3, 4, 5], &[2, 3, 6, 5]]).unwrap();
        assert_eq!(shape, vec![2, 3, 4, 6]);
        // Conflicting letter size errors out.
        assert!(op.infer_shape(&[&[2, 3, 4, 5], &[2, 3, 6, 7]]).is_err());
    }

    #[test]
    fn dot_product_scalar_output() {
        let eq: EinsumEq = "k,k->".parse().unwrap();
        let op = Einsum { equation: eq };
        assert_eq!(op.infer_shape(&[&[8], &[8]]).unwrap(), vec![1]);
    }
}
