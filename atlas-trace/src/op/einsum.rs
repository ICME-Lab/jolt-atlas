//! The fused einsum contraction and its parsed equation type.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    error::{KernelError, ShapeError},
    id::Scale,
    tensor::Tensor,
};

use super::{BoundCtx, Einsum, Eval, EvalCtx, OpSpec, PadBehavior, fused::rebase_eval};

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

/// Row-major strides for `dims`.
fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

/// Raw i64 contraction `acc[out] = Σ_sum Π_i operand_i[...]`.
fn einsum_acc_i64(eq: &EinsumEq, inputs: &[&Tensor<i32>]) -> Result<Tensor<i64>, ShapeError> {
    let dims: Vec<&[usize]> = inputs.iter().map(|t| t.dims()).collect();
    let sizes = eq.letter_sizes(&dims)?;

    let mut out_shape: Vec<usize> = eq.output.iter().map(|c| sizes[c]).collect();
    if out_shape.is_empty() {
        out_shape.push(1);
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
        Ok(rebase_eval(
            acc,
            self.rebase_divisor(ctx.scale).expect("fused"),
        ))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        let sizes = self.equation.letter_sizes(inputs)?;
        let mut shape: Vec<usize> = self.equation.output().iter().map(|c| sizes[c]).collect();
        if shape.is_empty() {
            shape.push(1);
        }
        Ok(shape)
    }

    fn clamp_bound(&self, ctx: &BoundCtx<'_>) -> Option<u128> {
        Some(einsum_acc_bound(&self.equation, ctx) >> ctx.scale.log2())
    }

    fn padding(&self) -> PadBehavior {
        // Zero-padded weight lanes contribute zero to every contraction.
        PadBehavior::ZeroPreserving
    }

    fn rebase_divisor(&self, scale: Scale) -> Option<i64> {
        Some(scale.denom_i64())
    }
}

/// Bound on `|Σ Π operands|`: `2^31 · max_row Σ|w|` when an operand is a
/// constant (weights are known exactly), else `2^62 · contracted size`.
/// Ported verbatim from the old crate's `einsum_acc_bound`.
fn einsum_acc_bound(eq: &EinsumEq, ctx: &BoundCtx<'_>) -> u128 {
    const I32_MAG: u128 = 1 << 31;

    // Prefer a constant operand: its row sums bound the contraction exactly.
    let mut best: Option<u128> = None;
    for (i, maybe_const) in ctx.input_consts.iter().enumerate() {
        if let Some(w) = maybe_const
            && let Some(m) = const_operand_row_sum_max(w, &eq.inputs[i], eq.output())
        {
            let b = I32_MAG * m;
            best = Some(best.map_or(b, |x: u128| x.min(b)));
        }
    }
    if let Some(b) = best {
        return b;
    }

    // All operands dynamic: each product is at most 2^62, summed over the
    // contracted index space.
    let mut sizes: HashMap<u8, usize> = HashMap::new();
    for (letters, dims) in eq.inputs.iter().zip(&ctx.input_dims) {
        for (j, &c) in letters.iter().enumerate() {
            if let Some(&d) = dims.get(j) {
                sizes.insert(c, d);
            }
        }
    }
    let contracted: u128 = sizes
        .iter()
        .filter(|(c, _)| !eq.output.contains(c))
        .map(|(_, &d)| d as u128)
        .product::<u128>()
        .max(1);
    (1u128 << 62) * contracted
}

/// `max` over the operand's free indices of `Σ` over its contracted indices
/// of `|w|`, for a constant operand `w` with letters `letters`.
fn const_operand_row_sum_max(w: &Tensor<i32>, letters: &[u8], output: &[u8]) -> Option<u128> {
    let dims = w.dims();
    if letters.len() != dims.len() {
        return None;
    }
    let free: Vec<usize> = (0..letters.len())
        .filter(|&i| output.contains(&letters[i]))
        .collect();
    let free_size: usize = free.iter().map(|&i| dims[i]).product::<usize>().max(1);
    let mut sums = vec![0u128; free_size];
    let mut coord = vec![0usize; dims.len()];
    for &v in w.data() {
        let mut key = 0usize;
        for &f in &free {
            key = key * dims[f] + coord[f];
        }
        sums[key] += (v as i64).unsigned_abs() as u128;
        for d in (0..dims.len()).rev() {
            coord[d] += 1;
            if coord[d] < dims[d] {
                break;
            }
            coord[d] = 0;
        }
    }
    sums.into_iter().max()
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
