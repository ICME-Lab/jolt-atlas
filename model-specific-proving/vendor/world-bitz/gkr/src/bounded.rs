//! Evaluate early sumcheck rounds from the packed table and materialize only the folded suffix.

use super::*;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[path = "metal.rs"]
mod metal;

/// At most 256 MiB for the two dense child buffers in one relation.
pub(super) const DENSE_NODES: usize = 1 << 24;
const TILE: usize = 1 << 12;

impl LutLayers {
    pub(super) fn nibble(&self, index: usize) -> usize {
        ((self.words()[index / NIBBLES_IN_WORD] >> (NIBBLE_BITS * (index % NIBBLES_IN_WORD))) & 15)
            as usize
    }

    fn child(&self, level: usize, index: usize) -> Field {
        if level >= 3 {
            let n = self.nodes::<LAYER2>() / 2;
            let len = n >> (level - 3);
            return (0..n / len)
                .map(|i| {
                    let low = index + i * len;
                    let high = low + n;
                    self.luts[low >> self.log_trees][self.nibble(low)]
                        * self.luts[high >> self.log_trees][self.nibble(high)]
                })
                .reduce(|a, b| a * b)
                .expect("a tree node has children");
        }
        let nodes = self.nodes::<LAYER2>() << (LAYER2 - level);
        let stride = nodes / 4;
        let part = index / stride;
        let position = index % stride + STREAM[level][part] * stride;
        self.luts[position >> self.log_trees][self.nibble(position) & MASK[level][part]]
    }
}

pub(super) fn build(
    row_images: &[Field],
    table: &BitTable,
    dense_nodes: usize,
    include_output: bool,
) -> (Vec<Field>, LayerWitnesses) {
    let lut_layers = LutLayers::new(table.as_matrix(), row_images);
    let n = lut_layers.nodes::<LAYER2>() / 2;
    let columns = table.shape().columns();
    let mut count = n;
    let mut level = 3;
    while count > dense_nodes && count > columns {
        count /= 2;
        level += 1;
    }

    let first = if level == 3 {
        layer3(&lut_layers)
    } else {
        (0..count)
            .into_par_iter()
            .with_min_len(PARALLEL_MIN_LANES)
            .map(|i| lut_layers.child(level, i))
            .collect()
    };
    let (output, upper_nodes) = fold_layers(first, columns, include_output);
    (
        output,
        LayerWitnesses {
            lut_layers,
            upper_nodes,
            first_level: level,
        },
    )
}

pub(super) fn prove(
    ps: &mut ProverState,
    mut point: Vec<Field>,
    witness: LayerWitnesses,
    security: SecurityLevel,
    dense_nodes: usize,
) -> (Vec<Field>, Field) {
    let LayerWitnesses {
        lut_layers,
        upper_nodes,
        first_level,
    } = witness;
    for mut children in upper_nodes.into_iter().rev() {
        let half = children.len() / 2;
        let (left, right) = children.split_at_mut(half);
        (point, _) = prove_relation(
            ps,
            &point,
            security,
            dense_nodes,
            Some((left, right)),
            None,
            |_| unreachable!("dense children supplied"),
        );
    }
    let mut value = Field::ZERO;
    for level in (0..first_level).rev() {
        (point, value) = prove_relation(
            ps,
            &point,
            security,
            dense_nodes,
            None,
            (level <= 2).then_some((&lut_layers, level)),
            |index| lut_layers.child(level, index),
        );
    }
    (point, value)
}

fn round_sum(
    precomputed: Option<(Wide256, Wide256)>,
    point: &[Field],
    cpu: impl Fn(Range<usize>, &[Field], &mut Vec<Field>) -> (Wide256, Wide256) + Sync,
) -> (Wide256, Wide256) {
    precomputed.unwrap_or_else(|| factored_sum(point, cpu))
}

fn factored_sum(
    point: &[Field],
    work: impl Fn(Range<usize>, &[Field], &mut Vec<Field>) -> (Wide256, Wide256) + Sync,
) -> (Wide256, Wide256) {
    let low_vars = point.len().min(TILE.ilog2() as usize);
    let (low, high) = point.split_at(low_vars);
    let lo = poly::eq_table(low);
    let hi = poly::eq_table(high);
    hi.par_iter()
        .enumerate()
        .map_init(Vec::new, |scratch, (i, &weight)| {
            let start = i * lo.len();
            let (b, inf) = work(start..start + lo.len(), &lo, scratch);
            if hi.len() == 1 {
                (b, inf)
            } else {
                (
                    Wide256::mul(b.reduce(), weight),
                    Wide256::mul(inf.reduce(), weight),
                )
            }
        })
        .reduce(
            || (Wide256::zero(), Wide256::zero()),
            |(a, b), (c, d)| (a + c, b + d),
        )
}

fn send_round(
    ps: &mut ProverState,
    eq_bound: &mut Field,
    sum: (Wide256, Wide256),
    z: Field,
    security: SecurityLevel,
) -> Field {
    let (b, inf) = sum;
    ps.prover_message(&[*eq_bound * b.reduce(), *eq_bound * inf.reduce()]);
    ps.grind(CUBIC_GRINDING_LABEL, security.grinding_bits(3));
    let r = ps.verifier_message();
    *eq_bound *= eq_factor(r, z);
    r
}

fn prove_relation(
    ps: &mut ProverState,
    point: &[Field],
    security: SecurityLevel,
    dense_nodes: usize,
    dense: Option<(&mut [Field], &mut [Field])>,
    source_lut: Option<(&LutLayers, usize)>,
    source: impl Fn(usize) -> Field + Sync,
) -> (Vec<Field>, Field) {
    let n = point.len();
    let mut challenges = vec![Field::ZERO; n + 1];
    let mut eq_bound = Field::ONE;
    let mut remaining = n;

    let mut left_owned;
    let mut right_owned;
    let (mut left, mut right) = match dense {
        Some(pair) => pair,
        None => {
            let side = 1 << n;
            let mut prefix = vec![Field::ONE];
            let bound = |side_start, local: usize, stride, weights: &[Field]| {
                if weights.len() == 1 {
                    return source(side_start + local);
                }
                let mut sum = Wide256::zero();
                for (i, &weight) in weights.iter().enumerate() {
                    sum += Wide256::mul(source(side_start + i * stride + local), weight);
                }
                sum.reduce()
            };
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            let gpu = source_lut.and_then(|(lut, level)| {
                if level == LEAVES && side > dense_nodes {
                    metal::Leaves::new(lut, point)
                } else {
                    None
                }
            });
            while (2 << remaining) > dense_nodes {
                let i = remaining - 1;
                let stride = 1 << remaining;
                let half = 1 << i;
                let at_one = sends_one(point[i]);
                // After binding one coordinate, a pair of nibbles determines both children.
                let paired = source_lut.and_then(|(lut, level)| {
                    (level == LEAVES && prefix.len() == 2 && half >= NIBBLES_IN_WORD).then_some(lut)
                });
                let projected = source_lut.filter(|_| paired.is_none()).map(|(lut, level)| {
                    packed_round::PackedRound::new(lut, level, stride, &prefix)
                });
                #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
                let gpu_sum = None;
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                let gpu_sum = gpu.as_ref().and_then(|leaves| match prefix.len() {
                    1 => Some(leaves.first(at_one)),
                    2 => Some(leaves.second(challenges[i + 1], at_one)),
                    _ => None,
                });
                let mut sums = round_sum(gpu_sum, &point[..i], |range, eq, scratch| {
                    if let Some(lut) = paired {
                        let group = 1 << lut.log_trees;
                        let mut sums = (Wide256::zero(), Wide256::zero());
                        for (block, weights) in eq.chunks(group).enumerate() {
                            let start = range.start + block * group;
                            let (b, inf) = leaf_pair_sums(
                                lut,
                                start,
                                weights,
                                half,
                                challenges[i + 1],
                                at_one,
                            );
                            sums.0 += b;
                            sums.1 += inf;
                        }
                        return sums;
                    }
                    if let Some(projected) = &projected {
                        scratch.resize(4 * range.len(), Field::ZERO);
                        let mut parts = scratch.chunks_exact_mut(range.len());
                        let [l0, l1, r0, r1] = std::array::from_fn(|_| parts.next().unwrap());
                        for (start, out) in [
                            (0, l0.as_mut()),
                            (half, l1.as_mut()),
                            (side, r0.as_mut()),
                            (side + half, r1.as_mut()),
                        ] {
                            projected.fill(start + range.start, out);
                        }
                        let mut b = Wide256::zero();
                        let mut inf = Wide256::zero();
                        for j in 0..range.len() {
                            let (l, r) = if at_one {
                                (l1[j], r1[j])
                            } else {
                                (l0[j], r0[j])
                            };
                            if l != Field::ZERO || r != Field::ZERO {
                                b += Wide256::mul(eq[j], l + r + l * r);
                            }
                            let dl = l1[j] + l0[j];
                            let dr = r1[j] + r0[j];
                            if dl != Field::ZERO && dr != Field::ZERO {
                                inf += Wide256::mul(eq[j] * dl, dr);
                            }
                        }
                        return (b, inf);
                    }

                    let mut b = Wide256::zero();
                    let mut inf = Wide256::zero();
                    for (j, &weight) in range.zip(eq) {
                        let l0 = bound(0, j, stride, &prefix);
                        let l1 = bound(0, j + half, stride, &prefix);
                        let r0 = bound(side, j, stride, &prefix);
                        let r1 = bound(side, j + half, stride, &prefix);
                        let (left, right) = if at_one { (l1, r1) } else { (l0, r0) };
                        b += Wide256::mul(weight * left, right);
                        inf += Wide256::mul(weight * (l1 - l0), r1 - r0);
                    }
                    (b, inf)
                });
                if projected.is_some() {
                    sums.0 += Wide256::mul(Field::ONE, Field::ONE);
                }
                let r = send_round(ps, &mut eq_bound, sums, point[i], security);
                challenges[i] = r;
                prefix = prefix
                    .into_iter()
                    .flat_map(|value| {
                        let high = value * r;
                        [value - high, high]
                    })
                    .collect();
                remaining -= 1;
            }
            let len = 1 << remaining;
            let projected = source_lut
                .map(|(lut, level)| packed_round::PackedRound::new(lut, level, len, &prefix));
            if let Some(projected) = projected {
                let materialize = |start| {
                    let mut values = Field::zeroed_vec(len);
                    values
                        .par_chunks_mut(TILE)
                        .enumerate()
                        .for_each(|(i, chunk)| {
                            projected.fill(start + i * TILE, chunk);
                            for v in chunk {
                                *v += Field::ONE;
                            }
                        });
                    values
                };
                (left_owned, right_owned) = rayon::join(|| materialize(0), || materialize(side));
            } else {
                (left_owned, right_owned) = rayon::join(
                    || {
                        (0..len)
                            .into_par_iter()
                            .with_min_len(TILE)
                            .map(|j| bound(0, j, len, &prefix))
                            .collect::<Vec<_>>()
                    },
                    || {
                        (0..len)
                            .into_par_iter()
                            .with_min_len(TILE)
                            .map(|j| bound(side, j, len, &prefix))
                            .collect::<Vec<_>>()
                    },
                );
            }
            (left_owned.as_mut_slice(), right_owned.as_mut_slice())
        }
    };

    #[cfg(feature = "gkr-backend")]
    let mut backend = DENSE_BACKEND.get().and_then(|run| run(left, right));
    #[cfg(feature = "gkr-backend")]
    let mut previous = None;
    for i in (0..remaining).rev() {
        #[cfg(feature = "gkr-backend")]
        if let Some(gpu) = &mut backend {
            let (low, high) = point[..i].split_at(i.min(TILE.ilog2() as usize));
            if let Some([value, infinity]) = gpu.step(previous, sends_one(point[i]),
                &poly::eq_table(low), &poly::eq_table(high))
            {
                let sums = (Wide256::mul(value, Field::ONE), Wide256::mul(infinity, Field::ONE));
                let r = send_round(ps, &mut eq_bound, sums, point[i], security);
                challenges[i] = r;
                previous = Some(r);
                continue;
            }
            if let Some(r) = previous {
                // The last successful GPU step holds the state before this bind.
                let mid = 2 << i;
                left = &mut left[..2 * mid];
                right = &mut right[..2 * mid];
                gpu.snapshot(left, right);
                let (l0,l1) = left.split_at_mut(mid);
                let (r0,r1) = right.split_at_mut(mid);
                bind(r,l0,l1,r0,r1);
                (left,right) = (l0,r0);
            }
            backend = None;
        }
        let half = 1 << i;
        let (l0, l1) = left.split_at_mut(half);
        let (r0, r1) = right.split_at_mut(half);
        let sums = factored_sum(&point[..i], |range, weights, _| {
            if sends_one(point[i]) {
                round_sums::<true>(
                    &l0[range.clone()],
                    &l1[range.clone()],
                    &r0[range.clone()],
                    &r1[range],
                    weights,
                )
            } else {
                round_sums::<false>(
                    &l0[range.clone()],
                    &l1[range.clone()],
                    &r0[range.clone()],
                    &r1[range],
                    weights,
                )
            }
        });
        let r = send_round(ps, &mut eq_bound, sums, point[i], security);
        challenges[i] = r;
        bind(r, l0, l1, r0, r1);
        (left, right) = (l0, r0);
    }

    ps.prover_message(&[left[0], right[0]]);
    let t = ps.verifier_message();
    challenges[n] = t;
    (challenges, interpolate(left[0], right[0], t))
}

/// Round two pairs nibbles from two row groups. Only the eq weights depend on the column.
fn leaf_pair_sums(
    lut: &LutLayers,
    start: usize,
    eq: &[Field],
    half: usize,
    first: Field,
    at_one: bool,
) -> (Wide256, Wide256) {
    let mut weights = [Field::ZERO; 256];
    let words = lut.words();
    let start = start / NIBBLES_IN_WORD;
    let stride = half / NIBBLES_IN_WORD;
    for (i, lanes) in eq.chunks_exact(NIBBLES_IN_WORD).enumerate() {
        let mut a = words[start + i];
        let mut b = words[start + i + stride];
        for &weight in lanes {
            weights[((a & 15) | ((b & 15) << 4)) as usize] += weight;
            a >>= 4;
            b >>= 4;
        }
    }
    leaf_pair_eval(lut, start * NIBBLES_IN_WORD, half, first, at_one, &weights)
}

fn leaf_pair_eval(
    lut: &LutLayers,
    start: usize,
    half: usize,
    first: Field,
    at_one: bool,
    weights: &[Field; 256],
) -> (Wide256, Wide256) {
    let children = [start, start + half].map(|start| {
        let images = &lut.luts[start >> lut.log_trees];
        let left: [Field; 4] =
            std::array::from_fn(|k| interpolate(images[k & 1], images[k & 2], first));
        let right: [Field; 4] =
            std::array::from_fn(|k| interpolate(images[(k & 1) << 2], images[(k & 2) << 2], first));
        (left, right)
    });
    let mut sums = (Wide256::zero(), Wide256::zero());
    for (key, w) in weights.iter().copied().enumerate() {
        if w == Field::ZERO {
            continue;
        }
        let [(l0, r0), (l1, r1)] = std::array::from_fn(|i| {
            let k = key >> (4 * i);
            (children[i].0[k & 3], children[i].1[(k >> 2) & 3])
        });
        let (l, r) = if at_one { (l1, r1) } else { (l0, r0) };
        sums.0 += Wide256::mul(w, l * r);
        sums.1 += Wide256::mul(w, (l0 + l1) * (r0 + r1));
    }
    sums
}
