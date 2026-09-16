//! Serial BN254 MSM with affine bucket sums and shared inversions.
//!
//! Points must be valid curve points, as required by Arkworks MSM. Equal
//! points, inverse points and the identity are handled explicitly.
use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::AffineRepr;
use ark_ff::{AdditiveGroup, Field, PrimeField, Zero};

fn sum_buckets(buckets: &mut [Vec<G1Affine>]) {
    loop {
        let pairs: usize = buckets.iter().map(|bucket| bucket.len() / 2).sum();
        if pairs == 0 {
            return;
        }
        let mut denominators = Vec::with_capacity(pairs);
        for bucket in buckets.iter() {
            for pair in bucket.chunks_exact(2) {
                let (p, q) = (pair[0], pair[1]);
                denominators.push(if p.is_zero() || q.is_zero() {
                    Fq::zero()
                } else if p.x == q.x {
                    if p.y == q.y {
                        p.y.double()
                    } else {
                        Fq::zero()
                    }
                } else {
                    q.x - p.x
                });
            }
        }
        ark_ff::batch_inversion(&mut denominators);
        let mut inverses = denominators.into_iter();
        for bucket in buckets.iter_mut() {
            let pairs = bucket.len() / 2;
            for index in 0..pairs {
                let (p, q) = (bucket[2 * index], bucket[2 * index + 1]);
                let inverse = inverses.next().unwrap();
                bucket[index] = if p.is_zero() {
                    q
                } else if q.is_zero() {
                    p
                } else if inverse.is_zero() {
                    G1Affine::identity()
                } else {
                    let numerator = if p.x == q.x {
                        let square = p.x.square();
                        square.double() + square
                    } else {
                        q.y - p.y
                    };
                    let slope = numerator * inverse;
                    let x = slope.square() - p.x - q.x;
                    let y = slope * (p.x - x) - p.y;
                    G1Affine::new_unchecked(x, y)
                };
            }
            let odd = bucket.len() % 2;
            if odd != 0 {
                bucket[pairs] = *bucket.last().unwrap();
            }
            bucket.truncate(pairs + odd);
        }
    }
}

// For pairs h[j] = p[2j] + p[2j+1], the weighted sum satisfies
// W(p) = 2 W(h) - sum_j p[2j]. Batch each level of pair additions and
// all sums of even positions, leaving only logarithmically many
// projective additions when combining the levels.
fn weighted_sum(buckets: &[Vec<G1Affine>]) -> G1Projective {
    let mut points: Vec<_> = buckets
        .iter()
        .map(|bucket| bucket.first().copied().unwrap_or_else(G1Affine::identity))
        .collect();
    assert!(points.len().is_power_of_two());
    let mut even_sums = Vec::new();
    while points.len() > 1 {
        even_sums.push(points.iter().step_by(2).copied().collect());
        let mut pairs: Vec<Vec<_>> = points.chunks_exact(2).map(|pair| pair.to_vec()).collect();
        sum_buckets(&mut pairs);
        points = pairs.into_iter().map(|pair| pair[0]).collect();
    }
    sum_buckets(&mut even_sums);
    let mut result = points[0].into_group();
    for sum in even_sums.iter().rev() {
        result.double_in_place();
        result -= sum[0];
    }
    result
}

/// Compute the exact sum of scalar multiples. This serial path groups
/// additions into batches so each batch shares one field inversion.
pub fn msm(bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    assert_eq!(bases.len(), scalars.len());
    if bases.is_empty() {
        return G1Projective::zero();
    }
    if bases.len() < 128 {
        return <G1Projective as ark_ec::VariableBaseMSM>::msm_unchecked(bases, scalars);
    }
    let width = if bases.len() < 256 {
        7
    } else if bases.len() < 4096 {
        8
    } else {
        10
    };
    let windows = 256usize.div_ceil(width);
    let radix = 1u64 << width;
    let mut digits = Vec::with_capacity(bases.len() * windows);
    for scalar in scalars {
        let limbs = scalar.into_bigint().0;
        let mut carry = 0;
        for window in 0..windows {
            let bit = window * width;
            let limb = bit / 64;
            let offset = bit % 64;
            let mut bits = limbs[limb] >> offset;
            if offset + width > 64 && limb + 1 < limbs.len() {
                bits |= limbs[limb + 1] << (64 - offset);
            }
            let value = (bits & (radix - 1)) + carry;
            carry = (value + radix / 2) >> width;
            digits.push((value as i64 - (carry * radix) as i64) as i16);
        }
        // BN254 scalars have at most 254 bits, leaving room for the carry.
        assert_eq!(carry, 0);
    }
    let mut result = G1Projective::zero();
    for window in (0..windows).rev() {
        for _ in 0..width {
            result.double_in_place();
        }
        let mut buckets = vec![Vec::new(); (radix / 2) as usize];
        for (index, base) in bases.iter().enumerate() {
            let digit = digits[index * windows + window];
            if digit != 0 && !base.is_zero() {
                buckets[digit.unsigned_abs() as usize - 1].push(if digit < 0 {
                    -*base
                } else {
                    *base
                });
            }
        }
        sum_buckets(&mut buckets);
        result += weighted_sum(&buckets);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::{CurveGroup, PrimeGroup, VariableBaseMSM};
    use ark_std::{test_rng, UniformRand};

    #[test]
    fn bucket_sums_include_exceptional_pairs() {
        let mut rng = test_rng();
        let p = G1Affine::rand(&mut rng);
        let q = G1Affine::rand(&mut rng);
        let zero = G1Affine::identity();
        let mut sets = vec![
            vec![],
            vec![zero],
            vec![p, p],
            vec![p, -p],
            vec![zero, p],
            vec![p, zero],
            vec![p, q, -p, -q, p],
            vec![zero, zero],
            vec![p; 17],
        ];
        let expected: Vec<G1Projective> =
            sets.iter().map(|set| set.iter().copied().sum()).collect();
        sum_buckets(&mut sets);
        for (set, expected) in sets.iter().zip(expected) {
            assert_eq!(set.first().copied().unwrap_or(zero), expected.into_affine());
        }
    }

    #[test]
    fn weighted_sum_matches_scalar_multiplication() {
        let mut rng = test_rng();
        for size in [1, 2, 4, 16, 128, 512] {
            let p = G1Affine::rand(&mut rng);
            let mut buckets: Vec<Vec<_>> = (0..size)
                .map(|i| match i % 5 {
                    0 => vec![],
                    1 => vec![p],
                    2 => vec![-p],
                    _ => vec![G1Affine::rand(&mut rng)],
                })
                .collect();
            let expected: G1Projective = buckets
                .iter()
                .enumerate()
                .map(|(i, bucket)| {
                    bucket
                        .first()
                        .copied()
                        .unwrap_or_else(G1Affine::identity)
                        .mul_bigint([(i + 1) as u64])
                })
                .sum();
            assert_eq!(weighted_sum(&buckets), expected);
            buckets.iter_mut().for_each(Vec::clear);
            assert!(weighted_sum(&buckets).is_zero());
        }
    }

    #[test]
    fn msm_with_repeated_bases() {
        let p = G1Projective::generator().into_affine();
        let bases = vec![p; 256];
        let scalars: Vec<_> = (0..256)
            .map(|i| {
                [
                    Fr::from(0u64),
                    Fr::from(1u64),
                    -Fr::from(1u64),
                    Fr::from(i as u64),
                ][i % 4]
            })
            .collect();
        let sum: Fr = scalars.iter().copied().sum();
        assert_eq!(msm(&bases, &scalars), p.mul_bigint(sum.into_bigint()));
        assert!(msm(&bases, &vec![Fr::from(0u64); 256]).is_zero());
    }

    #[test]
    fn msm_matches_arkworks() {
        let mut rng = test_rng();
        for size in [0, 1, 2, 17, 63, 128, 255, 256, 1024, 4096, 6950] {
            let p = G1Projective::generator().into_affine();
            let mut bases: Vec<_> = (0..size).map(|_| G1Affine::rand(&mut rng)).collect();
            let mut scalars: Vec<_> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
            for i in 0..size.min(12) {
                bases[i] = [p, -p, G1Affine::identity()][i % 3];
                scalars[i] = [Fr::from(0), Fr::from(1), -Fr::from(1), Fr::from(u64::MAX)][i % 4];
            }
            assert_eq!(
                msm(&bases, &scalars),
                G1Projective::msm(&bases, &scalars).unwrap()
            );
        }
    }
}
