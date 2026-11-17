//! BitZ as the proof's commitment scheme: every committed polynomial is the bit planes of one
//! bit vector, committed once and opened once.
//!
//! A polynomial with `2^n` coefficients of `B` bits is `B` planes of `2^n` bits; a signed one
//! is committed shifted by `2^(B − 1)`. With `2^t` rows in the bit matrix, a plane is placed
//! at a row-aligned offset and occupies `2^max(n, t)` bits, zero above its own `2^n`: whole
//! columns of the matrix, the plane's low `t` index bits being the row.
//!
//! The joint opening is of `Σ_p γ_p · f_p` at one point `r`, each `f_p` zero-padded to the
//! point's length, so `f_p` contributes `γ_p · pad_p · f_p(r_suffix)`. In terms of the bits
//! that is one inner product with a weight that is a tensor: the `eq` table of the point's
//! last `t` coordinates over the rows, times a weight per column, which on a plane's columns
//! is `γ_p · 2^bit · pad_p · eq` of the coordinates between the plane's own and the rows'.
//! That is the claim BitZ proves, with no evaluation of any single polynomial sent. Bits a
//! prover sets in a plane's padding change the polynomial that is opened, exactly as nonzero
//! coefficients above `2^n` would: the opened combination is of what is committed.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::unimplemented,
    clippy::todo,
    clippy::indexing_slicing
)]

use super::merkle::Digest;
use super::params::{Params, MAX_VARS, MIN_VARS};
use super::proof::{eq_table, Commitment, Proof};
use super::verifier;
use crate::commitment_scheme::{BatchCommitOutput, CommitmentScheme};
use atlas_algebra::field::fp128::Fp128;
use atlas_algebra::poly::multilinear::MultilinearPolynomial;
use atlas_algebra::serialize::{Canonical, Error};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_transcript::{ProvingTranscript, VerifyingTranscript};
use common::impl_canonical_struct;
use std::io::{Read, Write};

mod prover;

/// The security target of every opening.
pub const TARGET_BITS: u32 = 128;

/// The fewest rows of the bit matrix, as a power of two: a row index covers a packed element.
const MIN_LOG_ROWS: usize = 7;

/// The label of the transcript an opening runs on.
const OPENING_LABEL: &[u8] = b"atlas-bitz-opening";

/// How a polynomial is committed: its variables, the bits of a coefficient, and whether the
/// coefficients are shifted by `2^(bits − 1)` to be nonnegative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub num_vars: u8,
    pub bits: u8,
    pub signed: bool,
}

impl_canonical_struct!([] Shape { num_vars, bits, signed });

/// The commitment to a batch: the root, and each polynomial's shape in slot order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitzBatchCommitment {
    pub root: Digest,
    pub shapes: Vec<Shape>,
}

impl_canonical_struct!([] BitzBatchCommitment { root, shapes });

/// A polynomial's slot in its batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BitzSlot(pub u32);

impl Canonical for BitzSlot {
    fn encode<W: Write>(&self, writer: W) -> Result<(), Error> {
        self.0.encode(writer)
    }

    fn encoded_len(&self) -> usize {
        self.0.encoded_len()
    }

    fn decode<R: Read>(reader: R) -> Result<Self, Error> {
        Ok(Self(u32::decode(reader)?))
    }
}

/// A joint opening: the BitZ proof and the prover messages of the transcript it ran on, its
/// proof-of-work nonces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitzOpening {
    pub proof: Proof,
    pub narg: Vec<u8>,
}

impl_canonical_struct!([] BitzOpening { proof, narg });

/// Where a plane sits: its first column and `log2` of its column count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Placement {
    first_column: usize,
    log_columns: usize,
}

/// The placement of every plane of a batch in the bit matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    log_rows: usize,
    log_columns: usize,
    /// `planes[slot][bit]`.
    planes: Vec<Vec<Placement>>,
}

impl Layout {
    /// The layout of a batch of shapes, or `None` if a shape is malformed or the batch does
    /// not fit a commitment. Planes are placed largest first, each at the next free column, so
    /// every plane is aligned to its own size.
    pub fn new(shapes: &[Shape]) -> Option<Self> {
        let mut total_bits = 0u128;
        for shape in shapes {
            if shape.num_vars as usize > MAX_VARS || !(1..=64).contains(&shape.bits) {
                return None;
            }
            total_bits += (shape.bits as u128) << shape.num_vars;
        }
        if total_bits == 0 {
            return None;
        }
        let log_total = (128 - (total_bits - 1).leading_zeros()) as usize;
        let log_rows = log_total.max(MIN_VARS).div_ceil(2).max(MIN_LOG_ROWS);

        let mut order: Vec<(usize, usize, usize)> = shapes
            .iter()
            .enumerate()
            .flat_map(|(slot, shape)| {
                let log_columns = (shape.num_vars as usize).saturating_sub(log_rows);
                (0..shape.bits as usize).map(move |bit| (log_columns, slot, bit))
            })
            .collect();
        order.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

        let mut planes: Vec<Vec<Placement>> = shapes
            .iter()
            .map(|shape| {
                vec![
                    Placement {
                        first_column: 0,
                        log_columns: 0
                    };
                    shape.bits as usize
                ]
            })
            .collect();
        let mut next_column = 0usize;
        for (log_columns, slot, bit) in order {
            let placement = planes.get_mut(slot)?.get_mut(bit)?;
            *placement = Placement {
                first_column: next_column,
                log_columns,
            };
            next_column = next_column.checked_add(1usize.checked_shl(log_columns as u32)?)?;
        }
        let log_columns = (usize::BITS - (next_column - 1).leading_zeros()) as usize;
        let log_columns = log_columns.max(MIN_VARS.saturating_sub(log_rows));
        if log_rows + log_columns > MAX_VARS {
            return None;
        }
        Some(Self {
            log_rows,
            log_columns,
            planes,
        })
    }

    /// The BitZ parameters of the layout's matrix.
    fn params(&self) -> Option<Params> {
        Params::for_shape(self.log_rows, self.log_columns, TARGET_BITS)
    }
}

/// The weights of a joint opening.
struct Weights {
    /// The row point, first coordinate for the lowest row bit.
    row_point: Vec<Fp128>,
    columns: Vec<Fp128>,
    /// What the shifts of the signed polynomials add to the joint claim.
    shift: Fp128,
}

/// The weights with which the bits of a batch combine to `Σ_i coeffs_i · f_{slots_i}` at
/// `point`, each polynomial taken at the last coordinates of the point and zero-padded.
fn opening_weights(
    layout: &Layout,
    shapes: &[Shape],
    slots: &[u32],
    coeffs: &[Fp128],
    point: &[Fp128],
) -> Result<Weights, ProofVerifyError> {
    let malformed = |what: &str| ProofVerifyError::Malformed(format!("BitZ opening: {what}"));
    if slots.len() != coeffs.len() {
        return Err(malformed("slot and coefficient counts differ"));
    }
    let (zero, one) = (Fp128::from(0u128), Fp128::from(1u128));
    let log_rows = layout.log_rows;
    // The point, extended with zeros on the high side to at least the row variables: a zero
    // coordinate selects the low half, where the polynomials are.
    let extended: Vec<Fp128> = std::iter::repeat_n(zero, log_rows.saturating_sub(point.len()))
        .chain(point.iter().copied())
        .collect();
    let len = extended.len();
    // `pads[k]` is the product of `1 − coordinate` over the first `k` coordinates: the factor
    // of a polynomial zero-padded above its own `len − k` variables.
    let mut pads = Vec::with_capacity(len + 1);
    pads.push(one);
    for (k, &coordinate) in extended.iter().enumerate() {
        let last = pads.get(k).copied().ok_or_else(|| malformed("point"))?;
        pads.push(last * (one - coordinate));
    }
    let pad = |num_vars: usize| -> Result<Fp128, ProofVerifyError> {
        len.checked_sub(num_vars)
            .and_then(|k| pads.get(k).copied())
            .ok_or_else(|| malformed("a polynomial has more variables than the point"))
    };
    let (high, rows) = extended
        .split_at_checked(len - log_rows)
        .ok_or_else(|| malformed("point"))?;

    let mut columns = vec![zero; 1usize << layout.log_columns];
    let mut shift = zero;
    for (&slot, &coeff) in slots.iter().zip(coeffs) {
        let shape = shapes
            .get(slot as usize)
            .ok_or_else(|| malformed("unknown slot"))?;
        let planes = layout
            .planes
            .get(slot as usize)
            .ok_or_else(|| malformed("unknown slot"))?;
        let num_vars = shape.num_vars as usize;
        if num_vars > point.len() {
            return Err(malformed("a polynomial has more variables than the point"));
        }
        let scale = coeff * pad(num_vars.max(log_rows))?;
        if shape.signed {
            shift += coeff * pad(num_vars)? * Fp128::from(1u128 << (shape.bits - 1));
        }
        // The plane's own coordinates above the rows, lowest index bit first.
        let own: Vec<Fp128> = high
            .get(high.len().saturating_sub(num_vars.saturating_sub(log_rows))..)
            .ok_or_else(|| malformed("point"))?
            .iter()
            .rev()
            .copied()
            .collect();
        let local = eq_table(&own);
        for (bit, placement) in planes.iter().enumerate() {
            let weight = scale * Fp128::from(1u128 << bit);
            let range = columns
                .get_mut(placement.first_column..)
                .ok_or_else(|| malformed("layout"))?;
            if local.len() != 1usize << placement.log_columns {
                return Err(malformed("layout"));
            }
            for (column, &entry) in range.iter_mut().zip(&local) {
                *column += weight * entry;
            }
        }
    }
    Ok(Weights {
        row_point: rows.iter().rev().copied().collect(),
        columns,
        shift,
    })
}

/// The seed of an opening's own transcript, drawn from the caller's.
fn opening_seed<T: atlas_transcript::Transcript>(transcript: &mut T) -> [u8; 32] {
    let mut seed = [0u8; 32];
    let (low, high) = seed.split_at_mut(16);
    low.copy_from_slice(&transcript.challenge_u128().to_le_bytes());
    high.copy_from_slice(&transcript.challenge_u128().to_le_bytes());
    seed
}

/// BitZ over Fp128.
#[derive(Clone, Debug)]
pub struct BitzScheme;

impl CommitmentScheme for BitzScheme {
    type Field = Fp128;
    type ProverSetup = ();
    type VerifierSetup = ();
    type Commitment = BitzSlot;
    type Proof = BitzOpening;
    type OpeningProofHint = prover::BitzHint;
    type BatchCommitment = BitzBatchCommitment;

    fn setup_prover(_max_num_vars: usize) -> Self::ProverSetup {}

    fn setup_verifier(_setup: &Self::ProverSetup) -> Self::VerifierSetup {}

    #[tracing::instrument(skip_all, name = "BitzScheme::batch_commit_all")]
    fn batch_commit_all<U>(polys: &[U], _setup: &Self::ProverSetup) -> BatchCommitOutput<Self>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<Fp128>> + Sync,
    {
        prover::commit(polys)
    }

    /// A polynomial is its bit planes: one committed without a shift has coefficients below
    /// two to its declared bits.
    fn coefficient_bits(
        batch_commitment: &Self::BatchCommitment,
        commitment: &Self::Commitment,
    ) -> Option<u32> {
        let shape = batch_commitment.shapes.get(commitment.0 as usize)?;
        (!shape.signed).then_some(shape.bits as u32)
    }

    #[tracing::instrument(skip_all, name = "BitzScheme::prove_rlc")]
    fn prove_rlc<T: ProvingTranscript>(
        _setup: &Self::ProverSetup,
        _polynomials: &std::collections::BTreeMap<
            common::CommittedPoly,
            MultilinearPolynomial<Fp128>,
        >,
        coeffs: &[Fp128],
        hints: Vec<Self::OpeningProofHint>,
        opening_point: &[Fp128],
        transcript: &mut T,
    ) -> Self::Proof {
        prover::open(coeffs, &hints, opening_point, transcript)
    }

    #[tracing::instrument(skip_all, name = "BitzScheme::verify_rlc")]
    fn verify_rlc<T: VerifyingTranscript>(
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_point: &[Fp128],
        joint_claim: &Fp128,
        commitments: &[&Self::Commitment],
        coeffs: &[Fp128],
        num_vars: &[usize],
        batch_commitment: &Self::BatchCommitment,
    ) -> Result<(), ProofVerifyError> {
        let rejected = |what: &str| ProofVerifyError::InvalidOpeningProof(format!("BitZ: {what}"));
        let shapes = &batch_commitment.shapes;
        let layout = Layout::new(shapes).ok_or_else(|| rejected("the batch has no layout"))?;
        let params = layout
            .params()
            .ok_or_else(|| rejected("the layout has no parameters"))?;
        if commitments.len() != num_vars.len() {
            return Err(rejected("commitment and variable counts differ"));
        }
        let slots: Vec<u32> = commitments.iter().map(|slot| slot.0).collect();
        for (&slot, &opened) in slots.iter().zip(num_vars) {
            let declared = shapes
                .get(slot as usize)
                .ok_or_else(|| rejected("unknown slot"))?;
            if declared.num_vars as usize != opened {
                return Err(rejected(
                    "a polynomial is opened at another size than committed",
                ));
            }
        }
        let weights = opening_weights(&layout, shapes, &slots, coeffs, opening_point)?;
        let claim = *joint_claim + weights.shift;

        let seed = opening_seed(transcript);
        let mut inner = T::new(OPENING_LABEL, proof.narg.clone());
        inner.append_bytes(&seed);
        verifier::verify(
            &Commitment {
                root: batch_commitment.root,
            },
            &weights.row_point,
            &weights.columns,
            &claim,
            &proof.proof,
            &params,
            &mut inner,
        )?;
        inner.finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]
mod tests {
    use super::*;
    use atlas_algebra::poly::multilinear::PolynomialEvaluation;
    use atlas_algebra::poly::one_hot::OneHotPolynomial;
    use atlas_transcript::{Blake2bProver, Blake2bVerifier, Transcript};
    use common::CommittedPoly;
    use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};
    use std::collections::BTreeMap;

    const LABEL: &[u8] = b"bitz-scheme-test";

    fn random_field(rng: &mut StdRng) -> Fp128 {
        Fp128::from(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128)
    }

    /// A batch of every committed kind at mixed sizes: base-4 digits, a wide unsigned column,
    /// signed columns, one-hot columns with a full and a short address, and polynomials
    /// smaller than a matrix column.
    fn batch(rng: &mut StdRng, log_len: usize) -> Vec<MultilinearPolynomial<Fp128>> {
        let len = 1usize << log_len;
        let digits = |rng: &mut StdRng, len: usize| -> MultilinearPolynomial<Fp128> {
            MultilinearPolynomial::from((0..len).map(|_| rng.gen_range(0..4u8)).collect::<Vec<_>>())
        };
        let one_hot = |rng: &mut StdRng, k: usize, cycles: usize| -> MultilinearPolynomial<Fp128> {
            let indices = (0..cycles)
                .map(|_| rng.gen_bool(0.9).then(|| rng.gen_range(0..k) as u16))
                .collect();
            MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(indices, k))
        };
        vec![
            digits(rng, len),
            digits(rng, len),
            digits(rng, len / 4),
            MultilinearPolynomial::from((0..64).map(|_| rng.next_u64()).collect::<Vec<u64>>()),
            MultilinearPolynomial::from(
                (0..32).map(|_| rng.next_u32() as i32).collect::<Vec<i32>>(),
            ),
            MultilinearPolynomial::from(vec![i32::MIN, -1, 0, 1, i32::MAX, 7, -7, 12]),
            one_hot(rng, 16, len / 16),
            one_hot(rng, 4, 8),
            digits(rng, 8),
            MultilinearPolynomial::from(vec![0u8; 16]),
        ]
    }

    struct Opened {
        batch: BitzBatchCommitment,
        slots: Vec<BitzSlot>,
        num_vars: Vec<usize>,
        coeffs: Vec<Fp128>,
        point: Vec<Fp128>,
        joint_claim: Fp128,
        proof: BitzOpening,
    }

    /// Commits to `polys` and opens the ones in `opened` with random coefficients at a random
    /// point of `point_len` coordinates. The joint claim is from each polynomial's own
    /// evaluation at the point's suffix and its zero-padding factor.
    fn open(
        polys: &[MultilinearPolynomial<Fp128>],
        opened: &[usize],
        point_len: usize,
        seed: u64,
    ) -> Opened {
        let mut rng = StdRng::seed_from_u64(seed);
        let (batch, per_poly) = BitzScheme::batch_commit_all(polys, &());
        let point: Vec<Fp128> = (0..point_len).map(|_| random_field(&mut rng)).collect();
        let coeffs: Vec<Fp128> = opened.iter().map(|_| random_field(&mut rng)).collect();
        let one = Fp128::from(1u128);
        let joint_claim =
            opened
                .iter()
                .zip(&coeffs)
                .fold(Fp128::from(0u128), |acc, (&i, &coeff)| {
                    let n = polys[i].get_num_vars();
                    let (padding, suffix) = point.split_at(point_len - n);
                    let pad = padding.iter().fold(one, |pad, &r| pad * (one - r));
                    let value = match &polys[i] {
                        MultilinearPolynomial::OneHot(poly) => poly.evaluate(suffix),
                        dense => dense.evaluate(suffix),
                    };
                    acc + coeff * pad * value
                });
        let map: BTreeMap<CommittedPoly, MultilinearPolynomial<Fp128>> = opened
            .iter()
            .map(|&i| (CommittedPoly::RsqrtQuotient(i), polys[i].clone()))
            .collect();
        let hints = opened.iter().map(|&i| per_poly[i].1.clone()).collect();
        let mut transcript = Blake2bProver::new(LABEL);
        transcript.append_u64(seed);
        let proof = BitzScheme::prove_rlc(&(), &map, &coeffs, hints, &point, &mut transcript);
        Opened {
            batch,
            slots: opened.iter().map(|&i| per_poly[i].0).collect(),
            num_vars: opened.iter().map(|&i| polys[i].get_num_vars()).collect(),
            coeffs,
            point,
            joint_claim,
            proof,
        }
    }

    fn check(opened: &Opened, seed: u64) -> Result<(), ProofVerifyError> {
        let mut transcript = Blake2bVerifier::new(LABEL, Vec::new());
        transcript.append_u64(seed);
        let slots: Vec<&BitzSlot> = opened.slots.iter().collect();
        BitzScheme::verify_rlc(
            &opened.proof,
            &(),
            &mut transcript,
            &opened.point,
            &opened.joint_claim,
            &slots,
            &opened.coeffs,
            &opened.num_vars,
            &opened.batch,
        )
    }

    #[test]
    fn joint_openings_of_mixed_batches_verify() {
        let mut rng = StdRng::seed_from_u64(1);
        for (log_len, point_len) in [(12, 12), (12, 15), (14, 14)] {
            let polys = batch(&mut rng, log_len);
            let all: Vec<usize> = (0..polys.len()).collect();
            let opened = open(&polys, &all, point_len, 3);
            check(&opened, 3).unwrap_or_else(|e| panic!("{log_len}, {point_len}: {e}"));
            // A subset, in another order.
            let opened = open(&polys, &[7, 0, 4, 9], point_len, 4);
            check(&opened, 4).unwrap_or_else(|e| panic!("subset {log_len}, {point_len}: {e}"));
        }
    }

    #[test]
    fn a_point_shorter_than_the_rows_verifies() {
        // Every polynomial is smaller than a column of the matrix, and so is the point.
        let mut rng = StdRng::seed_from_u64(2);
        let polys: Vec<MultilinearPolynomial<Fp128>> = (0..5)
            .map(|i| {
                MultilinearPolynomial::from(
                    (0..1usize << (i + 1))
                        .map(|_| rng.gen_range(0..4u8))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let opened = open(&polys, &[0, 1, 2, 3, 4], 5, 9);
        check(&opened, 9).unwrap();
    }

    #[test]
    fn wrong_statements_are_rejected() {
        let mut rng = StdRng::seed_from_u64(5);
        let polys = batch(&mut rng, 12);
        let all: Vec<usize> = (0..polys.len()).collect();
        let honest = open(&polys, &all, 13, 6);
        check(&honest, 6).unwrap();
        let one = Fp128::from(1u128);
        let reject = |mutate: &dyn Fn(&mut Opened)| {
            let mut opened = open(&polys, &all, 13, 6);
            mutate(&mut opened);
            check(&opened, 6).is_err()
        };

        assert!(reject(&|o| o.joint_claim += one), "joint claim");
        assert!(reject(&|o| o.coeffs[2] += one), "coefficient");
        assert!(reject(&|o| o.point[12] += one), "row coordinate");
        assert!(reject(&|o| o.point[0] += one), "padding coordinate");
        assert!(reject(&|o| o.slots.swap(0, 2)), "slots of different sizes");
        assert!(reject(&|o| o.slots.swap(0, 1)), "slots of the same shape");
        assert!(reject(&|o| o.num_vars[3] += 1), "opened size");
        assert!(reject(&|o| o.slots[0] = BitzSlot(99)), "unknown slot");
        assert!(reject(&|o| o.batch.root[0] ^= 1), "root");
        assert!(reject(&|o| o.batch.shapes[0].bits = 3), "declared bits");
        assert!(
            reject(&|o| o.batch.shapes[4].signed = false),
            "declared sign"
        );
        assert!(
            reject(&|o| o.batch.shapes[2].num_vars += 1),
            "declared size"
        );
        assert!(
            reject(&|o| {
                o.batch.shapes.pop();
            }),
            "shape count"
        );
        assert!(
            reject(&|o| {
                o.proof.narg.push(0);
            }),
            "trailing transcript bytes"
        );
        assert!(
            reject(&|o| {
                o.proof.narg.pop();
            }),
            "short transcript"
        );
        assert!(reject(&|o| o.proof.proof.column_sums[0] ^= 1), "proof");
        // The same proof under another seed of the caller's transcript.
        assert!(check(&honest, 7).is_err());
    }

    #[test]
    fn layouts_are_aligned_and_disjoint() {
        let shapes = [
            Shape {
                num_vars: 12,
                bits: 2,
                signed: false,
            },
            Shape {
                num_vars: 3,
                bits: 1,
                signed: false,
            },
            Shape {
                num_vars: 10,
                bits: 32,
                signed: true,
            },
            Shape {
                num_vars: 14,
                bits: 1,
                signed: false,
            },
        ];
        let layout = Layout::new(&shapes).unwrap();
        let mut taken = vec![false; 1 << layout.log_columns];
        for (shape, planes) in shapes.iter().zip(&layout.planes) {
            assert_eq!(planes.len(), shape.bits as usize);
            for plane in planes {
                let expected = (shape.num_vars as usize).saturating_sub(layout.log_rows);
                assert_eq!(plane.log_columns, expected);
                assert_eq!(plane.first_column % (1 << plane.log_columns), 0);
                let columns = plane.first_column..plane.first_column + (1 << plane.log_columns);
                for (offset, slot) in taken[columns].iter_mut().enumerate() {
                    let column = plane.first_column + offset;
                    assert!(!std::mem::replace(slot, true), "column {column}");
                }
            }
        }
        assert!(layout.params().is_some());
        assert!(Layout::new(&[]).is_none());
        assert!(Layout::new(&[Shape {
            num_vars: 3,
            bits: 0,
            signed: false
        }])
        .is_none());
        assert!(Layout::new(&[Shape {
            num_vars: 3,
            bits: 65,
            signed: false
        }])
        .is_none());
        assert!(Layout::new(&[Shape {
            num_vars: 40,
            bits: 1,
            signed: false
        }])
        .is_none());
        assert!(Layout::new(&[Shape {
            num_vars: 31,
            bits: 2,
            signed: false
        }])
        .is_none());
    }
}
