//! Pedersen commitment scheme for small vectors (e.g., sumcheck round polynomials).
//!
//! Commitments are of the form:
//!   C = Sum_i m_i * G_i + r * H
//! where G_i are message generators and H is the blinding generator.

use crate::curve::JoltCurve;
use crate::field::JoltField;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use rand_core::CryptoRngCore;

#[derive(Clone, Debug)]
pub struct PedersenGenerators<C: JoltCurve> {
    pub message_generators: Vec<C::G1>,
    pub blinding_generator: C::G1,
    /// Pre-converted affine bases: [msg_0, ..., msg_{n-1}, blinding]
    affine_bases: Vec<C::G1Affine>,
}

impl<C: JoltCurve> CanonicalSerialize for PedersenGenerators<C> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.message_generators
            .serialize_with_mode(&mut writer, compress)?;
        self.blinding_generator
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.message_generators.serialized_size(compress)
            + self.blinding_generator.serialized_size(compress)
    }
}

impl<C: JoltCurve> Valid for PedersenGenerators<C> {
    fn check(&self) -> Result<(), SerializationError> {
        self.message_generators.check()?;
        self.blinding_generator.check()
    }
}

impl<C: JoltCurve> CanonicalDeserialize for PedersenGenerators<C> {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let message_generators =
            Vec::<C::G1>::deserialize_with_mode(&mut reader, compress, validate)?;
        let blinding_generator = C::G1::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self::new(message_generators, blinding_generator))
    }
}

impl<C: JoltCurve> PedersenGenerators<C> {
    pub fn new(message_generators: Vec<C::G1>, blinding_generator: C::G1) -> Self {
        assert!(
            !message_generators.is_empty(),
            "Need at least one generator"
        );
        let mut affine_bases: Vec<C::G1Affine> =
            message_generators.iter().map(C::g1_to_affine).collect();
        affine_bases.push(C::g1_to_affine(&blinding_generator));
        Self {
            message_generators,
            blinding_generator,
            affine_bases,
        }
    }

    /// Single MSM including blinding.
    pub fn commit(&self, coeffs: &[C::F], blinding: &C::F) -> C::G1 {
        let n = coeffs.len();
        assert!(
            n <= self.message_generators.len(),
            "Too many coefficients: {} > {}",
            n,
            self.message_generators.len()
        );

        let blinding_affine_idx = self.message_generators.len();
        let mut combined_bases = Vec::with_capacity(n + 1);
        combined_bases.extend_from_slice(&self.affine_bases[..n]);
        combined_bases.push(self.affine_bases[blinding_affine_idx]);

        let mut combined_scalars = Vec::with_capacity(n + 1);
        combined_scalars.extend_from_slice(coeffs);
        combined_scalars.push(*blinding);

        C::g1_affine_msm(&combined_bases, &combined_scalars)
    }

    /// Commit values in chunks, each with a fresh random blinding.
    pub fn commit_chunked<R: CryptoRngCore>(
        &self,
        values: &[C::F],
        rng: &mut R,
    ) -> Vec<(C::G1, C::F)> {
        values
            .chunks(self.message_generators.len())
            .map(|chunk| {
                let blinding = C::F::random(rng);
                let commitment = self.commit(chunk, &blinding);
                (commitment, blinding)
            })
            .collect()
    }

    pub fn verify(&self, commitment: &C::G1, coeffs: &[C::F], blinding: &C::F) -> bool {
        let expected = self.commit(coeffs, blinding);
        *commitment == expected
    }
}

#[cfg(test)]
impl PedersenGenerators<crate::curve::Bn254Curve> {
    /// Test-only: derives generators deterministically from hash.
    pub fn deterministic(count: usize) -> Self {
        use ark_bn254::G1Projective;
        use ark_std::UniformRand;
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;
        use sha3::Digest;

        let hash_to_g1 = |domain: &[u8]| -> crate::curve::Bn254G1 {
            let hash = sha3::Sha3_256::digest(domain);
            let mut rng = ChaCha20Rng::from_seed(hash.into());
            crate::curve::Bn254G1(G1Projective::rand(&mut rng))
        };

        let generators = (0..count)
            .map(|i| {
                let mut domain = b"jolt_pedersen_msg_gen_v1_".to_vec();
                domain.extend_from_slice(&(i as u64).to_le_bytes());
                hash_to_g1(&domain)
            })
            .collect();
        let blinding_generator = hash_to_g1(b"jolt_pedersen_blinding_h2c_v1");
        Self::new(generators, blinding_generator)
    }
}

/// Hyrax-style helpers for row-committed polynomial evaluation.
pub mod hyrax {
    use crate::field::JoltField;
    use crate::poly::eq_poly::EqPolynomial;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

    /// combined[k] = Sum_i eq(ry_row, i) * flat[i*cols + k]
    pub fn combined_row<F: JoltField>(flat: &[F], cols: usize, ry_row: &[F]) -> Vec<F> {
        let num_rows = 1usize << ry_row.len();
        let eq_row: Vec<F> = EqPolynomial::evals(ry_row);

        let mut combined = vec![F::zero(); cols];
        for i in 0..num_rows {
            let w: F = eq_row[i];
            if w.is_zero() {
                continue;
            }
            let base = i * cols;
            for k in 0..cols {
                if base + k < flat.len() {
                    combined[k] += w * flat[base + k];
                }
            }
        }
        combined
    }

    /// eval = Sum_k combined_row[k] * eq(ry_col, k)
    pub fn evaluate<F: JoltField>(combined_row: &[F], ry_col: &[F]) -> F {
        let eq_col: Vec<F> = EqPolynomial::evals(ry_col);
        combined_row
            .iter()
            .zip(eq_col.iter())
            .map(|(c, e)| *c * *e)
            .sum()
    }

    /// combined_blinding = Sum_i eq(ry_row, i) * row_blindings[i]
    pub fn combined_blinding<F: JoltField>(row_blindings: &[F], ry_row: &[F]) -> F {
        let eq_row: Vec<F> = EqPolynomial::evals(ry_row);
        row_blindings
            .iter()
            .zip(eq_row.iter())
            .map(|(b, e)| *b * *e)
            .sum()
    }

    #[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
    pub struct HyraxOpeningProof<F: JoltField> {
        pub combined_row: Vec<F>,
        pub combined_blinding: F,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::{Bn254Curve, JoltGroupElement};
    use ark_bn254::Fr;
    use ark_std::UniformRand;
    use rand::thread_rng;

    #[test]
    fn test_pedersen_commit_verify() {
        let mut rng = thread_rng();
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(10);

        let coeffs: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
        let blinding = Fr::rand(&mut rng);

        let commitment = gens.commit(&coeffs, &blinding);
        assert!(gens.verify(&commitment, &coeffs, &blinding));

        let wrong_blinding = Fr::rand(&mut rng);
        assert!(!gens.verify(&commitment, &coeffs, &wrong_blinding));
    }

    #[test]
    fn test_commitment_homomorphism() {
        let mut rng = thread_rng();
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(10);

        let coeffs1: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
        let coeffs2: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
        let r1 = Fr::rand(&mut rng);
        let r2 = Fr::rand(&mut rng);

        let c1 = gens.commit(&coeffs1, &r1);
        let c2 = gens.commit(&coeffs2, &r2);
        let c_sum = c1 + c2;

        let coeffs_sum: Vec<Fr> = coeffs1
            .iter()
            .zip(coeffs2.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let r_sum = r1 + r2;
        let c_expected = gens.commit(&coeffs_sum, &r_sum);

        assert_eq!(c_sum, c_expected);
    }

    #[test]
    fn test_blinding_generator_independent() {
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(3);
        for msg_gen in &gens.message_generators {
            assert_ne!(gens.blinding_generator, *msg_gen);
        }
        assert!(!gens.blinding_generator.is_zero());
    }
}
