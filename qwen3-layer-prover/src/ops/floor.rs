use joltworks::{
    field::JoltField, poly::eq_poly::EqPolynomial, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use crate::{
    claim::{Claim, Shape, TensorId},
    error::{ProverError, Result},
    ops::round::{
        ROUND_FRAC_BITS, RoundParams, RoundProof, RoundWitness, prove_round, verify_round,
    },
};

// Floor rebase implemented through the canonical Shout-backed round op.
//
// For integer fixed-point values,
//
//     floor(x / 2^F) = round((x - 2^(F-1)) / 2^F)
//
// with the round convention used by `round.rs`:
//
//     y = floor(z / 2^F) + msb(z mod 2^F)
//
// This keeps floor from maintaining its own frac-bit decomposition.  The round
// proof opens `shifted = x - 2^(F-1)`.  The original `x` claim returned to the
// caller is reconstructed deterministically as `shifted + 2^(F-1) * valid`,
// where `valid` is the known MLE mask for the logical tensor domain.  This is
// why `floor` returns the same two claims as `round`: the original input claim
// and the round lookup RA claim.

const HALF_SCALE: i64 = 1_i64 << (ROUND_FRAC_BITS - 1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FloorParams {
    pub shape: Shape,
    pub input_tensor: TensorId,
    pub output_tensor: TensorId,
    pub frac_bit_tensors: [TensorId; ROUND_FRAC_BITS],
}

impl FloorParams {
    pub fn with_frac_bit_tensors(
        shape: impl Into<Vec<usize>>,
        input_tensor: impl Into<String>,
        output_tensor: impl Into<String>,
        frac_bit_tensors: [String; ROUND_FRAC_BITS],
    ) -> Self {
        Self {
            shape: Shape::new(shape),
            input_tensor: TensorId::new(input_tensor),
            output_tensor: TensorId::new(output_tensor),
            frac_bit_tensors: frac_bit_tensors.map(TensorId::new),
        }
    }

    fn round_params(&self) -> RoundParams {
        RoundParams {
            shape: self.shape.clone(),
            input_tensor: TensorId::new(format!("{}_shifted", self.input_tensor.0)),
            output_tensor: self.output_tensor.clone(),
            frac_bit_tensors: self.frac_bit_tensors.clone(),
            frac_bits: ROUND_FRAC_BITS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FloorWitness<'a> {
    pub input: &'a [i64],
    pub output: &'a [i32],
    pub frac_bits: [&'a [u8]; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct FloorProof<F: JoltField, T: Transcript> {
    pub round: RoundProof<F, T>,
}

pub fn prove_floor<F, T>(
    output_claims: Vec<Claim<F>>,
    witness: &FloorWitness<'_>,
    params: &FloorParams,
    transcript: &mut T,
) -> Result<(FloorProof<F, T>, Claim<F>, Claim<F>)>
where
    F: JoltField,
    T: Transcript,
{
    validate_inputs(&output_claims, witness, params)?;
    let shifted = witness
        .input
        .iter()
        .map(|value| value - HALF_SCALE)
        .collect::<Vec<_>>();
    let round_witness = RoundWitness::from_input_output(&shifted, witness.output);
    let (round, shifted_claim, ra_claim) = prove_round(
        output_claims,
        &round_witness,
        &params.round_params(),
        transcript,
    )?;
    Ok((
        FloorProof { round },
        unshift_claim(shifted_claim, params),
        ra_claim,
    ))
}

pub fn verify_floor<F, T>(
    output_claims: Vec<Claim<F>>,
    proof: &FloorProof<F, T>,
    params: &FloorParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let (shifted_claim, ra_claim) = verify_round(
        output_claims,
        &proof.round,
        &params.round_params(),
        transcript,
    )?;
    Ok((unshift_claim(shifted_claim, params), ra_claim))
}

fn unshift_claim<F: JoltField>(mut claim: Claim<F>, params: &FloorParams) -> Claim<F> {
    let valid = valid_eval(&params.shape, &claim.point);
    claim.tensor = params.input_tensor.clone();
    claim.value += F::from_u64(HALF_SCALE as u64) * valid;
    claim
}

fn valid_eval<F: JoltField>(shape: &Shape, point: &[F]) -> F {
    let mut offset = 0;
    let mut out = F::one();
    for &dim in shape.dims() {
        let padded = dim.next_power_of_two();
        let vars = padded.trailing_zeros() as usize;
        let eq = EqPolynomial::<F>::evals(&point[offset..offset + vars]);
        out *= eq.iter().take(dim).copied().sum::<F>();
        offset += vars;
    }
    out
}

fn validate_inputs<F: JoltField>(
    output_claims: &[Claim<F>],
    witness: &FloorWitness<'_>,
    params: &FloorParams,
) -> Result<()> {
    let len = params.shape.numel();
    ensure_len("floor input", len, witness.input.len())?;
    ensure_len("floor output", len, witness.output.len())?;
    for claim in output_claims {
        if claim.logical_shape != params.shape {
            return Err(ProverError::ShapeMismatch {
                name: "floor output claim",
                expected: params.shape.0.clone(),
                actual: claim.logical_shape.0.clone(),
            });
        }
    }
    for (idx, (&input, &output)) in witness.input.iter().zip(witness.output.iter()).enumerate() {
        let expected =
            ((input - input.rem_euclid(1_i64 << ROUND_FRAC_BITS)) >> ROUND_FRAC_BITS) as i32;
        if expected != output {
            return Err(ProverError::MatMulAccumulatorMismatch {
                row: idx,
                col: 0,
                expected: i64::from(expected),
                actual: i64::from(output),
            });
        }
    }
    Ok(())
}

fn ensure_len(name: &'static str, expected: usize, actual: usize) -> Result<()> {
    if actual != expected {
        return Err(ProverError::TensorLenMismatch {
            name,
            shape: vec![expected],
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::{
        field::JoltField, poly::eq_poly::EqPolynomial, transcripts::Blake2bTranscript,
    };

    use super::*;

    #[test]
    fn proves_and_verifies_floor() {
        let params = FloorParams::with_frac_bit_tensors(
            vec![1, 6],
            "acc",
            "floor",
            std::array::from_fn(|idx| format!("floor_frac_{idx}")),
        );
        let input = vec![-300_i64, -129, -128, -1, 0, 383];
        let output = input
            .iter()
            .map(|&x| ((x - x.rem_euclid(256)) / 256) as i32)
            .collect::<Vec<_>>();
        let witness = FloorWitness {
            input: &input,
            output: &output,
            frac_bits: std::array::from_fn(|_| &[][..]),
        };
        let point = vec![Fr::from(7_u64), Fr::from(11_u64), Fr::from(13_u64)];
        let output_claim = Claim {
            tensor: params.output_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            value: eval_i32(&output, &params.shape, &point),
            point,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input_claim, _) = prove_floor::<Fr, _>(
            vec![output_claim.clone()],
            &witness,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_input, _) = verify_floor::<Fr, _>(
            vec![output_claim],
            &proof,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();
        assert_eq!(verified_input, input_claim);
        assert_eq!(
            input_claim.value,
            eval_i64(&input, &params.shape, &input_claim.point)
        );
    }

    fn eval_i32<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
        let values = values
            .iter()
            .map(|&v| field_from_i64::<F>(i64::from(v)))
            .collect::<Vec<_>>();
        eval_padded(&values, shape, point)
    }

    fn eval_i64<F: JoltField>(values: &[i64], shape: &Shape, point: &[F]) -> F {
        let values = values
            .iter()
            .map(|&v| field_from_i64::<F>(v))
            .collect::<Vec<_>>();
        eval_padded(&values, shape, point)
    }

    fn eval_padded<F: JoltField>(values: &[F], shape: &Shape, point: &[F]) -> F {
        let padded_shape = shape.padded_power_of_two();
        let mut padded = vec![F::zero(); padded_shape.numel()];
        for (idx, value) in values.iter().enumerate() {
            padded[idx] = *value;
        }
        let eq = EqPolynomial::<F>::evals(point);
        padded
            .iter()
            .zip(eq)
            .fold(F::zero(), |acc, (value, eq)| acc + *value * eq)
    }

    fn field_from_i64<F: JoltField>(value: i64) -> F {
        if value >= 0 {
            F::from_u64(value as u64)
        } else {
            -F::from_u64(value.unsigned_abs())
        }
    }
}
