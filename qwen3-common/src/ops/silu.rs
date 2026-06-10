use ark_bn254::Fr;
use joltworks::transcripts::Transcript;

use crate::{BitOpeningClaims, EvalClaim, MatrixShape, RaOpeningClaims, SumCheckRounds};

use crate::{FIXED_SCALE, FRAC_BITS};

pub struct SiluTensorOutput {
    pub rounds: SumCheckRounds<4>,
    pub input: Fr,
    pub index: Fr,
    pub base: Fr,
    pub slope: Fr,
    pub output: Fr,
    pub input_bits: [Fr; FRAC_BITS],
    pub output_bits: [Fr; FRAC_BITS],
}

pub struct SiluReadOutput {
    pub rounds: SumCheckRounds<3>,
    pub ra: Fr,
}

pub struct SiluRaVirtualOutput {
    pub rounds: SumCheckRounds<3>,
    pub ra: Fr,
}

pub struct SiluHammingWeightOutput {
    pub rounds: SumCheckRounds<2>,
    pub ra: Fr,
}

pub struct SiluBooleanityOutput {
    pub rounds: SumCheckRounds<4>,
    pub ra: Fr,
}

pub struct SiluLookupOutput {
    pub read: SiluReadOutput,
    pub ra_virtual: SiluRaVirtualOutput,
    pub hamming_weight: SiluHammingWeightOutput,
    pub booleanity: SiluBooleanityOutput,
}

pub struct SiluOutput {
    pub tensor: SiluTensorOutput,
    pub lookup: SiluLookupOutput,
}

pub struct SiluVerifierOutput {
    pub input: EvalClaim,
    pub input_bits: BitOpeningClaims,
    pub output_bits: BitOpeningClaims,
    pub ra: RaOpeningClaims,
}

pub struct SiluAdvice {
    pub min_n: i64,
    pub max_n: i64,
}

pub struct SiluTables {
    pub id: Vec<i32>,
    pub base: Vec<i32>,
    pub slope: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiluParams {
    pub shape: MatrixShape,
}

impl SiluParams {
    pub fn new(rows: usize, cols: usize) -> Option<Self> {
        Some(Self {
            shape: MatrixShape::new(rows, cols)?,
        })
    }
}

pub fn draw_silu_tensor_challenges<T>(
    transcript: &mut T,
) -> Option<(Fr, [Fr; FRAC_BITS], [Fr; FRAC_BITS])>
where
    T: Transcript,
{
    transcript.append_message(b"q3/silu/tensor/v1");
    let input_round_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let input_bit_booleanity_challenges = transcript
        .challenge_vector::<Fr>(FRAC_BITS)
        .try_into()
        .ok()?;
    let output_bit_booleanity_challenges = transcript
        .challenge_vector::<Fr>(FRAC_BITS)
        .try_into()
        .ok()?;
    Some((
        input_round_mix,
        input_bit_booleanity_challenges,
        output_bit_booleanity_challenges,
    ))
}

pub fn draw_silu_lookup_challenges<T>(transcript: &mut T) -> Option<([Fr; 4], Fr)>
where
    T: Transcript,
{
    transcript.append_message(b"q3/silu/lookup/v1");
    let read_challenges = transcript.challenge_vector::<Fr>(4).try_into().ok()?;
    let ra_booleanity_challenge = transcript.challenge_scalar_optimized::<Fr>().into();
    Some((read_challenges, ra_booleanity_challenge))
}

pub fn build_silu_tables(advice: &SiluAdvice) -> Option<SiluTables> {
    validate_silu_range(advice.min_n, advice.max_n)?;
    let entries = usize::try_from(advice.max_n - advice.min_n + 1).ok()?;
    let lut_len = padded_silu_lut_len(entries);
    Some(SiluTables {
        id: id_table(entries, lut_len)?,
        base: silu_table(advice.min_n, entries, lut_len, silu_base)?,
        slope: silu_table(advice.min_n, entries, lut_len, silu_slope)?,
    })
}

pub fn validate_silu_range(min_n: i64, max_n: i64) -> Option<()> {
    (max_n >= min_n).then_some(())
}

pub fn padded_silu_lut_len(entries: usize) -> usize {
    (entries + 1).next_power_of_two().max(16)
}

fn id_table(entries: usize, lut_len: usize) -> Option<Vec<i32>> {
    let mut table = vec![0_i32; lut_len];
    for (index, value) in table.iter_mut().take(entries).enumerate() {
        *value = i32::try_from(index).ok()?;
    }
    Some(table)
}

fn silu_table(
    min_n: i64,
    entries: usize,
    lut_len: usize,
    table_fn: fn(i64) -> i64,
) -> Option<Vec<i32>> {
    let mut table = vec![0_i32; lut_len];
    for (index, value) in table.iter_mut().take(entries).enumerate() {
        *value = i32::try_from(table_fn(min_n + index as i64)).ok()?;
    }
    Some(table)
}

fn silu_base(n: i64) -> i64 {
    let n_q8 = n * FIXED_SCALE;
    n_q8 * sigmoid_q8(n)
}

fn silu_slope(n: i64) -> i64 {
    let n_q8 = n * FIXED_SCALE;
    let sig = sigmoid_q8(n);
    let sig_slope = round_shift_signed_i64(sig * (FIXED_SCALE - sig), FRAC_BITS);
    sig + round_shift_signed_i64(n_q8 * sig_slope, FRAC_BITS)
}

fn sigmoid_q8(n: i64) -> i64 {
    let sig = 1.0 / (1.0 + (-(n as f64)).exp());
    (sig * FIXED_SCALE as f64).round() as i64
}

fn round_shift_signed_i64(x: i64, shift: usize) -> i64 {
    let q = x.div_euclid(1_i64 << shift);
    let r = x - (q << shift);
    if r * 2 >= 1_i64 << shift { q + 1 } else { q }
}
