use ark_bn254::Fr;
use joltworks::transcripts::Transcript;

use crate::{BitOpeningClaims, EvalClaim, MatrixShape, RaOpeningClaims, SumCheckRounds};

use crate::{FIXED_SCALE, FRAC_BITS};

pub struct SoftmaxOutputToAccOutput {
    pub rounds: SumCheckRounds<4>,
    pub output: Fr,
    pub floor: Fr,
    pub acc: Fr,
    pub output_bits: [Fr; FRAC_BITS],
    pub floor_bits: [Fr; FRAC_BITS],
}

pub struct SoftmaxExpOutput {
    pub rounds: SumCheckRounds<4>,
    pub input: Fr,
    pub index: Fr,
    pub exp_base: Fr,
    pub exp: Fr,
    pub exp_acc: Fr,
    pub acc: Fr,
    pub row_max: Fr,
    pub sum: Fr,
    pub frac_bits: [Fr; FRAC_BITS],
    pub exp_bits: [Fr; FRAC_BITS],
}

pub struct SoftmaxRowSumOutput {
    pub rounds: SumCheckRounds<4>,
    pub exp: Fr,
    pub input: Fr,
}

pub struct SoftmaxReadOutput {
    pub rounds: SumCheckRounds<3>,
    pub ra: Fr,
}

pub struct SoftmaxRaVirtualOutput {
    pub rounds: SumCheckRounds<3>,
    pub ra: Fr,
}

pub struct SoftmaxHammingWeightOutput {
    pub rounds: SumCheckRounds<2>,
    pub ra: Fr,
}

pub struct SoftmaxBooleanityOutput {
    pub rounds: SumCheckRounds<4>,
    pub ra: Fr,
}

pub struct SoftmaxLookupOutput {
    pub read: SoftmaxReadOutput,
    pub ra_virtual: SoftmaxRaVirtualOutput,
    pub hamming_weight: SoftmaxHammingWeightOutput,
    pub booleanity: SoftmaxBooleanityOutput,
}

pub struct SoftmaxOutput {
    pub output_to_acc: SoftmaxOutputToAccOutput,
    pub row_sum: SoftmaxRowSumOutput,
    pub exp: SoftmaxExpOutput,
    pub lookup: SoftmaxLookupOutput,
}

pub struct SoftmaxVerifierOutput {
    pub input: EvalClaim,
    pub output_bits: BitOpeningClaims,
    pub floor_bits: BitOpeningClaims,
    pub frac_bits: BitOpeningClaims,
    pub exp_bits: BitOpeningClaims,
    pub ra: RaOpeningClaims,
}

pub struct SoftmaxAdvice {
    pub min_diff: i64,
    pub max_diff: i64,
    pub row_max: Vec<i32>,
    pub max_index: Vec<usize>,
    pub sum: Vec<i32>,
}

pub struct SoftmaxTables {
    pub id: Vec<i32>,
    pub exp_base: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxParams {
    pub shape: MatrixShape,
}

impl SoftmaxParams {
    pub fn new(rows: usize, cols: usize) -> Option<Self> {
        Some(Self {
            shape: MatrixShape::new(rows, cols)?,
        })
    }
}

pub fn draw_softmax_output_to_acc_challenges<T>(
    transcript: &mut T,
) -> Option<(Fr, Fr, [Fr; FRAC_BITS], [Fr; FRAC_BITS])>
where
    T: Transcript,
{
    transcript.append_message(b"q3/softmax/output_to_acc/v1");
    let output_round_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let floor_round_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let output_bit_challenges = transcript
        .challenge_vector::<Fr>(FRAC_BITS)
        .try_into()
        .ok()?;
    let floor_bit_challenges = transcript
        .challenge_vector::<Fr>(FRAC_BITS)
        .try_into()
        .ok()?;
    Some((
        output_round_mix,
        floor_round_mix,
        output_bit_challenges,
        floor_bit_challenges,
    ))
}

pub fn draw_softmax_exp_challenges<T>(
    transcript: &mut T,
) -> Option<(Fr, Fr, Fr, Fr, Fr, Fr, [Fr; FRAC_BITS], [Fr; FRAC_BITS])>
where
    T: Transcript,
{
    transcript.append_message(b"q3/softmax/exp_tensor/v1");
    let row_sum_exp_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let row_max_input_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let acc_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let exp_acc_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let exp_round_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let diff_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let frac_bit_challenges = transcript
        .challenge_vector::<Fr>(FRAC_BITS)
        .try_into()
        .ok()?;
    let exp_bit_challenges = transcript
        .challenge_vector::<Fr>(FRAC_BITS)
        .try_into()
        .ok()?;
    Some((
        row_sum_exp_mix,
        row_max_input_mix,
        acc_mix,
        exp_acc_mix,
        exp_round_mix,
        diff_mix,
        frac_bit_challenges,
        exp_bit_challenges,
    ))
}

pub fn draw_softmax_row_sum_challenge<T>(transcript: &mut T) -> Fr
where
    T: Transcript,
{
    transcript.append_message(b"q3/softmax/row_sum/v1");
    transcript.challenge_scalar_optimized::<Fr>().into()
}

pub fn draw_softmax_lookup_challenges<T>(transcript: &mut T) -> Option<([Fr; 3], Fr)>
where
    T: Transcript,
{
    transcript.append_message(b"q3/softmax/lookup/v1");
    let read_challenges = transcript.challenge_vector::<Fr>(3).try_into().ok()?;
    let ra_booleanity_challenge = transcript.challenge_scalar_optimized::<Fr>().into();
    Some((read_challenges, ra_booleanity_challenge))
}

pub fn build_softmax_tables(advice: &SoftmaxAdvice) -> Option<SoftmaxTables> {
    validate_softmax_range(advice.min_diff, advice.max_diff)?;
    let entries = usize::try_from(advice.max_diff - advice.min_diff + 1).ok()?;
    let lut_len = padded_softmax_lut_len(entries);
    Some(SoftmaxTables {
        id: id_table(entries, lut_len)?,
        exp_base: exp_table(advice.min_diff, entries, lut_len)?,
    })
}

pub fn validate_softmax_range(min_diff: i64, max_diff: i64) -> Option<()> {
    (max_diff >= min_diff).then_some(())
}

pub fn padded_softmax_lut_len(entries: usize) -> usize {
    entries.next_power_of_two().max(2)
}

fn id_table(entries: usize, lut_len: usize) -> Option<Vec<i32>> {
    let mut table = vec![0_i32; lut_len];
    for (index, value) in table.iter_mut().take(entries).enumerate() {
        *value = i32::try_from(index).ok()?;
    }
    Some(table)
}

fn exp_table(min_diff: i64, entries: usize, lut_len: usize) -> Option<Vec<i32>> {
    let mut table = vec![0_i32; lut_len];
    for (index, value) in table.iter_mut().take(entries).enumerate() {
        *value = i32::try_from(exp_lut_q8(min_diff + index as i64)).ok()?;
    }
    Some(table)
}

fn exp_lut_q8(n: i64) -> i64 {
    let n = n.clamp(-16, 0);
    (f64::exp(n as f64) * FIXED_SCALE as f64).round() as i64
}
