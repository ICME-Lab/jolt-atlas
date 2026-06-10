use ark_bn254::Fr;
use joltworks::transcripts::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalClaim {
    pub value: Fr,
    pub point: Vec<Fr>,
}

impl EvalClaim {
    pub fn new(value: Fr, point: Vec<Fr>) -> Self {
        Self { value, point }
    }
}

pub fn append_eval_claim<T>(transcript: &mut T, claim: &EvalClaim)
where
    T: Transcript,
{
    transcript.append_scalar(&claim.value);
    transcript.append_scalars(&claim.point);
}
