//! A residual addition only splits an existing evaluation claim.

#[cfg(test)]
use crate::norm::Claim;
#[cfg(test)]
use jolt_field::Field;
#[cfg(test)]
use jolt_transcript::{Encoding, Transcript};

/// The caller sends `left_value` before any challenge that uses either child.
/// The two returned claims still have to be discharged by their producers.
#[cfg(test)]
pub(crate) fn split<F: Field + Encoding>(
    output: Claim<F>,
    left_value: F,
    transcript: &mut impl Transcript,
) -> (Claim<F>, Claim<F>) {
    transcript.append_values(&[left_value]);
    let left = Claim {
        point: output.point.clone(),
        value: left_value,
    };
    let right = Claim {
        point: output.point,
        value: output.value - left_value,
    };
    (left, right)
}

pub fn add(input: &[i32], update: &[i32]) -> Result<Vec<i32>, &'static str> {
    if input.len() != update.len() {
        return Err("residual shape mismatch");
    }
    input
        .iter()
        .zip(update)
        .map(|(&x, &y)| x.checked_add(y).ok_or("residual exceeds i32"))
        .collect()
}
