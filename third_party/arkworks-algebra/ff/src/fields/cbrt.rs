use core::array::from_fn;

use crate::Field;

// Pairing value compression paper
// https://eprint.iacr.org/2005/133.pdf
// Cube root paper
// https://eprint.iacr.org/2009/457.pdf
pub enum CbrtPrecomputation<F: Field> {
    TonelliShanks {
        three_adicity: u32,
        cubic_nonresidue_to_trace: F,
        trace_of_modulus_minus_one: &'static [u64],
        trace_plus_or_minus_one_div_three: &'static [u64],
        trace_minus_one_div_by_three: bool,
    },
}

// Compute the cube root of a cube in Fr using Tonelli-Shanks.
/// p4 section 3.1 from https://eprint.iacr.org/2009/457.pdf
pub fn cube_root_tonelli_shanks<F: Field>(
    a: F,
    three_adicity: u32,
    cubic_nonresidue_to_trace: &F,
    trace_of_modulus_minus_one: &[u64],
    trace_plus_or_minus_one_div_three: &[u64],
    trace_minus_one_div_by_three: bool,
) -> F {
    // Step 1 is preprocessing

    // Step 2
    let mut c = cubic_nonresidue_to_trace.inverse().unwrap();
    let mut r = a.pow(trace_of_modulus_minus_one);

    // Step 3 Computation of the cube root of (a^t)^-1
    let mut h = F::one();
    let exp = 3_u32.pow(three_adicity - 1);
    let exp: [u64; 32] = from_fn(|i| ((exp >> (31 - i)) & 1) as u64);

    let c_prime = c.pow(&exp);

    for i in 1..three_adicity {
        // TODO: check this to_digit()
        let exp = 3_u64.pow(three_adicity - i - 1);
        let exp: [u64; 32] = from_fn(|i| ((exp >> (31 - i)) & 1) as u64);

        let d = r.pow(exp);
        if d == c_prime {
            h *= c;
            r *= c.square() * c;
        } else if d != F::one() {
            // The condition is equivalent to d = c_prime^2
            h *= c.square();
            r *= (c.square() * c).square();
        }
        c *= c.square();
    }

    // Step 4
    let r = a.pow(trace_plus_or_minus_one_div_three) * h;
    if trace_minus_one_div_by_three {
        r.inverse().unwrap()
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use ark_test_curves::bn254::fr::Fr;
    use core::str::FromStr;

    use super::*;

    fn find_cubic_nonresidue<F: Field>(char_minus_1_div_three: &[u64]) -> F {
        let mut x = F::from(2u64);
        loop {
            // if x^(p-1)/3 is not 1, then x is a cubic non-residue
            if x.pow(char_minus_1_div_three) != F::one() {
                break;
            }
            x += F::from(1u64);
        }
        x
    }

    fn test_find_cubic_nonresidue() {
        // (21888242871839275222246405745257275088548364400416034343698204186575808495617 - 1) // 3

        let char_minus_1_div_three = num_bigint::BigUint::from_str(
            "7296080957279758407415468581752425029516121466805344781232734728858602831872",
        )
        .unwrap()
        .to_u64_digits();

        let nonresidue = find_cubic_nonresidue::<Fr>(&char_minus_1_div_three);
        panic!("nonresidue: {:?}", nonresidue);
    }
}
