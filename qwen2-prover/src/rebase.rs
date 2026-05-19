#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rounding {
    Floor,
    Nearest,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantParams {
    pub scale: f64,
    pub zero_point: i128,
    pub bits: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShiftMultiplier {
    pub multiplier: i128,
    pub shift: u32,
    pub rounding: Rounding,
}

impl ShiftMultiplier {
    pub fn apply(self, value: i128) -> i128 {
        assert!(self.shift < 127);
        let product = value * self.multiplier;
        match self.rounding {
            Rounding::Floor => product >> self.shift,
            Rounding::Nearest if self.shift == 0 => product,
            Rounding::Nearest => {
                let half = 1i128 << (self.shift - 1);
                if product >= 0 {
                    (product + half) >> self.shift
                } else {
                    -(((-product) + half) >> self.shift)
                }
            }
        }
    }
}

pub fn q_for_bits(bits: u8) -> i128 {
    assert!(bits > 0);
    assert!(bits < 127);
    (1i128 << bits) - 1
}

pub fn make_shift_multiplier_floor(
    numerator: i128,
    denominator: i128,
    shift: u32,
) -> ShiftMultiplier {
    assert!(denominator > 0);
    assert!(shift < 127);
    ShiftMultiplier {
        multiplier: (numerator << shift) / denominator,
        shift,
        rounding: Rounding::Floor,
    }
}

pub fn make_shift_multiplier_nearest(
    numerator: i128,
    denominator: i128,
    shift: u32,
) -> ShiftMultiplier {
    assert!(denominator > 0);
    assert!(shift < 127);
    ShiftMultiplier {
        multiplier: ((numerator << shift) + denominator / 2) / denominator,
        shift,
        rounding: Rounding::Nearest,
    }
}

pub fn make_shift_multiplier_from_ratio(
    ratio: f64,
    shift: u32,
    rounding: Rounding,
) -> ShiftMultiplier {
    assert!(ratio.is_finite());
    assert!(ratio >= 0.0);
    assert!(shift < 127);
    let scaled = ratio * ((1u128 << shift) as f64);
    let multiplier = match rounding {
        Rounding::Floor => scaled.floor(),
        Rounding::Nearest => scaled.round(),
    } as i128;
    ShiftMultiplier {
        multiplier,
        shift,
        rounding,
    }
}

pub fn di_rebase_value(p: i128, p_min: i128, p_max: i128, bits: u8, rounding: Rounding) -> i128 {
    assert!(p_max > p_min);
    let q = q_for_bits(bits);
    let den = p_max - p_min;
    let round_bias = match rounding {
        Rounding::Floor => 0,
        Rounding::Nearest => den / 2,
    };
    let y = ((p - p_min) * q + round_bias) / den;
    debug_assert!(
        0 <= y && y <= q,
        "DI-Rebase output out of range: {y}, expected [0, {q}]"
    );
    y
}

pub fn di_rebase_slice(
    p: &[i128],
    p_min: i128,
    p_max: i128,
    bits: u8,
    rounding: Rounding,
) -> Vec<i128> {
    p.iter()
        .map(|&value| di_rebase_value(value, p_min, p_max, bits, rounding))
        .collect()
}

pub fn di_rebase_output_params(p_scale: f64, p_min: i128, p_max: i128, bits: u8) -> QuantParams {
    assert!(p_max > p_min);
    let q = q_for_bits(bits);
    let den = p_max - p_min;
    let zero_point = ((-p_min) * q + den / 2) / den;
    QuantParams {
        scale: p_scale * den as f64 / q as f64,
        zero_point,
        bits,
    }
}

pub fn di_scale_shift_value(value: i128, multiplier: ShiftMultiplier) -> i128 {
    multiplier.apply(value)
}

pub fn di_rebase_shift_multiplier(
    p_min: i128,
    p_max: i128,
    bits: u8,
    multiplier_shift: u32,
    rounding: Rounding,
) -> ShiftMultiplier {
    assert!(p_max > p_min);
    let q = q_for_bits(bits);
    let den = p_max - p_min;
    match rounding {
        Rounding::Floor => make_shift_multiplier_floor(q, den, multiplier_shift),
        Rounding::Nearest => make_shift_multiplier_nearest(q, den, multiplier_shift),
    }
}

pub fn di_rebase_shift_value(
    p: i128,
    p_min: i128,
    p_max: i128,
    bits: u8,
    multiplier: ShiftMultiplier,
) -> i128 {
    assert!(p_max > p_min);
    let q = q_for_bits(bits);
    let y = multiplier.apply(p - p_min);
    debug_assert!(
        0 <= y && y <= q,
        "DI-Rebase shift output out of range: {y}, expected [0, {q}]"
    );
    y
}

pub fn di_rebase_shift_slice(
    p: &[i128],
    p_min: i128,
    p_max: i128,
    bits: u8,
    multiplier: ShiftMultiplier,
) -> Vec<i128> {
    p.iter()
        .map(|&value| di_rebase_shift_value(value, p_min, p_max, bits, multiplier))
        .collect()
}

pub fn di_rebase(
    p: &[i128],
    p_scale: f64,
    p_min: i128,
    p_max: i128,
    bits: u8,
    rounding: Rounding,
) -> (Vec<i128>, QuantParams) {
    let values = di_rebase_slice(p, p_min, p_max, bits, rounding);
    let params = di_rebase_output_params(p_scale, p_min, p_max, bits);
    (values, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_maps_endpoints_and_midpoint() {
        assert_eq!(di_rebase_value(10, 10, 20, 8, Rounding::Nearest), 0);
        assert_eq!(di_rebase_value(15, 10, 20, 8, Rounding::Nearest), 128);
        assert_eq!(di_rebase_value(20, 10, 20, 8, Rounding::Nearest), 255);
    }

    #[test]
    fn floor_maps_midpoint_down() {
        assert_eq!(di_rebase_value(15, 10, 20, 8, Rounding::Floor), 127);
    }

    #[test]
    fn output_scale_uses_input_scale_and_range() {
        let params = di_rebase_output_params(0.25, -5, 5, 8);
        assert_eq!(params.bits, 8);
        assert_eq!(params.zero_point, 128);
        assert!((params.scale - 10.0 * 0.25 / 255.0).abs() < 1e-12);
    }

    #[test]
    fn shift_rebase_matches_exact_floor_for_simple_range() {
        let m = di_rebase_shift_multiplier(0, 16, 8, 8, Rounding::Floor);
        assert_eq!(di_rebase_shift_value(0, 0, 16, 8, m), 0);
        assert_eq!(di_rebase_shift_value(8, 0, 16, 8, m), 127);
        assert_eq!(di_rebase_shift_value(16, 0, 16, 8, m), 255);
    }

    #[test]
    fn scale_shift_applies_integer_multiplier_and_shift() {
        let m = make_shift_multiplier_floor(3, 2, 8);
        assert_eq!(di_scale_shift_value(10, m), 15);
    }

    #[test]
    fn nearest_shift_handles_negative_values_symmetrically() {
        let m = ShiftMultiplier {
            multiplier: 1,
            shift: 1,
            rounding: Rounding::Nearest,
        };
        assert_eq!(di_scale_shift_value(15, m), 8);
        assert_eq!(di_scale_shift_value(-15, m), -8);
        assert_eq!(di_scale_shift_value(14, m), 7);
        assert_eq!(di_scale_shift_value(-14, m), -7);
    }
}
