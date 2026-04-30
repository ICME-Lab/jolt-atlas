pub const S: i32 = 8;
pub const ONE: i32 = 1 << S;

pub fn silu(min: i32, max: i32) -> (Vec<i32>, i32) {
    table(min, max, |x| x / (1.0 + (-x).exp()))
}

pub fn exp(min: i32, max: i32) -> (Vec<i32>, i32) {
    table(min, max, f64::exp)
}

pub fn rsqrt(min: i32, max: i32) -> (Vec<i32>, i32) {
    table(min, max, |x| if x <= 0.0 { 0.0 } else { 1.0 / x.sqrt() })
}

fn table(min: i32, max: i32, f: impl Fn(f64) -> f64) -> (Vec<i32>, i32) {
    assert!(min <= max);
    let mut xs = Vec::with_capacity((max - min + 1) as usize);
    for q in min..=max {
        xs.push(to_q(f(from_q(q))));
    }
    (xs, -min)
}

fn from_q(x: i32) -> f64 {
    x as f64 / ONE as f64
}

fn to_q(x: f64) -> i32 {
    let y = (x * ONE as f64).round();
    if y > i32::MAX as f64 {
        i32::MAX
    } else if y < i32::MIN as f64 {
        i32::MIN
    } else {
        y as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_lut() {
        let (xs, z) = exp(-1, 1);
        assert_eq!(z, 1);
        assert_eq!(xs[z as usize], ONE);
    }

    #[test]
    fn silu_lut() {
        let (xs, z) = silu(-1, 1);
        assert_eq!(xs[z as usize], 0);
    }

    #[test]
    fn rsqrt_lut() {
        let (xs, z) = rsqrt(0, 256);
        assert_eq!(z, 0);
        assert_eq!(xs[0], 0);
        assert_eq!(xs[256], ONE);
    }
}
