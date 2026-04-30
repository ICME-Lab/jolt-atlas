pub fn act(seed: u64) -> Vec<i32> {
    let mut s = seed;
    let mut xs = Vec::with_capacity(crate::X_LEN);
    for _ in 0..crate::X_LEN {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        xs.push(((s >> 32) % 513) as i32 - 256);
    }
    xs
}

#[cfg(test)]
mod tests {
    #[test]
    fn activation_shape() {
        let xs = super::act(1);
        assert_eq!(xs.len(), crate::X_LEN);
        assert!(xs.iter().all(|&x| (-512..=512).contains(&x)));
    }
}
