/// Helper function to convert Vec<u64> to iterator of i32
pub fn u64_vec_to_i32_iter(vec: &[u64]) -> impl Iterator<Item = i32> + '_ {
    vec.iter().map(|v| *v as u32 as i32)
}

pub fn u64_vec_to_i32_saturating(vec: &[u64]) -> impl Iterator<Item = i32> + '_ {
    vec.iter().map(|v| u64_to_i32_saturating(*v))
}

pub fn u64_to_i32_saturating(v: u64) -> i32 {
    if v > i32::MAX as u64 {
        i32::MAX
    } else {
        v as i32
    }
}
