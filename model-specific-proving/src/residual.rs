//! A residual addition only splits an existing evaluation claim.

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
