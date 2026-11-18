//! Binding a polynomial's lowest variable where its values lie.

use rayon::prelude::*;

/// The first run [`fold_pairs_in_place`] folds through a buffer of its own.
const FIRST_RUN: usize = 1 << 12;

/// `values[i] = fold(values[2i], values[2i + 1])` for `i < values.len() / 2`, in place: the
/// lowest variable of a polynomial bound with no allocation. Output `i` overwrites an input
/// of output `i / 2`, so outputs are written in runs `[a, 2a)`: a run reads `[2a, 4a)`, past
/// every value it writes and every value an earlier run read, and its outputs fold in
/// parallel. The first run, `[0, a)`, reads values it overwrites and goes through a buffer.
/// `values.len()` must be even.
pub fn fold_pairs_in_place<T: Copy + Send + Sync>(
    values: &mut [T],
    fold: impl Fn(T, T) -> T + Sync,
) {
    debug_assert_eq!(values.len() % 2, 0);
    let n = values.len() / 2;
    let first = n.min(FIRST_RUN);
    let head: Vec<T> = values[..2 * first]
        .chunks_exact(2)
        .map(|pair| fold(pair[0], pair[1]))
        .collect();
    values[..first].copy_from_slice(&head);
    let mut start = first;
    while start < n {
        let end = (2 * start).min(n);
        let (out, rest) = values.split_at_mut(2 * start);
        out[start..end]
            .par_iter_mut()
            .zip(rest[..2 * (end - start)].par_chunks_exact(2))
            .with_min_len(512)
            .for_each(|(slot, pair)| *slot = fold(pair[0], pair[1]));
        start = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The in-place fold is the fold into a new vector, at lengths around the first run.
    #[test]
    fn folds_in_place_as_into_a_new_vector() {
        for half in [
            1usize,
            7,
            FIRST_RUN - 1,
            FIRST_RUN,
            FIRST_RUN + 3,
            5 * FIRST_RUN + 11,
        ] {
            let values: Vec<u64> = (0..2 * half as u64)
                .map(|i| i * 2_654_435_761 % 1_000_003)
                .collect();
            let want: Vec<u64> = values.chunks_exact(2).map(|p| p[0] * 3 + p[1]).collect();
            let mut got = values.clone();
            fold_pairs_in_place(&mut got, |a, b| a * 3 + b);
            assert_eq!(&got[..half], &want[..]);
        }
    }
}
