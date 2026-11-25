//! Project columns once for the row rounds, then fold the packed rows at their challenges.
use super::*;

pub(super) fn prove(
    factors: Factors<'_>,
    packed: &[F128],
    target: F128,
    security: SecurityLevel,
    transcript: &mut ProverState,
    dense_entries: usize,
) -> Result<OpeningQuery, ProveError> {
    let rows = factors.rows.len();
    let words = rows / ELEMENT_BITS;
    let project = |sums: &mut [F128], (col, bits): (usize, &[F128])| {
        for (chunk, word) in sums.chunks_exact_mut(ELEMENT_BITS).zip(bits) {
            for row in iter_over_set_bits(word.to_u128()) {
                chunk[row] += factors.columns[col];
            }
        }
    };

    #[cfg(feature = "parallel")]
    let sums = packed
        .par_chunks_exact(words)
        .enumerate()
        .with_min_len(4096)
        .fold(
            || vec![F128::ZERO; rows],
            |mut sums, item| {
                project(&mut sums, item);
                sums
            },
        )
        .reduce(
            || vec![F128::ZERO; rows],
            |mut left, right| {
                for (dst, src) in left.iter_mut().zip(right) {
                    *dst += src;
                }
                left
            },
        );
    #[cfg(not(feature = "parallel"))]
    let sums = {
        let mut sums = vec![F128::ZERO; rows];
        for item in packed.chunks_exact(words).enumerate() {
            project(&mut sums, item);
        }
        sums
    };
    if factors
        .rows
        .iter()
        .zip(&sums)
        .map(|(&w, &v)| w * v)
        .sum::<F128>()
        != target
    {
        return Err(ProveError::ClaimDoesNotHold);
    }
    let mut pair = Pair::new(factors.rows.to_vec(), sums);
    let (point, running) = sumcheck::prove_rounds(
        &mut pair,
        rows.ilog2() as usize,
        target,
        security,
        transcript,
    );
    let scale = poly::f128::evaluate(factors.rows, &point);
    Ok(bounded::finish(
        vec![scale],
        factors.columns,
        packed,
        point,
        running,
        security,
        transcript,
        dense_entries,
    ))
}
