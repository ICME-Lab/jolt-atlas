//! Fold packed low coordinates on demand before the dense sumcheck tail.

use super::*;
use field::Wide256;

const WORD_LOG: usize = ELEMENT_BITS.ilog2() as usize;

struct PackedFold<'a> {
    packed: &'a [F128],
    bits: usize,
    bytes: Vec<[F128; 256]>,
}

impl<'a> PackedFold<'a> {
    fn new(packed: &'a [F128], bound: &[F128]) -> Self {
        let weights = poly::eq_table(bound);
        // One lookup per input byte adds the weights selected by that byte.
        let bytes = weights
            .chunks(8)
            .map(|weights| {
                let mut sums = [F128::ZERO; 256];
                for (i, &weight) in weights.iter().enumerate() {
                    for j in 0..1 << i {
                        sums[(1 << i) + j] = sums[j] + weight;
                    }
                }
                sums
            })
            .collect();
        Self {
            packed,
            bits: bound.len(),
            bytes,
        }
    }

    #[inline]
    fn value(&self, index: usize) -> F128 {
        if self.bits <= WORD_LOG {
            let position = index << self.bits;
            let mut word = self.packed[position >> WORD_LOG].to_u128() >> (position % ELEMENT_BITS);
            if self.bits < 3 {
                word &= (1u128 << (1 << self.bits)) - 1;
            }
            let mut sum = F128::ZERO;
            for table in &self.bytes {
                sum += table[word as u8 as usize];
                word >>= 8;
            }
            return sum;
        }
        let words = 1 << (self.bits - WORD_LOG);
        self.packed[index * words..][..words]
            .iter()
            .flat_map(|word| word.to_u128().to_le_bytes())
            .zip(&self.bytes)
            .fold(F128::ZERO, |sum, (byte, table)| sum + table[byte as usize])
    }
}

pub(super) fn prove(
    factors: Factors<'_>,
    packed: &[F128],
    first: F128,
    running: F128,
    security: SecurityLevel,
    transcript: &mut ProverState,
    dense_entries: usize,
) -> OpeningQuery {
    let rows = sumcheck::folded(factors.rows, first);
    finish(
        rows,
        factors.columns,
        packed,
        vec![first],
        running,
        security,
        transcript,
        dense_entries,
    )
}

pub(super) fn finish(
    mut rows: Vec<F128>,
    column_weights: &[F128],
    packed: &[F128],
    mut point: Vec<F128>,
    mut running: F128,
    security: SecurityLevel,
    transcript: &mut ProverState,
    dense_entries: usize,
) -> OpeningQuery {
    let mut columns = std::borrow::Cow::Borrowed(column_weights);

    while rows.len() * columns.len() > dense_entries {
        let read = PackedFold::new(packed, &point);
        let pair = |i: usize| {
            let w = |j: usize| rows[j & (rows.len() - 1)] * columns[j / rows.len()];
            let a = i * 2;
            let (w0, w1) = (w(a), w(a + 1));
            let (v0, v1) = (read.value(a), read.value(a + 1));
            (Wide256::mul(w0, v0), Wide256::mul(w0 + w1, v0 + v1))
        };
        let reduce = |(a0, a2): (Wide256, Wide256), (b0, b2)| (a0 + b0, a2 + b2);
        let identity = || (Wide256::zero(), Wide256::zero());
        let pairs = 0..rows.len() * columns.len() / 2;
        #[cfg(feature = "parallel")]
        let (a0, a2) = pairs
            .into_par_iter()
            .with_min_len(4096)
            .map(pair)
            .reduce(identity, reduce);
        #[cfg(not(feature = "parallel"))]
        let (a0, a2) = pairs.map(pair).fold(identity(), reduce);

        let message = [a0.reduce(), a2.reduce()];
        transcript.prover_message(&message);
        transcript.grind(GRINDING_LABEL, security.grinding_bits(2));
        let challenge = transcript.verifier_message();
        running = advance(running, message, challenge);
        point.push(challenge);
        if rows.len() > 1 {
            rows = sumcheck::folded(&rows, challenge);
        } else {
            columns = std::borrow::Cow::Owned(sumcheck::folded(&columns, challenge));
        }
    }

    let read = PackedFold::new(packed, &point);
    if rows.len() == 1 && columns.len() > 1 {
        let scale = rows[0];
        #[cfg(feature = "parallel")]
        let values: Vec<_> = (0..columns.len())
            .into_par_iter()
            .with_min_len(4096)
            .map(|i| read.value(i))
            .collect();
        #[cfg(not(feature = "parallel"))]
        let values: Vec<_> = (0..columns.len()).map(|i| read.value(i)).collect();
        let message = sumcheck::message(&columns, &values).map(|term| scale * term);
        transcript.prover_message(&message);
        transcript.grind(GRINDING_LABEL, security.grinding_bits(2));
        let challenge = transcript.verifier_message();
        running = advance(running, message, challenge);
        point.push(challenge);
        let (weights, values) = (
            sumcheck::folded_with_scale(&columns, challenge, scale),
            sumcheck::folded(&values, challenge),
        );
        let (rest, target) = sumcheck::prove(
            &mut Pair::new(weights, values),
            running,
            security,
            transcript,
        );
        point.extend(rest);
        return OpeningQuery::Mle { point, target };
    }
    let len = rows.len() * columns.len();
    let value = |i| read.value(i);
    let weight = |i: usize| rows[i & (rows.len() - 1)] * columns[i / rows.len()];
    #[cfg(feature = "parallel")]
    let (weights, values) = rayon::join(
        || {
            (0..len)
                .into_par_iter()
                .with_min_len(4096)
                .map(weight)
                .collect()
        },
        || {
            (0..len)
                .into_par_iter()
                .with_min_len(4096)
                .map(value)
                .collect()
        },
    );
    #[cfg(not(feature = "parallel"))]
    let (weights, values) = (
        (0..len).map(weight).collect(),
        (0..len).map(value).collect(),
    );

    let (rest, target) = sumcheck::prove(
        &mut Pair::new(weights, values),
        running,
        security,
        transcript,
    );
    point.extend(rest);
    OpeningQuery::Mle { point, target }
}
