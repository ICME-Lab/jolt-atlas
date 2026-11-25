//! Authenticate virtual lookup digits using committed binary planes.

use super::*;

pub(super) fn link<F: ProofField>(
    digits: &[&[F]],
    queries: &[(&Lookup, Read<'_, F>)],
    cell: &[F],
    values: &[[F; 8]],
    io: &mut Session<'_, '_, F>,
) -> Result<()> {
    let mix = io.point(queries.len() * digits.len())?;
    let sum = dot(
        values
            .iter()
            .flat_map(|row| row[..digits.len()].iter().copied()),
        mix.iter().copied(),
    );
    let mut shifts = Vec::with_capacity(digits.len());
    let mut shift = digits.iter().map(|digit| digit.len()).sum::<usize>();
    for digit in digits {
        shift -= digit.len();
        shifts.push(shift);
    }
    let inverse: Vec<Vec<Option<F>>> = digits
        .iter()
        .map(|digit| {
            digit
                .iter()
                .map(|&r| (r + r - F::one()).inverse())
                .collect()
        })
        .collect();
    let replay = cell.len().saturating_sub(12).min(4);
    let (end, evaluations) = io.fold_weighted(
        instance(sum, cell.len(), 5),
        &mix,
        || {
            queries
                .iter()
                .flat_map(|(_, read)| {
                    digits.iter().zip(&shifts).map(|(digit, &shift)| {
                        let addresses = read.witness();
                        let columns = std::array::from_fn::<_, 4, _>(|i| {
                            let read = move |cell| {
                                let Some(&r) = digit.get(i) else {
                                    return F::one();
                                };
                                if addresses.digit(cell, shift + digit.len() - 1 - i, 1) == 0 {
                                    F::one() - r
                                } else {
                                    r
                                }
                            };
                            Column::source(addresses.len(), read).replay(replay)
                        });
                        let [a, b, c, d] = columns;
                        [a, b, c, d, Column::equality(cell)]
                    })
                })
                .collect()
        },
        [0, 1, 2, 3],
        4,
        |columns: &[F]| columns.iter().copied().product(),
    )?;
    let mut expected = F::zero();
    for (((lookup, read), terms), mixing) in queries
        .iter()
        .zip(evaluations.chunks_exact(digits.len()))
        .zip(mix.chunks_exact(digits.len()))
    {
        for ((((&slot, digit), factors), &coefficient), (scales, &shift)) in lookup
            .cols
            .iter()
            .zip(digits)
            .zip(terms)
            .zip(mixing)
            .zip(inverse.iter().zip(&shifts))
        {
            let mut product = coefficient;
            for (i, ((&r, &factor), scale)) in digit.iter().zip(factors).zip(scales).enumerate() {
                product *= factor;
                let (bit, axis_plane) = match scale {
                    Some(scale) => ((factor - (F::one() - r)) * *scale, digit.len() - 1 - i),
                    None => {
                        if factor != r {
                            return Err("invalid constant lookup selector");
                        }
                        continue;
                    }
                };
                let plane = axis_plane;
                let axis = (0..digit.len())
                    .rev()
                    .map(|axis| F::from_bool(plane & (1 << axis) != 0));
                io.openings.push(Opening::new(
                    slot,
                    Claim {
                        point: axis.chain(end.challenges.iter().copied()).collect(),
                        value: bit,
                    },
                ));
            }
            // A half selector is constant. When all selectors are constant, authenticate
            // one plane at a fresh point so every committed slot still has an opening.
            if scales.iter().all(Option::is_none) {
                let [value] = io.values(|| {
                    [dot(
                        (0..read.witness().len())
                            .map(|i| F::from_bool(read.witness().digit(i, shift, 1) != 0)),
                        weights(&end.challenges),
                    )]
                })?;
                io.openings.push(Opening::new(
                    slot,
                    Claim {
                        point: [vec![F::zero(); digit.len()], end.challenges.clone()].concat(),
                        value,
                    },
                ));
            }
            expected += product;
        }
    }
    if end.output_claim != eq(cell, &end.challenges) * expected {
        return Err("invalid packed lookup digit evaluation");
    }
    Ok(())
}
