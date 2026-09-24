use crate::{
    ops::{Op, Reshape},
    tensor::Tensor,
};

impl Op for Reshape {
    /// Reinterpreting the buffer would scramble a padded tensor, whose real
    /// elements are interleaved with padding. Crop to the declared input shape
    /// first, reshape those elements alone, then pad the result back out.
    ///
    /// The output's padding is therefore zero whatever the input's was, which
    /// matches the selector `build_reshape_selectors` builds for the proof: it
    /// is zero on padding cells and maps only real elements.
    #[tracing::instrument(name = "Reshape::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let input = inputs[0];
        let padded = input.dims() != self.input_shape;
        let ranges: Vec<_> = self.input_shape.iter().map(|&d| 0..d).collect();
        let mut t = input
            .get_slice(&ranges)
            .expect("reshape input does not contain its declared shape");
        t.reshape(&self.output_shape)
            .expect("reshape output shape has a different element count");
        if padded {
            t.pad_next_power_of_two();
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A padded reshape must move real elements to the positions the output's
    /// own padding implies, not reinterpret the buffer.
    fn check(input_dims: &[usize], output_dims: &[usize]) {
        let values: Vec<i32> = (0..input_dims.iter().product::<usize>())
            .map(|v| v as i32 + 1)
            .collect();
        let mut input = Tensor::new(Some(&values), input_dims).unwrap();
        input.pad_next_power_of_two();
        let mut expected = Tensor::new(Some(&values), output_dims).unwrap();
        expected.pad_next_power_of_two();
        let op = Reshape {
            input_shape: input_dims.to_vec(),
            output_shape: output_dims.to_vec(),
        };
        assert_eq!(op.f(vec![&input]), expected);
    }

    #[test]
    fn reshape_moves_padding_between_groups() {
        check(&[2, 3, 4], &[6, 4]);
        check(&[6, 4], &[2, 3, 4]);
        check(&[2, 3, 5], &[6, 5]);
        check(&[6, 5], &[2, 3, 5]);
        check(&[1, 2, 7, 32, 64], &[1, 14, 32, 64]);
        check(&[1, 14, 32, 32], &[1, 2, 7, 32, 32]);
        check(&[3, 3], &[9]);
        check(&[9], &[3, 3]);
    }

    /// Padding each dimension separately can leave the two padded domains
    /// different sizes: `[3, 5]` pads to 4x8 = 32 while `[15]` pads to 16.
    #[test]
    fn reshape_between_unequal_padded_domains() {
        check(&[3, 5], &[15]);
        check(&[15], &[3, 5]);
        check(&[3, 3, 5], &[45]);
        check(&[45], &[3, 3, 5]);
    }

    #[test]
    fn unpadded_input_stays_unpadded() {
        let values: Vec<i32> = (1..=6).collect();
        let input = Tensor::new(Some(&values), &[2, 3]).unwrap();
        let op = Reshape {
            input_shape: vec![2, 3],
            output_shape: vec![6],
        };
        assert_eq!(
            op.f(vec![&input]),
            Tensor::new(Some(&values), &[6]).unwrap()
        );
    }

    /// Dirty padding in the input must not reach the output.
    #[test]
    fn padded_input_is_laundered_clean() {
        let mut input = Tensor::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();
        input.pad_next_power_of_two();
        for (i, v) in input.inner.iter_mut().enumerate() {
            if i % 4 == 3 {
                *v = 999; // the padding column
            }
        }
        let op = Reshape {
            input_shape: vec![2, 3],
            output_shape: vec![6],
        };
        let mut expected = Tensor::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
        expected.pad_next_power_of_two();
        assert_eq!(op.f(vec![&input]), expected);
    }
}
