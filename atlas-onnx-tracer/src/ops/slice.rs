use crate::{
    ops::{Op, Slice},
    tensor::Tensor,
};

impl Op for Slice {
    #[tracing::instrument(name = "Slice::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let [data] = inputs[..] else {
            panic!("Slice expects exactly one input tensor")
        };
        assert!(
            self.axis < data.dims().len(),
            "Slice axis {} out of range for rank {}",
            self.axis,
            data.dims().len()
        );
        assert!(
            self.start <= self.end,
            "Slice start {} must be <= end {}",
            self.start,
            self.end
        );
        assert!(
            self.end <= data.dims()[self.axis],
            "Slice end {} out of bounds for dim {} on axis {}",
            self.end,
            data.dims()[self.axis],
            self.axis
        );

        let mut ranges: Vec<std::ops::Range<usize>> =
            data.dims().iter().map(|&d| 0..d).collect::<Vec<_>>();
        ranges[self.axis] = self.start..self.end;
        data.get_slice(&ranges).expect("Slice output must be valid")
    }
}
