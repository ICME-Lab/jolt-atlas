use crate::{
    ops::{Gather, Op},
    tensor::{self, Tensor},
};

impl Op for Gather {
    #[tracing::instrument(name = "Gather::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        assert_eq!(
            self.axis, 0,
            "Only axis 0 is currently supported for Gather"
        );
        let [x, y] = inputs[..] else {
            panic!("Expected exactly two inputs")
        };
        tensor::ops::gather(x, &y.map(|v| v as usize), self.axis).unwrap()
    }
}
