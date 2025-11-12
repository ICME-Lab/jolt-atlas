use onnx_tracer::tensor::Tensor;

pub mod mcc;
pub mod precompile_pp;

/// Helper function to convert Vec<u64> to iterator of i32
pub fn u64_vec_to_i32_iter(vec: &[u64]) -> impl Iterator<Item = i32> + '_ {
    vec.iter().map(|v| *v as u32 as i32)
}

pub fn tensor_to_u64s(t: &Tensor<i32>) -> Vec<u64> {
    tensor_to_nums(t)
}

pub fn tensor_to_u32s(t: &Tensor<i32>) -> Vec<u32> {
    tensor_to_nums(t)
}

pub trait FromI32 {
    fn from_i32(value: i32) -> Self;
}

impl FromI32 for u32 {
    fn from_i32(value: i32) -> Self {
        value as u32
    }
}

impl FromI32 for u64 {
    fn from_i32(value: i32) -> Self {
        value as u32 as u64
    }
}

pub fn tensor_to_nums<T: FromI32>(t: &Tensor<i32>) -> Vec<T> {
    t.inner.iter().map(|&v| T::from_i32(v)).collect()
}

// Used in virtual instructions to manage virtual sequence remaining elements
pub struct VirtualSequenceCounter {
    counter: usize,
}

impl VirtualSequenceCounter {
    pub fn new(length: usize) -> Self {
        Self { counter: length }
    }

    pub fn dec(&mut self) -> usize {
        assert!(self.counter > 0, "Virtual sequence counter underflow");
        self.counter -= 1;
        self.counter
    }

    pub fn get(&self) -> usize {
        self.counter
    }

    pub fn subtract(&mut self, value: usize) {
        assert!(
            self.counter >= value,
            "Virtual sequence counter underflow on subtract"
        );
        self.counter -= value;
    }
}
