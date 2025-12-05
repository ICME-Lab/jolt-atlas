pub mod display_trace;
pub mod f32;
pub mod parallel_utils;
pub mod parsing;

/// Helper function to convert Vec<u64> to iterator of i32
pub fn u64_vec_to_i32_iter(vec: &[u64]) -> impl Iterator<Item = i32> + '_ {
    vec.iter().map(|v| *v as u32 as i32)
}

// Used in virtual instructions to manage virtual sequence remaining elements
pub struct VirtualSequenceCounter {
    counter: usize,
}

impl VirtualSequenceCounter {
    pub fn new(start: usize) -> Self {
        Self { counter: start }
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

pub type VirtualSlotCounter = VirtualSequenceCounter;

impl VirtualSlotCounter {
    pub fn inc(&mut self) -> usize {
        let tmp = self.counter;
        self.counter += 1;
        tmp
    }
}

// converts a i32 to a u64 preserving sign-bit
// Used in the zkVM to convert raw trace values into the zkVM's 64 bit container type
pub fn normalize(value: &i32) -> u64 {
    *value as u32 as u64
}
