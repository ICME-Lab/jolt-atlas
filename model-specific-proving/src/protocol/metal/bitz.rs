use std::{ffi::c_void, ptr::NonNull};

use world_field::F128;

use super::device;

pub(crate) fn enable_bitz() {
    world_prover::set_tree_backend(bitz_tree);
    world_prover::set_dense_backend(dense_bitz);
}

fn bitz_tree(
    layer: &[world_field::F128],
    groups: usize,
    include_output: bool,
) -> Option<Vec<Vec<world_field::F128>>> {
    unsafe extern "C" {
        fn atlas_metal_tree(
            device: *mut c_void,
            input: *const F128,
            count: u32,
            groups: u32,
            include_output: bool,
            output: *mut F128,
        ) -> bool;
    }
    if !bitz_mode("tree") || layer.len() < (1 << 18) {
        return None;
    }
    let stop = groups * if include_output { 1 } else { 2 };
    let sizes: Vec<_> = std::iter::successors(Some(layer.len() / 2), |n| Some(n / 2))
        .take_while(|&n| n >= stop)
        .collect();
    if sizes.is_empty() {
        return None;
    }
    let mut upper = vec![F128 { lo: 0, hi: 0 }; sizes.iter().sum()];
    let success = unsafe {
        atlas_metal_tree(
            device()?,
            layer.as_ptr(),
            layer.len() as u32,
            groups as u32,
            include_output,
            upper.as_mut_ptr(),
        )
    };
    if !success {
        return None;
    }
    let mut tail = upper.as_slice();
    Some(
        sizes
            .into_iter()
            .map(|size| {
                let (next, rest) = tail.split_at(size);
                tail = rest;
                next.to_vec()
            })
            .collect(),
    )
}

fn bitz_mode(mode: &str) -> bool {
    std::env::var("ATLAS_METAL_BITZ").is_ok_and(|v| v == "1" || v == mode)
}

#[repr(C)]
struct BinaryArgs {
    challenge: F128,
    len: u32,
    low_size: u32,
    binding: u32,
    at_one: u32,
}

struct Binary {
    job: NonNull<c_void>,
    len: usize,
}

unsafe extern "C" {
    fn atlas_binary_start(
        device: *mut c_void,
        left: *const F128,
        right: *const F128,
        bytes: usize,
    ) -> *mut c_void;
    fn atlas_binary_round(
        job: *mut c_void,
        args: &BinaryArgs,
        args_bytes: usize,
        count: u32,
        binding: bool,
        low: *const F128,
        low_bytes: usize,
        high: *const F128,
        high_bytes: usize,
        output: *mut F128,
    ) -> bool;
    fn atlas_binary_snapshot(job: *mut c_void, left: *mut F128, right: *mut F128, len: usize);
    fn atlas_binary_drop(job: *mut c_void);
}

fn dense_bitz(left: &[F128], right: &[F128]) -> Option<Box<dyn world_prover::DenseRounds>> {
    if !bitz_mode("dense") || !(1 << 18..=1 << 23).contains(&left.len()) {
        return None;
    }
    let job = NonNull::new(unsafe {
        atlas_binary_start(device()?, left.as_ptr(), right.as_ptr(), size_of_val(left))
    })?;
    Some(Box::new(Binary {
        job,
        len: left.len(),
    }))
}

impl world_prover::DenseRounds for Binary {
    fn step(
        &mut self,
        previous: Option<F128>,
        at_one: bool,
        low: &[F128],
        high: &[F128],
    ) -> Option<[F128; 2]> {
        let next = self.len >> u32::from(previous.is_some());
        if next < 16384 {
            return None;
        }
        let zero = F128 { lo: 0, hi: 0 };
        let args = BinaryArgs {
            challenge: previous.unwrap_or(zero),
            len: next as u32,
            low_size: low.len() as u32,
            binding: u32::from(previous.is_some()),
            at_one: u32::from(at_one),
        };
        let mut sums = [zero; 2];
        let success = unsafe {
            atlas_binary_round(
                self.job.as_ptr(),
                &args,
                size_of::<BinaryArgs>(),
                (next / 2) as u32,
                previous.is_some(),
                low.as_ptr(),
                size_of_val(low),
                high.as_ptr(),
                size_of_val(high),
                sums.as_mut_ptr(),
            )
        };
        if !success {
            return None;
        }
        self.len = next;
        Some(sums)
    }

    fn snapshot(&self, left: &mut [F128], right: &mut [F128]) {
        assert_eq!(left.len(), self.len);
        assert_eq!(right.len(), self.len);
        unsafe {
            atlas_binary_snapshot(
                self.job.as_ptr(),
                left.as_mut_ptr(),
                right.as_mut_ptr(),
                size_of_val(left),
            );
        }
    }
}

impl Drop for Binary {
    fn drop(&mut self) {
        unsafe { atlas_binary_drop(self.job.as_ptr()) };
    }
}
