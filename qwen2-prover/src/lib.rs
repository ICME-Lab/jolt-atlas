pub mod cpu;
pub mod float;
pub mod illm;
pub mod lut;
pub mod rand;
pub mod rebase;
pub mod text;
pub mod weights;

pub const SEQ: usize = 32;
pub const HIDDEN: usize = 896;
pub const INTERMEDIATE: usize = 4864;
pub const LAYERS: usize = 24;
pub const HEADS: usize = 14;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = HIDDEN / HEADS;
pub const KV_GROUP: usize = HEADS / KV_HEADS;
pub const VOCAB: usize = 151_936;
pub const ROPE_THETA: f64 = 1_000_000.0;

pub const X_LEN: usize = SEQ * HIDDEN;
pub const Q_LEN: usize = SEQ * HEADS * HEAD_DIM;
pub const KV_LEN: usize = SEQ * KV_HEADS * HEAD_DIM;
pub const SCORE_LEN: usize = HEADS * SEQ * SEQ;
pub const MLP_LEN: usize = SEQ * INTERMEDIATE;

pub const ADD: &str = include_str!("kernels/add.wgsl");
pub const COPY: &str = include_str!("kernels/copy.wgsl");
pub const MATMUL: &str = include_str!("kernels/matmul.wgsl");
pub const MUL: &str = include_str!("kernels/mul.wgsl");
pub const RMS_NORM: &str = include_str!("kernels/rms_norm.wgsl");
pub const ROPE: &str = include_str!("kernels/rope.wgsl");
pub const SILU_MUL: &str = include_str!("kernels/silu_mul.wgsl");
pub const SOFTMAX: &str = include_str!("kernels/softmax.wgsl");
pub const SCORE_QK: &str = include_str!("kernels/score_qk.wgsl");
pub const ATTN_V: &str = include_str!("kernels/attn_v.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen2_0_5b_shapes() {
        assert_eq!(HEAD_DIM, 64);
        assert_eq!(KV_GROUP, 7);
        assert_eq!(X_LEN, 28_672);
        assert_eq!(Q_LEN, X_LEN);
        assert_eq!(KV_LEN, 4_096);
        assert_eq!(SCORE_LEN, 14_336);
        assert_eq!(MLP_LEN, 155_648);
    }
}
