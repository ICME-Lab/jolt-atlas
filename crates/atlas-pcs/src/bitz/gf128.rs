//! The binary field `F_{2^128} = F_2[x] / (x^128 + x^7 + x^2 + x + 1)`.
//!
//! An element is its coefficient vector: bit `i` of the `u128` is the coefficient of `x^i`.
//! Addition is XOR. Multiplication is a carry-less 128 × 128 product, from three 64 × 64
//! products, followed by reduction with `x^128 = x^7 + x^2 + x + 1`; the 64 × 64 products use
//! `PMULL` on aarch64 and `PCLMULQDQ` on x86-64 when the CPU has them, and a bitwise loop
//! otherwise.
//!
//! [`GENERATOR`] generates the multiplicative group, of order `2^128 − 1`: BitZ proves integer
//! claims in its exponent.

use atlas_algebra::serialize::{Canonical, Error};
use std::io::{Read, Write};
use std::ops::{Add, AddAssign, Mul, MulAssign};

/// An element of `F_{2^128}`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Gf128(pub u128);

/// A generator of the multiplicative group `F_{2^128}^*`.
pub const GENERATOR: Gf128 = Gf128(GENERATOR_BITS);

/// The smallest element, as an integer, whose order is `2^128 − 1` (pinned by
/// `generator_has_full_order`).
const GENERATOR_BITS: u128 = 0x2;

/// The prime factors of `2^128 − 1`, the order of the multiplicative group.
pub const GROUP_ORDER_FACTORS: [u128; 9] = [
    3,
    5,
    17,
    257,
    641,
    65_537,
    274_177,
    6_700_417,
    67_280_421_310_721,
];

impl Gf128 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    /// A vector of `len` zeros. The memory comes zeroed from the allocator, so nothing is
    /// written until an element is, on whichever thread writes it.
    pub fn zeros(len: usize) -> Vec<Self> {
        let mut words = std::mem::ManuallyDrop::new(vec![0u128; len]);
        // SAFETY: `Gf128` is `repr(transparent)` over `u128`, so the two vectors have the same
        // element layout, and the pointer, length and capacity are those of a live `Vec` that
        // is not dropped.
        unsafe {
            Vec::from_raw_parts(
                words.as_mut_ptr().cast::<Self>(),
                words.len(),
                words.capacity(),
            )
        }
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn square(self) -> Self {
        self * self
    }

    /// `self^e`, by square and multiply from the low bit.
    pub fn pow(self, mut e: u128) -> Self {
        let mut base = self;
        let mut acc = Self::ONE;
        while e != 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base = base.square();
            e >>= 1;
        }
        acc
    }

    /// The multiplicative inverse, `self^(2^128 − 2)`; `None` for zero.
    pub fn inverse(self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.pow(u128::MAX - 1))
        }
    }

    /// The product computed without CPU carry-less multiply instructions.
    pub fn mul_portable(self, rhs: Self) -> Self {
        let (hi, lo) = mul_wide_portable(self.0, rhs.0);
        Self(reduce(hi, lo))
    }
}

impl Add for Gf128 {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl AddAssign for Gf128 {
    #[inline]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Mul for Gf128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(mul(self.0, rhs.0))
    }
}

impl MulAssign for Gf128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Canonical for Gf128 {
    fn encode<W: Write>(&self, writer: W) -> Result<(), Error> {
        self.0.encode(writer)
    }

    fn encoded_len(&self) -> usize {
        16
    }

    fn decode<R: Read>(reader: R) -> Result<Self, Error> {
        Ok(Self(u128::decode(reader)?))
    }
}

/// Combines the three 64 × 64 products of Karatsuba's method, `a0·b0`, `a1·b1` and
/// `(a0 + a1)·(b0 + b1)`, into the 256-bit product as `(high, low)` halves.
#[inline(always)]
fn karatsuba(low: u128, high: u128, cross: u128) -> (u128, u128) {
    let mid = cross ^ low ^ high;
    (high ^ (mid >> 64), low ^ (mid << 64))
}

/// Reduces a 256-bit polynomial `high · x^128 + low` modulo `x^128 + x^7 + x^2 + x + 1`.
#[inline(always)]
fn reduce(hi: u128, lo: u128) -> u128 {
    // high · (x^7 + x^2 + x + 1) has degree at most 134: its low 128 bits, and the seven bits
    // that overflow, which are folded once more.
    let folded = hi ^ (hi << 1) ^ (hi << 2) ^ (hi << 7);
    let overflow = (hi >> 127) ^ (hi >> 126) ^ (hi >> 121);
    lo ^ folded ^ overflow ^ (overflow << 1) ^ (overflow << 2) ^ (overflow << 7)
}

/// The carry-less product of two 64-bit polynomials, one bit of `b` at a time.
fn clmul64_portable(a: u64, b: u64) -> u128 {
    let a = a as u128;
    let mut acc = 0u128;
    let mut b = b;
    while b != 0 {
        let i = b.trailing_zeros();
        acc ^= a << i;
        b &= b - 1;
    }
    acc
}

fn mul_wide_portable(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    karatsuba(
        clmul64_portable(a0, b0),
        clmul64_portable(a1, b1),
        clmul64_portable(a0 ^ a1, b0 ^ b1),
    )
}

/// The unreduced product, in one call whether or not the build enables the CPU's carry-less multiply:
/// with the feature enabled at compile time the accelerated body is inlined, and otherwise
/// it is reached through one runtime-detected call for the whole multiplication.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    #[inline]
    #[target_feature(enable = "neon,aes")]
    unsafe fn accelerated(a: u128, b: u128) -> (u128, u128) {
        use core::arch::aarch64::vmull_p64;
        let (a0, a1) = (a as u64, (a >> 64) as u64);
        let (b0, b1) = (b as u64, (b >> 64) as u64);
        karatsuba(
            vmull_p64(a0, b0),
            vmull_p64(a1, b1),
            vmull_p64(a0 ^ a1, b0 ^ b1),
        )
    }
    if cfg!(target_feature = "aes") || std::arch::is_aarch64_feature_detected!("aes") {
        // SAFETY: the `aes` feature, which provides `PMULL`, is present on this CPU.
        unsafe { accelerated(a, b) }
    } else {
        mul_wide_portable(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    #[inline]
    #[target_feature(enable = "pclmulqdq,sse2")]
    unsafe fn accelerated(a: u128, b: u128) -> (u128, u128) {
        use core::arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_set_epi64x};
        let clmul = |x: u64, y: u64| -> u128 {
            let product: __m128i =
                _mm_clmulepi64_si128(_mm_set_epi64x(0, x as i64), _mm_set_epi64x(0, y as i64), 0);
            core::mem::transmute::<__m128i, u128>(product)
        };
        let (a0, a1) = (a as u64, (a >> 64) as u64);
        let (b0, b1) = (b as u64, (b >> 64) as u64);
        karatsuba(clmul(a0, b0), clmul(a1, b1), clmul(a0 ^ a1, b0 ^ b1))
    }
    if cfg!(target_feature = "pclmulqdq") || std::arch::is_x86_feature_detected!("pclmulqdq") {
        // SAFETY: `PCLMULQDQ` is present on this CPU; SSE2 is part of x86-64.
        unsafe { accelerated(a, b) }
    } else {
        mul_wide_portable(a, b)
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    mul_wide_portable(a, b)
}

/// The product of two elements.
#[inline(always)]
fn mul(a: u128, b: u128) -> u128 {
    let (hi, lo) = mul_wide(a, b);
    reduce(hi, lo)
}

/// A sum of products, kept unreduced: reduction is `F_2`-linear, so the 256-bit carry-less
/// products are added as they are and reduced once.
#[derive(Clone, Copy, Debug, Default)]
pub struct WideSum {
    hi: u128,
    lo: u128,
}

impl WideSum {
    pub const ZERO: Self = Self { hi: 0, lo: 0 };

    /// Adds `a · b`.
    #[inline(always)]
    pub fn add_product(&mut self, a: Gf128, b: Gf128) {
        let (hi, lo) = mul_wide(a.0, b.0);
        self.hi ^= hi;
        self.lo ^= lo;
    }

    /// The sum of two sums.
    #[inline]
    pub fn plus(self, other: Self) -> Self {
        Self {
            hi: self.hi ^ other.hi,
            lo: self.lo ^ other.lo,
        }
    }

    /// The sum as a field element.
    #[inline]
    pub fn reduce(self) -> Gf128 {
        Gf128(reduce(self.hi, self.lo))
    }
}

/// The sums `Σ_i p0_i·q0_i` and `Σ_i (p0_i + p1_i)·(q0_i + q1_i)` over slices of one length:
/// the value at zero and the leading coefficient of `Σ_i P_i(X)·Q_i(X)` for the linear `P_i`,
/// `Q_i` through the given values at zero and one.
pub fn product_sums(p0: &[Gf128], p1: &[Gf128], q0: &[Gf128], q1: &[Gf128]) -> [Gf128; 2] {
    let n = p0.len();
    assert!(p1.len() == n && q0.len() == n && q1.len() == n);
    #[cfg(target_arch = "aarch64")]
    if cfg!(target_feature = "aes") || std::arch::is_aarch64_feature_detected!("aes") {
        // SAFETY: the `aes` feature, which provides `PMULL`, is present on this CPU.
        return unsafe { neon::product_sums(p0, p1, q0, q1) };
    }
    let mut sums = [WideSum::ZERO; 2];
    for i in 0..n {
        sums[0].add_product(p0[i], q0[i]);
        sums[1].add_product(p0[i] + p1[i], q0[i] + q1[i]);
    }
    [sums[0].reduce(), sums[1].reduce()]
}

/// Adds to `sums[i]` the two products [`product_sums`] takes of index `i`, unreduced.
pub fn add_products(
    sums: &mut [[WideSum; 2]],
    p0: &[Gf128],
    p1: &[Gf128],
    q0: &[Gf128],
    q1: &[Gf128],
) {
    let n = sums.len();
    assert!(p0.len() == n && p1.len() == n && q0.len() == n && q1.len() == n);
    #[cfg(target_arch = "aarch64")]
    if cfg!(target_feature = "aes") || std::arch::is_aarch64_feature_detected!("aes") {
        // SAFETY: the `aes` feature, which provides `PMULL`, is present on this CPU.
        return unsafe { neon::add_products(sums, p0, p1, q0, q1) };
    }
    for (i, sum) in sums.iter_mut().enumerate() {
        sum[0].add_product(p0[i], q0[i]);
        sum[1].add_product(p0[i] + p1[i], q0[i] + q1[i]);
    }
}

/// `values_i ← values_i · factors_i`.
pub fn scale_pairs(values: &mut [Gf128], factors: &[Gf128]) {
    assert_eq!(values.len(), factors.len());
    #[cfg(target_arch = "aarch64")]
    if cfg!(target_feature = "aes") || std::arch::is_aarch64_feature_detected!("aes") {
        // SAFETY: the `aes` feature, which provides `PMULL`, is present on this CPU.
        return unsafe { neon::scale_pairs(values, factors) };
    }
    for (value, &factor) in values.iter_mut().zip(factors) {
        *value *= factor;
    }
}

/// The inverse of every element, with one inversion. `None` if an element is zero.
pub fn inverses(values: &[Gf128]) -> Option<Vec<Gf128>> {
    let mut prefix = Vec::with_capacity(values.len());
    let mut product = Gf128::ONE;
    for &value in values {
        prefix.push(product);
        product *= value;
    }
    let mut suffix_inverse = product.inverse()?;
    let mut out = prefix;
    for (slot, &value) in out.iter_mut().zip(values).rev() {
        *slot *= suffix_inverse;
        suffix_inverse *= value;
    }
    Some(out)
}

/// `low_i ← low_i + r·(low_i + high_i)`: the linear function through `low_i` at zero and
/// `high_i` at one, at `r`, written over `low`.
pub fn fold_into_low(low: &mut [Gf128], high: &[Gf128], r: Gf128) {
    assert_eq!(low.len(), high.len());
    #[cfg(target_arch = "aarch64")]
    if cfg!(target_feature = "aes") || std::arch::is_aarch64_feature_detected!("aes") {
        // SAFETY: the `aes` feature, which provides `PMULL`, is present on this CPU.
        return unsafe { neon::fold_into_low(low, high, r) };
    }
    for (x, &y) in low.iter_mut().zip(high) {
        *x += r * (*x + y);
    }
}

/// `out_i = a_i · b_i`.
pub fn mul_pairs(out: &mut [Gf128], a: &[Gf128], b: &[Gf128]) {
    assert!(a.len() == out.len() && b.len() == out.len());
    #[cfg(target_arch = "aarch64")]
    if cfg!(target_feature = "aes") || std::arch::is_aarch64_feature_detected!("aes") {
        // SAFETY: the `aes` feature, which provides `PMULL`, is present on this CPU.
        return unsafe { neon::mul_pairs(out, a, b) };
    }
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x * y;
    }
}

/// The slice kernels with every element held in a vector register from load to store: a
/// multiplication called on its own returns its 256-bit product through memory.
#[cfg(target_arch = "aarch64")]
mod neon {
    #![allow(clippy::indexing_slicing)]
    use super::{Gf128, WideSum};
    use core::arch::aarch64::{
        uint64x2_t, vdupq_n_u64, veorq_u64, vextq_u64, vgetq_lane_u64, vld1q_u64, vmull_high_p64,
        vmull_p64, vreinterpretq_p64_u64, vreinterpretq_u64_p128, vst1q_u64,
    };

    /// The element as `[low 64 bits, high 64 bits]`.
    #[inline(always)]
    unsafe fn load(x: &Gf128) -> uint64x2_t {
        // SAFETY: a `Gf128` is a `u128`: sixteen readable bytes, low half first.
        unsafe { vld1q_u64((x as *const Gf128).cast()) }
    }

    #[inline(always)]
    unsafe fn store(out: &mut Gf128, value: uint64x2_t) {
        // SAFETY: a `Gf128` is a `u128`: sixteen writable bytes, low half first.
        unsafe { vst1q_u64((out as *mut Gf128).cast(), value) }
    }

    /// The carry-less product of the low halves.
    #[inline]
    #[target_feature(enable = "neon,aes")]
    fn clmul_low(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0)))
    }

    /// The carry-less product of the high halves.
    #[inline]
    #[target_feature(enable = "neon,aes")]
    fn clmul_high(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(a),
            vreinterpretq_p64_u64(b),
        ))
    }

    /// The unreduced product as `(high 128 bits, low 128 bits)`, by Karatsuba.
    #[inline]
    #[target_feature(enable = "neon,aes")]
    fn wide(a: uint64x2_t, b: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
        let zero = vdupq_n_u64(0);
        let (low, high) = (clmul_low(a, b), clmul_high(a, b));
        let cross = clmul_low(
            veorq_u64(a, vextq_u64(a, a, 1)),
            veorq_u64(b, vextq_u64(b, b, 1)),
        );
        let mid = veorq_u64(veorq_u64(cross, low), high);
        (
            veorq_u64(high, vextq_u64(mid, zero, 1)),
            veorq_u64(low, vextq_u64(zero, mid, 1)),
        )
    }

    /// `high · x^128 + low` modulo `x^128 + x^7 + x^2 + x + 1`: the high half of `high`
    /// times `x^7 + x^2 + x + 1` lands on bit 64 and overflows by seven bits, which join the
    /// low half of `high` for the second multiplication.
    #[inline]
    #[target_feature(enable = "neon,aes")]
    fn reduce(high: uint64x2_t, low: uint64x2_t) -> uint64x2_t {
        let zero = vdupq_n_u64(0);
        let modulus = vdupq_n_u64(0x87);
        let upper = clmul_high(high, modulus);
        let low = veorq_u64(low, vextq_u64(zero, upper, 1));
        let rest = veorq_u64(high, vextq_u64(upper, zero, 1));
        veorq_u64(low, clmul_low(rest, modulus))
    }

    #[inline]
    #[target_feature(enable = "neon,aes")]
    fn mul(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        let (high, low) = wide(a, b);
        reduce(high, low)
    }

    #[target_feature(enable = "neon,aes")]
    pub(super) unsafe fn product_sums(
        p0: &[Gf128],
        p1: &[Gf128],
        q0: &[Gf128],
        q1: &[Gf128],
    ) -> [Gf128; 2] {
        // SAFETY: this function has `PMULL`.
        unsafe {
            let zero = vdupq_n_u64(0);
            let (mut zero_high, mut zero_low) = (zero, zero);
            let (mut lead_high, mut lead_low) = (zero, zero);
            for i in 0..p0.len() {
                let (a0, a1) = (load(&p0[i]), load(&p1[i]));
                let (b0, b1) = (load(&q0[i]), load(&q1[i]));
                let (high, low) = wide(a0, b0);
                zero_high = veorq_u64(zero_high, high);
                zero_low = veorq_u64(zero_low, low);
                let (high, low) = wide(veorq_u64(a0, a1), veorq_u64(b0, b1));
                lead_high = veorq_u64(lead_high, high);
                lead_low = veorq_u64(lead_low, low);
            }
            let mut sums = [Gf128::ZERO; 2];
            store(&mut sums[0], reduce(zero_high, zero_low));
            store(&mut sums[1], reduce(lead_high, lead_low));
            sums
        }
    }

    /// Adds the unreduced product `(high, low)` to `sum`.
    #[inline(always)]
    unsafe fn accumulate(sum: &mut WideSum, (high, low): (uint64x2_t, uint64x2_t)) {
        // SAFETY: each half of a `WideSum` is a `u128`: sixteen bytes, low half first.
        unsafe {
            let hi = (&mut sum.hi as *mut u128).cast::<u64>();
            let lo = (&mut sum.lo as *mut u128).cast::<u64>();
            vst1q_u64(hi, veorq_u64(vld1q_u64(hi), high));
            vst1q_u64(lo, veorq_u64(vld1q_u64(lo), low));
        }
    }

    #[target_feature(enable = "neon,aes")]
    pub(super) unsafe fn add_products(
        sums: &mut [[WideSum; 2]],
        p0: &[Gf128],
        p1: &[Gf128],
        q0: &[Gf128],
        q1: &[Gf128],
    ) {
        // SAFETY: this function has `PMULL`.
        unsafe {
            for (i, sum) in sums.iter_mut().enumerate() {
                let (a0, a1) = (load(&p0[i]), load(&p1[i]));
                let (b0, b1) = (load(&q0[i]), load(&q1[i]));
                accumulate(&mut sum[0], wide(a0, b0));
                accumulate(&mut sum[1], wide(veorq_u64(a0, a1), veorq_u64(b0, b1)));
            }
        }
    }

    #[target_feature(enable = "neon,aes")]
    pub(super) unsafe fn scale_pairs(values: &mut [Gf128], factors: &[Gf128]) {
        // SAFETY: this function has `PMULL`.
        unsafe {
            for (value, factor) in values.iter_mut().zip(factors) {
                let product = mul(load(value), load(factor));
                store(value, product);
            }
        }
    }

    #[target_feature(enable = "neon,aes")]
    pub(super) unsafe fn fold_into_low(low: &mut [Gf128], high: &[Gf128], r: Gf128) {
        // SAFETY: this function has `PMULL`.
        unsafe {
            let r = load(&r);
            for (x, y) in low.iter_mut().zip(high) {
                let (a, b) = (load(x), load(y));
                store(x, veorq_u64(a, mul(r, veorq_u64(a, b))));
            }
        }
    }

    #[target_feature(enable = "neon,aes")]
    pub(super) unsafe fn mul_pairs(out: &mut [Gf128], a: &[Gf128], b: &[Gf128]) {
        // SAFETY: this function has `PMULL`.
        unsafe {
            for i in 0..out.len() {
                store(&mut out[i], mul(load(&a[i]), load(&b[i])));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_elements(rng: &mut StdRng, n: usize) -> Vec<Gf128> {
        (0..n)
            .map(|_| Gf128((rng.next_u64() as u128) << 64 | rng.next_u64() as u128))
            .collect()
    }

    /// The slice kernels against the element operations they stand for.
    #[test]
    fn slice_kernels_are_the_element_operations() {
        let mut rng = StdRng::seed_from_u64(0x51CE);
        for n in [0usize, 1, 7, 64] {
            let [w, p0, p1, q0, q1] = [(); 5].map(|()| random_elements(&mut rng, n));
            let mut want = [Gf128::ZERO; 2];
            for i in 0..n {
                want[0] += p0[i] * q0[i];
                want[1] += (p0[i] + p1[i]) * (q0[i] + q1[i]);
            }
            assert_eq!(product_sums(&p0, &p1, &q0, &q1), want);

            // Accumulated twice, each sum is its two products twice: zero.
            let mut sums = vec![[WideSum::ZERO; 2]; n];
            add_products(&mut sums, &p0, &p1, &q0, &q1);
            for i in 0..n {
                assert_eq!(sums[i][0].reduce(), p0[i] * q0[i]);
                assert_eq!(sums[i][1].reduce(), (p0[i] + p1[i]) * (q0[i] + q1[i]));
            }
            add_products(&mut sums, &p0, &p1, &q0, &q1);
            assert!(sums.iter().flatten().all(|sum| sum.reduce().is_zero()));

            let mut scaled = p0.clone();
            scale_pairs(&mut scaled, &w);
            let inverse = inverses(&w).expect("random elements are nonzero");
            for i in 0..n {
                assert_eq!(scaled[i], p0[i] * w[i]);
                assert_eq!(inverse[i] * w[i], Gf128::ONE);
            }
            assert!(n == 0 || inverses(&[w[0], Gf128::ZERO]).is_none());

            let r = random_elements(&mut rng, 1)[0];
            let mut folded = p0.clone();
            fold_into_low(&mut folded, &p1, r);
            let mut products = vec![Gf128::ZERO; n];
            mul_pairs(&mut products, &q0, &q1);
            for i in 0..n {
                assert_eq!(folded[i], p0[i] + r * (p0[i] + p1[i]));
                assert_eq!(products[i], q0[i] * q1[i]);
            }
        }
    }
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    /// `x^128` reduced: `x^7 + x^2 + x + 1`.
    const REDUCTION: u128 = 0x87;

    /// Multiplication from the definition: add `a · x^i` for every set bit `i` of `b`, where
    /// multiplying by `x` shifts and, on overflow, adds `x^7 + x^2 + x + 1`.
    fn mul_reference(a: Gf128, b: Gf128) -> Gf128 {
        let (mut a, b) = (a.0, b.0);
        let mut acc = 0u128;
        for i in 0..128 {
            if (b >> i) & 1 == 1 {
                acc ^= a;
            }
            let carry = a >> 127;
            a <<= 1;
            if carry == 1 {
                a ^= REDUCTION;
            }
        }
        Gf128(acc)
    }

    fn random(rng: &mut StdRng) -> Gf128 {
        Gf128(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128)
    }

    fn samples() -> Vec<Gf128> {
        let mut rng = StdRng::seed_from_u64(0x6f128);
        let mut values = vec![
            Gf128::ZERO,
            Gf128::ONE,
            Gf128(2),
            Gf128(REDUCTION),
            Gf128(1 << 127),
            Gf128(u128::MAX),
            Gf128(u64::MAX as u128),
            Gf128((u64::MAX as u128) << 64),
        ];
        values.extend((0..64).map(|_| random(&mut rng)));
        values
    }

    #[test]
    fn multiplication_matches_the_definition() {
        let values = samples();
        for &a in &values {
            for &b in &values {
                let expected = mul_reference(a, b);
                assert_eq!(a * b, expected, "{a:?} * {b:?}");
                assert_eq!(a.mul_portable(b), expected, "portable {a:?} * {b:?}");
            }
        }
    }

    #[test]
    fn field_laws_hold() {
        let values = samples();
        for &a in &values {
            assert_eq!(a * Gf128::ONE, a);
            assert_eq!(a + a, Gf128::ZERO);
            assert_eq!(a.square(), mul_reference(a, a));
            match a.inverse() {
                Some(inv) => assert_eq!(mul_reference(a, inv), Gf128::ONE),
                None => assert!(a.is_zero()),
            }
        }
        for w in values.windows(3) {
            let (a, b, c) = (w[0], w[1], w[2]);
            assert_eq!(a * b, b * a);
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn generator_has_full_order() {
        let order = u128::MAX;
        assert_eq!(GROUP_ORDER_FACTORS.iter().product::<u128>(), order);
        let has_full_order = |g: Gf128| {
            g.pow(order) == Gf128::ONE
                && GROUP_ORDER_FACTORS
                    .iter()
                    .all(|p| g.pow(order / p) != Gf128::ONE)
        };
        assert!(has_full_order(GENERATOR));
        // The pinned generator is the smallest such element.
        assert!((2..GENERATOR_BITS).all(|g| !has_full_order(Gf128(g))));
    }

    #[test]
    fn wide_sums_reduce_to_the_sum_of_products() {
        let values = samples();
        let mut wide = WideSum::ZERO;
        let mut expected = Gf128::ZERO;
        for pair in values.chunks_exact(2) {
            wide.add_product(pair[0], pair[1]);
            expected += mul_reference(pair[0], pair[1]);
        }
        assert_eq!(wide.reduce(), expected);
        assert_eq!(wide.plus(wide).reduce(), Gf128::ZERO);
        assert_eq!(WideSum::ZERO.reduce(), Gf128::ZERO);
    }

    #[test]
    fn zero_vectors_are_zero_and_writable() {
        let mut zeros = Gf128::zeros(1000);
        assert_eq!(zeros.len(), 1000);
        assert!(zeros.iter().all(|z| z.is_zero()));
        zeros[999] = Gf128::ONE;
        zeros.push(Gf128(7));
        assert_eq!((zeros[999], zeros[1000]), (Gf128::ONE, Gf128(7)));
        assert!(Gf128::zeros(0).is_empty());
    }

    #[test]
    fn encoding_round_trips() {
        for a in samples() {
            let bytes = atlas_algebra::serialize::to_bytes(&a).unwrap();
            assert_eq!(bytes.len(), 16);
            assert_eq!(
                atlas_algebra::serialize::from_bytes::<Gf128>(&bytes).unwrap(),
                a
            );
        }
    }
}
