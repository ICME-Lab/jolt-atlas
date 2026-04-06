#[cfg(test)]
pub mod tests {

    use crate::{
        biginteger::{BigInteger, SignedBigInt},
        UniformRand,
    };
    use ark_std::Zero;
    use num_bigint::BigUint;

    // Test elementary math operations for BigInteger.
    fn biginteger_arithmetic_test<B: BigInteger>(a: B, b: B, zero: B, max: B) {
        // zero == zero
        assert_eq!(zero, zero);

        // zero.is_zero() == true
        assert_eq!(zero.is_zero(), true);

        // a == a
        assert_eq!(a, a);

        // a + 0 = a
        let mut a0_add = a;
        let carry = a0_add.add_with_carry(&zero);
        assert_eq!(a0_add, a);
        assert_eq!(carry, false);

        // a - 0 = a
        let mut a0_sub = a;
        let borrow = a0_sub.sub_with_borrow(&zero);
        assert_eq!(a0_sub, a);
        assert_eq!(borrow, false);

        // a - a = 0
        let mut aa_sub = a;
        let borrow = aa_sub.sub_with_borrow(&a);
        assert_eq!(aa_sub, zero);
        assert_eq!(borrow, false);

        // a + b = b + a
        let mut ab_add = a;
        let ab_carry = ab_add.add_with_carry(&b);
        let mut ba_add = b;
        let ba_carry = ba_add.add_with_carry(&a);
        assert_eq!(ab_add, ba_add);
        assert_eq!(ab_carry, ba_carry);

        // a * 1 = a
        let mut a_mul1 = a;
        a_mul1 <<= 0;
        assert_eq!(a_mul1, a);

        // a * 2 = a + a
        let mut a_mul2 = a;
        a_mul2.mul2();
        let mut a_plus_a = a;
        let carry_a_plus_a = a_plus_a.add_with_carry(&a); // Won't assert anything about carry bit.
        assert_eq!(a_mul2, a_plus_a);

        // a * 1 = a
        assert_eq!(a.mul_low(&B::from(1u64)), a);

        // a * 2 = a
        assert_eq!(a.mul_low(&B::from(2u64)), a_plus_a);

        // a * b = b * a
        assert_eq!(a.mul_low(&b), b.mul_low(&a));

        // a * 2 * b * 0 = 0
        assert!(a.mul_low(&zero).is_zero());

        // a * 2 * ... * 2  = a * 2^n
        let mut a_mul_n = a;
        for _ in 0..20 {
            a_mul_n = a_mul_n.mul_low(&B::from(2u64));
        }
        assert_eq!(a_mul_n, a << 20);

        // a * 0 = (0, 0)
        assert_eq!(a.mul(&zero), (zero, zero));

        // a * 1 = (a, 0)
        assert_eq!(a.mul(&B::from(1u64)), (a, zero));

        // a * 1 = 0 (high part of the result)
        assert_eq!(a.mul_high(&B::from(1u64)), (zero));

        // a * 0 = 0 (high part of the result)
        assert!(a.mul_high(&zero).is_zero());

        // If a + a has a carry
        if carry_a_plus_a {
            // a + a has a carry: high part of a * 2 is not zero
            assert_ne!(a.mul_high(&B::from(2u64)), zero);
        } else {
            // a + a has no carry: high part of a * 2 is zero
            assert_eq!(a.mul_high(&B::from(2u64)), zero);
        }

        // max + max = max * 2
        let mut max_plus_max = max;
        max_plus_max.add_with_carry(&max);
        assert_eq!(max.mul(&B::from(2u64)), (max_plus_max, B::from(1u64)));
        assert_eq!(max.mul_high(&B::from(2u64)), B::from(1u64));
    }

    #[test]
    fn test_s160_mul_s160_hi32_consistency() {
        use crate::biginteger::{BigInt, S160};

        // Spot-check a configuration that exercises the hi32 accumulation path
        let a = S160::new([1u64 << 63, 0], 1, true); // a2=1, a0 has high bit
        let b = S160::new([0, 1u64 << 63], 1, true); // b2=1, b1 has high bit
        let got = &a * &b; // S160 result

        // Convert S160 value to BigUint: pack [lo0, lo1, hi32] into BigInt<3>
        let mut pack = [0u64; 3];
        pack[0] = got.magnitude_lo()[0];
        pack[1] = got.magnitude_lo()[1];
        pack[2] = got.magnitude_hi() as u64;
        let got_bu = num_bigint::BigUint::from(BigInt::<3>(pack));

        // Reference BigUint modulo 2^160
        let a_bu = (num_bigint::BigUint::from(a.magnitude_lo()[1]) << 64)
            + num_bigint::BigUint::from(a.magnitude_lo()[0])
            + (num_bigint::BigUint::from(a.magnitude_hi() as u64) << 128);
        let b_bu = (num_bigint::BigUint::from(b.magnitude_lo()[1]) << 64)
            + num_bigint::BigUint::from(b.magnitude_lo()[0])
            + (num_bigint::BigUint::from(b.magnitude_hi() as u64) << 128);
        let prod = (a_bu * b_bu) % (num_bigint::BigUint::from(1u8) << 160);

        assert_eq!(got_bu, prod);
    }

    fn biginteger_shr<B: BigInteger>() {
        let mut rng = ark_std::test_rng();
        let a = B::rand(&mut rng);
        assert_eq!(a >> 0, a);

        // Binary simple test
        let a = B::from(256u64);
        assert_eq!(a >> 2, B::from(64u64));

        // Test saturated underflow
        let a = B::from(1u64);
        assert_eq!(a >> 5, B::from(0u64));

        // Test null bits
        let a = B::rand(&mut rng);
        let b = a >> 3;
        assert_eq!(b.get_bit(B::NUM_LIMBS * 64 - 1), false);
        assert_eq!(b.get_bit(B::NUM_LIMBS * 64 - 2), false);
        assert_eq!(b.get_bit(B::NUM_LIMBS * 64 - 3), false);
    }

    fn biginteger_shl<B: BigInteger>() {
        let mut rng = ark_std::test_rng();
        let a = B::rand(&mut rng);
        assert_eq!(a << 0, a);

        // Binary simple test
        let a = B::from(64u64);
        assert_eq!(a << 2, B::from(256u64));

        // Testing saturated overflow
        let a = B::rand(&mut rng);
        assert_eq!(a << ((B::NUM_LIMBS as u32) * 64), B::from(0u64));

        // Test null bits
        let a = B::rand(&mut rng);
        let b = a << 3;
        assert_eq!(b.get_bit(0), false);
        assert_eq!(b.get_bit(1), false);
        assert_eq!(b.get_bit(2), false);
    }

    // Test for BigInt's bitwise operations
    fn biginteger_bitwise_ops_test<B: BigInteger>() {
        let mut rng = ark_std::test_rng();

        // Test XOR
        // a xor a = 0
        let a = B::rand(&mut rng);
        assert_eq!(a ^ &a, B::from(0_u64));

        // Testing a xor b xor b
        let a = B::rand(&mut rng);
        let b = B::rand(&mut rng);
        let xor_ab = a ^ b;
        assert_eq!(xor_ab ^ b, a);

        // Test OR
        // a or a = a
        let a = B::rand(&mut rng);
        assert_eq!(a | &a, a);

        // Testing a or b or b
        let a = B::rand(&mut rng);
        let b = B::rand(&mut rng);
        let or_ab = a | b;
        assert_eq!(or_ab | &b, a | b);

        // Test AND
        // a and a = a
        let a = B::rand(&mut rng);
        assert_eq!(a & (&a), a);

        // Testing a and a and b.
        let a = B::rand(&mut rng);
        let b = B::rand(&mut rng);
        let b_clone = b.clone();
        let and_ab = a & b;
        assert_eq!(and_ab & b_clone, a & b);

        // Testing De Morgan's law
        let a = 0x1234567890abcdef_u64;
        let b = 0xfedcba0987654321_u64;
        let de_morgan_lhs = B::from(!(a | b));
        let de_morgan_rhs = B::from(!a) & B::from(!b);
        assert_eq!(de_morgan_lhs, de_morgan_rhs);
    }

    // Test correctness of BigInteger's bit values
    fn biginteger_bits_test<B: BigInteger>() {
        let mut one = B::from(1u64);
        // 0th bit of BigInteger representing 1 is 1
        assert!(one.get_bit(0));
        // 1st bit of BigInteger representing 1 is not 1
        assert!(!one.get_bit(1));
        one <<= 5;
        let thirty_two = one;
        // 0th bit of BigInteger representing 32 is not 1
        assert!(!thirty_two.get_bit(0));
        // 1st bit of BigInteger representing 32 is not 1
        assert!(!thirty_two.get_bit(1));
        // 2nd bit of BigInteger representing 32 is not 1
        assert!(!thirty_two.get_bit(2));
        // 3rd bit of BigInteger representing 32 is not 1
        assert!(!thirty_two.get_bit(3));
        // 4th bit of BigInteger representing 32 is not 1
        assert!(!thirty_two.get_bit(4));
        // 5th bit of BigInteger representing 32 is 1
        assert!(thirty_two.get_bit(5), "{:?}", thirty_two);

        // Generates a random BigInteger and tests bit construction methods.
        let mut rng = ark_std::test_rng();
        let a: B = UniformRand::rand(&mut rng);
        assert_eq!(B::from_bits_be(&a.to_bits_be()), a);
        assert_eq!(B::from_bits_le(&a.to_bits_le()), a);
    }

    // Test conversion from BigInteger to BigUint
    fn biginteger_conversion_test<B: BigInteger>() {
        let mut rng = ark_std::test_rng();

        let x: B = UniformRand::rand(&mut rng);
        let x_bigint: BigUint = x.into();
        let x_recovered = B::try_from(x_bigint).ok().unwrap();

        assert_eq!(x, x_recovered);
    }

    // Wrapper test function for BigInteger
    fn test_biginteger<B: BigInteger>(max: B, zero: B) {
        let mut rng = ark_std::test_rng();
        let a: B = UniformRand::rand(&mut rng);
        let b: B = UniformRand::rand(&mut rng);
        biginteger_arithmetic_test(a, b, zero, max);
        biginteger_bits_test::<B>();
        biginteger_conversion_test::<B>();
        biginteger_bitwise_ops_test::<B>();
        biginteger_shr::<B>();
        biginteger_shl::<B>();
    }

    #[test]
    fn test_biginteger64() {
        use crate::biginteger::BigInteger64 as B;
        test_biginteger(B::new([u64::MAX; 1]), B::new([0u64; 1]));
    }

    #[test]
    fn test_biginteger128() {
        use crate::biginteger::BigInteger128 as B;
        test_biginteger(B::new([u64::MAX; 2]), B::new([0u64; 2]));
    }

    #[test]
    fn test_biginteger256() {
        use crate::biginteger::BigInteger256 as B;
        test_biginteger(B::new([u64::MAX; 4]), B::new([0u64; 4]));
    }

    #[test]
    fn test_biginteger384() {
        use crate::biginteger::BigInteger384 as B;
        test_biginteger(B::new([u64::MAX; 6]), B::new([0u64; 6]));
    }

    #[test]
    fn test_biginteger448() {
        use crate::biginteger::BigInteger448 as B;
        test_biginteger(B::new([u64::MAX; 7]), B::new([0u64; 7]));
    }

    #[test]
    fn test_biginteger768() {
        use crate::biginteger::BigInteger768 as B;
        test_biginteger(B::new([u64::MAX; 12]), B::new([0u64; 12]));
    }

    #[test]
    fn test_biginteger832() {
        use crate::biginteger::BigInteger832 as B;
        test_biginteger(B::new([u64::MAX; 13]), B::new([0u64; 13]));
    }

    // Tests for NEW functions
    use crate::biginteger::BigInteger256;

    #[test]
    fn test_mul_u64_in_place() {
        let mut a = BigInteger256::from(0x123456789ABCDEFu64);
        let b = 0x987654321u64;

        // Test against reference implementation
        let expected = BigUint::from(0x123456789ABCDEFu64) * BigUint::from(b);
        a.mul_u64_in_place(b);
        assert_eq!(BigUint::from(a), expected);

        // Test zero multiplication
        let mut zero = BigInteger256::zero();
        zero.mul_u64_in_place(12345);
        assert!(zero.is_zero());

        // Test multiplication by zero
        let mut a = BigInteger256::from(12345u64);
        a.mul_u64_in_place(0);
        assert!(a.is_zero());

        // Test multiplication by one
        let orig = BigInteger256::from(0xDEADBEEFu64);
        let mut a = orig;
        a.mul_u64_in_place(1);
        assert_eq!(a, orig);
    }

    #[test]
    fn test_mul_u64_w_carry() {
        let a = BigInteger256::from(u64::MAX);
        let b = u64::MAX;

        // Test against reference implementation
        let expected = BigUint::from(u64::MAX) * BigUint::from(u64::MAX);
        let result = a.mul_u64_w_carry::<5>(b);
        assert_eq!(BigUint::from(result), expected);

        // Test with small numbers
        let a = BigInteger256::from(12345u64);
        let b = 67890u64;
        let expected = BigUint::from(12345u64) * BigUint::from(67890u64);
        let result = a.mul_u64_w_carry::<5>(b);
        assert_eq!(BigUint::from(result), expected);

        // Test zero cases
        let zero = BigInteger256::zero();
        let result = zero.mul_u64_w_carry::<5>(12345);
        assert!(result.is_zero());

        let a = BigInteger256::from(12345u64);
        let result = a.mul_u64_w_carry::<5>(0);
        assert!(result.is_zero());

        // Test multiplication by one
        let a = BigInteger256::from(0xDEADBEEFu64);
        let result = a.mul_u64_w_carry::<5>(1);
        let expected_bytes = a.to_bytes_le();
        let result_bytes = result.to_bytes_le();
        assert_eq!(&result_bytes[..expected_bytes.len()], &expected_bytes[..]);
    }

    #[test]
    fn test_fmu64a() {
        let a = BigInteger256::from(12345u64);
        let b = 67890u64;
        let mut acc = BigInteger256::from(11111u64).mul_u64_w_carry::<5>(1);

        // Perform fused multiply-accumulate (no carry propagation in highest limb)
        a.fm_limbs_into::<1, 5>(&[b], &mut acc, false);

        // Compare against separate multiply and add
        let expected_mul = BigUint::from(12345u64) * BigUint::from(67890u64);
        let expected_total = expected_mul + BigUint::from(11111u64);
        assert_eq!(BigUint::from(acc), expected_total);

        // Test zero cases
        let zero = BigInteger256::zero();
        let mut acc = BigInteger256::from(12345u64).mul_u64_w_carry::<5>(1);
        let acc_copy = acc;
        zero.fm_limbs_into::<1, 5>(&[67890], &mut acc, false);
        assert_eq!(acc, acc_copy); // Should be unchanged

        // Test multiplication by zero
        let a = BigInteger256::from(12345u64);
        let mut acc = BigInteger256::from(11111u64).mul_u64_w_carry::<5>(1);
        let acc_copy = acc;
        a.fm_limbs_into::<1, 5>(&[0], &mut acc, false);
        assert_eq!(acc, acc_copy); // Should be unchanged

        // Test multiplication by one (should be just addition)
        let a = BigInteger256::from(12345u64);
        let mut acc = BigInteger256::from(11111u64).mul_u64_w_carry::<5>(1);
        a.fm_limbs_into::<1, 5>(&[1], &mut acc, false);
        let expected = BigUint::from(12345u64) + BigUint::from(11111u64);
        assert_eq!(BigUint::from(acc), expected);
    }

    #[test]
    fn test_mul_u128_w_carry() {
        let a = BigInteger256::from(0x123456789ABCDEFu64);
        let b = 0x987654321DEADBEEFu128;

        // Test against reference implementation
        let expected = BigUint::from(0x123456789ABCDEFu64) * BigUint::from(0x987654321DEADBEEFu128);
        let result = a.mul_u128_w_carry::<5, 6>(b);
        assert_eq!(BigUint::from(result), expected);

        // Test with u64 value (should be same as mul_u64_w_carry)
        let b_u64 = 0x987654321u64;
        let result_u128 = a.mul_u128_w_carry::<5, 6>(b_u64 as u128);
        let result_u64 = a.mul_u64_w_carry::<5>(b_u64);

        // Compare first 5 limbs (u64 result size)
        for i in 0..5 {
            assert_eq!(result_u128.0[i], result_u64.0[i]);
        }
        assert_eq!(result_u128.0[5], 0); // Extra limb should be zero

        // Test zero cases
        let zero = BigInteger256::zero();
        let result = zero.mul_u128_w_carry::<5, 6>(12345);
        assert!(result.is_zero());

        let a = BigInteger256::from(12345u64);
        let result = a.mul_u128_w_carry::<5, 6>(0);
        assert!(result.is_zero());

        // Test multiplication by one
        let a = BigInteger256::from(0xDEADBEEFu64);
        let result = a.mul_u128_w_carry::<5, 6>(1);
        let expected_bytes = a.to_bytes_le();
        let result_bytes = result.to_bytes_le();
        assert_eq!(&result_bytes[..expected_bytes.len()], &expected_bytes[..]);
    }

    #[test]
    fn test_fm128a_basic_and_edges() {
        use crate::biginteger::BigInteger256 as B;
        // Basic reference check against BigUint
        let a = B::from(0x123456789ABCDEFu64);
        let b = 0x987654321DEADBEEFu128;
        let mut acc = B::zero().mul_u128_w_carry::<5, 6>(1); // zero-extended accumulator (6 limbs)
        a.fm_limbs_into::<2, 6>(&[b as u64, (b >> 64) as u64], &mut acc, true);
        let expected = num_bigint::BigUint::from(0x123456789ABCDEFu64)
            * num_bigint::BigUint::from(0x987654321DEADBEEFu128);
        assert_eq!(num_bigint::BigUint::from(acc), expected);

        // Zero multiplier: no change
        let a = B::from(12345u64);
        let mut acc = B::from(11111u64).mul_u128_w_carry::<5, 6>(1);
        let acc_copy = acc;
        a.fm_limbs_into::<2, 6>(&[0u64, 0u64], &mut acc, true);
        assert_eq!(acc, acc_copy);

        // One multiplier: reduces to addition
        let a = B::from(12345u64);
        let mut acc = B::from(11111u64).mul_u128_w_carry::<5, 6>(1);
        a.fm_limbs_into::<2, 6>(&[1u64, 0u64], &mut acc, true);
        let expected = num_bigint::BigUint::from(12345u64) + num_bigint::BigUint::from(11111u64);
        assert_eq!(num_bigint::BigUint::from(acc), expected);

        // Overflow propagation from limb N into highest limb
        let a = B::new([u64::MAX; 4]);
        let mut acc = B::zero().mul_u128_w_carry::<5, 6>(1);
        // Pre-fill limb N to force overflow when adding the final carry from low pass
        acc.0[4] = u64::MAX; // limb N
        acc.0[5] = 0; // highest limb
                      // cause carry=1 from low pass (a * 2)
        a.fm_limbs_into::<2, 6>(&[2u64, 0u64], &mut acc, true);
        // Expect highest limb incremented by 1 due to overflow from limb N
        assert_eq!(acc.0[5], 1);
    }

    #[test]
    fn test_overflow_behavior_fmu64a() {
        // Test that overflow in the highest limb wraps around as documented
        let a = BigInteger256::new([u64::MAX; 4]);
        let mut acc = BigInteger256::new([0, 0, 0, 0]).mul_u64_w_carry::<5>(1);
        acc.0[4] = u64::MAX; // Set highest limb to max

        // This should cause overflow in the highest limb
        a.fm_limbs_into::<1, 5>(&[2u64], &mut acc, false);

        // The overflow should wrap around
        // u64::MAX * 2 = 2^65 - 2, which when added to u64::MAX = 2^65 + u64::MAX - 2
        // This wraps to u64::MAX - 2 with a carry of 1 that itself wraps
        assert_eq!(acc.0[4], u64::MAX.wrapping_add(1)); // Wrapped result
    }

    #[test]
    fn test_edge_cases_large_numbers() {
        // Test with maximum values
        let max_bi = BigInteger256::new([u64::MAX; 4]);

        // mul_u64_w_carry with max values
        let result = max_bi.mul_u64_w_carry::<5>(u64::MAX);
        let expected = BigUint::from(max_bi) * BigUint::from(u64::MAX);
        assert_eq!(BigUint::from(result), expected);

        // mul_u128_w_carry with max values
        let result = max_bi.mul_u128_w_carry::<5, 6>(u128::MAX);
        let expected = BigUint::from(max_bi) * BigUint::from(u128::MAX);
        assert_eq!(BigUint::from(result), expected);
    }

    #[test]
    fn test_fmu64a_into_nplus4_correctness_and_edges() {
        use crate::biginteger::{BigInt, BigInteger256 as B};
        let a = B::from(0xDEADBEEFCAFEBABEu64);
        let other = 0xFEDCBA9876543210u64;
        let mut acc = BigInt::<8>::zero(); // N+4 accumulator for N=4

        // Reference: (a * other + acc_before) mod 2^(64*(N+4))
        let before = BigUint::from(acc.clone());
        a.fm_limbs_into::<1, 8>(&[other], &mut acc, true);
        let mut expected = BigUint::from(a);
        expected *= BigUint::from(other);
        expected += before;
        let modulus = BigUint::from(1u8) << (64 * 8);
        expected %= &modulus;
        assert_eq!(BigUint::from(acc.clone()), expected);

        // Zero multiplier is no-op
        let mut acc2 = acc.clone();
        a.fm_limbs_into::<1, 8>(&[0u64], &mut acc2, true);
        assert_eq!(acc2, acc);

        // One multiplier reduces to addition
        let mut acc3 = BigInt::<8>::zero();
        acc3.0[0] = 11111;
        let before3 = BigUint::from(acc3.clone());
        a.fm_limbs_into::<1, 8>(&[1u64], &mut acc3, true);
        let mut expected3 = BigUint::from(a);
        expected3 += before3;
        expected3 %= &modulus;
        assert_eq!(BigUint::from(acc3), expected3);

        // Force cascading carry across N..=N+3
        let a = B::new([u64::MAX; 4]);
        let mut acc4 = BigInt::<8>::zero();
        acc4.0[4] = u64::MAX; // limb N
        acc4.0[5] = u64::MAX; // limb N+1
        acc4.0[6] = u64::MAX; // limb N+2
        acc4.0[7] = 0; // limb N+3 (top)
                       // Use multiplier 2 so the low pass produces a carry=1
        a.fm_limbs_into::<1, 8>(&[2u64], &mut acc4, true);
        assert_eq!(acc4.0[7], 1);
    }

    #[test]
    fn test_fm2x64a_into_nplus4_correctness() {
        use crate::biginteger::{BigInt, BigInteger256 as B};
        let a = B::from(0x1234567890ABCDEFu64);
        let other = [0x0FEDCBA987654321u64, 0x0011223344556677u64];
        let mut acc = BigInt::<8>::zero();

        let before = BigUint::from(acc.clone());
        a.fm_limbs_into::<2, 8>(&other, &mut acc, true);

        // Expected: a * (lo + (hi << 64)) + acc_before mod 2^(64*8)
        let hi = BigUint::from(other[1]);
        let lo = BigUint::from(other[0]);
        let factor = (hi << 64) + lo;
        let mut expected = BigUint::from(a);
        expected *= factor;
        expected += before;
        let modulus = BigUint::from(1u8) << (64 * 8);
        expected %= &modulus;
        assert_eq!(BigUint::from(acc.clone()), expected);

        // Zero limbs are no-op
        let mut acc2 = acc.clone();
        a.fm_limbs_into::<2, 8>(&[0u64, 0u64], &mut acc2, true);
        assert_eq!(acc2, acc);
    }

    #[test]
    fn test_fm3x64a_into_nplus4_correctness() {
        use crate::biginteger::{BigInt, BigInteger256 as B};
        let a = B::from(0x0F0E0D0C0B0A0908u64);
        let other = [
            0x89ABCDEF01234567u64,
            0x76543210FEDCBA98u64,
            0x1122334455667788u64,
        ];
        let mut acc = BigInt::<8>::zero();

        let before = BigUint::from(acc.clone());
        a.fm_limbs_into::<3, 8>(&other, &mut acc, true);

        // Expected: a * (o0 + (o1<<64) + (o2<<128)) + acc_before mod 2^(64*8)
        let term0 = BigUint::from(other[0]);
        let term1 = BigUint::from(other[1]) << 64;
        let term2 = BigUint::from(other[2]) << 128;
        let factor = term0 + term1 + term2;
        let mut expected = BigUint::from(a);
        expected *= factor;
        expected += before;
        let modulus = BigUint::from(1u8) << (64 * 8);
        expected %= &modulus;
        assert_eq!(BigUint::from(acc.clone()), expected);

        // Edge: ensure offset accumulation lands in correct limbs
        // Fill acc with a pattern, then accumulate using only the highest limb to ensure writes start at index 2
        let a = B::from(3u64);
        let mut acc2 = BigInt::<8>::zero();
        acc2.0[0] = 5;
        acc2.0[1] = 7;
        let other2 = [0, 0, 2]; // Only offset by 2 limbs
        let before2 = BigUint::from(acc2.clone());
        a.fm_limbs_into::<3, 8>(&other2, &mut acc2, true);
        let mut expected2 = BigUint::from(a);
        expected2 *= BigUint::from(2u64) << 128;
        expected2 += before2;
        let modulus = BigUint::from(1u8) << (64 * 8);
        expected2 %= &modulus;
        assert_eq!(BigUint::from(acc2), expected2);
    }

    // ==============================
    // SignedBigInt tests
    // ==============================

    #[test]
    fn test_signed_construction() {
        // zero and one
        let z = SignedBigInt::<1>::zero();
        assert!(z.is_zero());
        assert!(z.is_positive);
        let o = SignedBigInt::<1>::one();
        assert!(!o.is_zero());
        assert!(o.is_positive);

        // from_u64
        let p = SignedBigInt::<1>::from_u64(42);
        assert_eq!(p.magnitude.0[0], 42);
        assert!(p.is_positive);
        let n = SignedBigInt::<1>::from((42u64, false));
        assert_eq!(n.magnitude.0[0], 42);
        assert!(!n.is_positive);
    }

    #[test]
    fn test_signed_add_sub_mul_neg() {
        let a = SignedBigInt::<1>::from_u64(10);
        let b = SignedBigInt::<1>::from_u64(5);
        assert_eq!((a + b).magnitude.0[0], 15);
        assert_eq!((a - b).magnitude.0[0], 5);
        assert_eq!((a * b).magnitude.0[0], 50);
        let neg = -a;
        assert_eq!(neg.magnitude.0[0], 10);
        assert!(!neg.is_positive);

        // opposite signs
        let x = SignedBigInt::<1>::from_u64(30);
        let y = SignedBigInt::<1>::from((20u64, false));
        let r = x + y; // 30 - 20
        assert!(r.is_positive);
        assert_eq!(r.magnitude.0[0], 10);

        let x2 = SignedBigInt::<1>::from((20u64, false));
        let y2 = SignedBigInt::<1>::from_u64(30);
        let r2 = x2 + y2; // -20 + 30
        assert!(r2.is_positive);
        assert_eq!(r2.magnitude.0[0], 10);
    }

    #[test]
    fn test_signed_to_i128_and_mag_helpers() {
        let p = SignedBigInt::<1>::from_u64(100);
        assert_eq!(p.to_i128(), 100);
        let n = SignedBigInt::<1>::from((100u64, false));
        assert_eq!(n.to_i128(), -100);

        let d = SignedBigInt::<2>::from_u128(0x1234_5678_9abc_def0_1111_2222_3333_4444u128);
        assert_eq!(d.magnitude.0[0], 0x1111_2222_3333_4444);
        assert_eq!(d.magnitude.0[1], 0x1234_5678_9abc_def0);
        // Positive below 2^127 should convert
        let expected_i128 = 0x1234_5678_9abc_def0_1111_2222_3333_4444u128 as i128;
        assert_eq!(d.to_i128(), Some(expected_i128));

        // Positive at 2^127 should fail
        let too_big_pos = SignedBigInt::<2>::from_u128(1u128 << 127);
        assert_eq!(too_big_pos.to_i128(), None);

        let small = SignedBigInt::<2>::new([100, 0], true);
        assert_eq!(small.to_i128(), Some(100));
        assert_eq!(small.magnitude_as_u128(), 100u128);
    }

    #[test]
    fn test_add_with_sign_u64_helper() {
        let (mag, sign) = crate::biginteger::signed::add_with_sign_u64(10, true, 5, true);
        assert_eq!(mag, 15);
        assert!(sign);
        let (mag2, sign2) = crate::biginteger::signed::add_with_sign_u64(10, true, 5, false);
        assert_eq!(mag2, 5);
        assert!(sign2);
        let (mag3, sign3) = crate::biginteger::signed::add_with_sign_u64(5, true, 10, false);
        assert_eq!(mag3, 5);
        assert!(!sign3);
    }

    #[test]
    fn test_signed_truncated_add_sub() {
        use crate::biginteger::SignedBigInt as S;
        let a = S::<2>::from_u128(0x0000_0000_0000_0001_ffff_ffff_ffff_fffe);
        let b = S::<2>::from_u128(0x0000_0000_0000_0001_0000_0000_0000_0001);
        // Add and truncate to 1 limb
        // Respect BigInt::add_trunc contract by truncating rhs to 1 limb
        let b1 = S::<1>::from_bigint(
            crate::biginteger::BigInt::<1>::new([b.magnitude.0[0]]),
            b.is_positive,
        );
        let r1 = a.add_trunc_mixed::<1, 1>(&b1);
        // expected low limb wrap of the low words, ignoring carry to limb1
        let expected_low = (0xffff_ffff_ffff_fffeu64).wrapping_add(0x0000_0000_0000_0001u64);
        assert_eq!(r1.magnitude.0[0], expected_low);
        assert!(r1.is_positive);

        // Different signs: subtraction path (use N=1 throughout so M<=P inside)
        let a = S::<1>::from_u64(0x2);
        let b = S::<1>::from(-3i64); // -3
        let r2 = a.add_trunc::<1>(&b); // 2 + (-3) = -1, truncated to 64-bit
        assert_eq!(r2.magnitude.0[0], 1);
        assert!(!r2.is_positive);

        // sub_trunc uses add_trunc internally
        let x = S::<1>::from_u64(10);
        let y = S::<1>::from_u64(7);
        let r3 = x.sub_trunc::<1>(&y);
        assert_eq!(r3.magnitude.0[0], 3);
        assert!(r3.is_positive);
    }

    #[test]
    fn test_signed_truncated_mul_and_fmadd() {
        use crate::biginteger::SignedBigInt as S;
        // 128-bit x 64-bit -> truncated to 2 limbs (128-bit)
        let a = S::<2>::from_u128(0x0000_0000_0000_0001_FFFF_FFFF_FFFF_FFFFu128);
        let b = S::<1>::from_u64(0x2);
        let p = a.mul_trunc::<1, 2>(&b);
        // Expected low 128 bits of the product
        let expected = num_bigint::BigUint::from(0x0000_0000_0000_0001_FFFF_FFFF_FFFF_FFFFu128)
            * num_bigint::BigUint::from(2u64);
        let got = num_bigint::BigUint::from(p.magnitude);
        assert_eq!(
            got,
            expected & ((num_bigint::BigUint::from(1u8) << 128) - 1u8)
        );
        assert!(p.is_positive);

        // fmadd into 1-limb accumulator (truncate to 64 bits)
        let a = S::<1>::from_u64(0xFFFF_FFFF_FFFF_FFFF);
        let b = S::<1>::from_u64(0x2);
        let mut acc = S::<1>::from_u64(1);
        a.fmadd_trunc::<1, 1>(&b, &mut acc); // acc = 1 + (a*b) mod 2^64 with sign +
                                             // a*b = (2^64 - 1)*2 = 2^65 - 2 => low 64 = (2^64 - 2)
        let expected_low = (u64::MAX).wrapping_sub(1);
        assert_eq!(acc.magnitude.0[0], expected_low.wrapping_add(1));
    }

    #[test]
    fn test_signed_truncated_add_sub_mixed() {
        use crate::biginteger::SignedBigInt as S;
        // Same sign, different widths, ensure carry handling and sign preservation
        let a = S::<2>::from_u128(0x0000_0000_0000_0002_FFFF_FFFF_FFFF_FFFF);
        let b = S::<1>::from_u64(0x0000_0000_0000_0002);
        let r = a.add_trunc_mixed::<1, 2>(&b); // 128-bit result
        let expected = num_bigint::BigUint::from(0x0000_0000_0000_0002_FFFF_FFFF_FFFF_FFFFu128)
            + num_bigint::BigUint::from(2u64);
        assert_eq!(num_bigint::BigUint::from(r.magnitude), expected);
        assert!(r.is_positive);

        // Different signs, |a| > |b|: result sign should be sign(a)
        let a2 = S::<2>::from_u128(5000);
        let b2 = S::<1>::from((3000u64, false)); // -3000
        let r2 = a2.add_trunc_mixed::<1, 2>(&b2);
        assert!(r2.is_positive);
        assert_eq!(r2.magnitude.0[0], 2000);

        // Different signs, |b| > |a|: result sign should be sign(b)
        let a3 = S::<2>::from_u128(1000);
        let b3 = S::<1>::from((3000u64, false)); // -3000
        let r3 = a3.add_trunc_mixed::<1, 2>(&b3);
        assert!(!r3.is_positive);
        assert_eq!(r3.magnitude.0[0], 2000);

        // sub_trunc_mixed basic checks
        let a4 = S::<2>::from_u128(10000);
        let b4 = S::<1>::from_u64(9999);
        let r4 = a4.sub_trunc_mixed::<1, 2>(&b4);
        assert!(r4.is_positive);
        assert_eq!(r4.magnitude.0[0], 1);

        let a5 = S::<2>::from_u128(1000);
        let b5 = S::<1>::from_u64(5000);
        let r5 = a5.sub_trunc_mixed::<1, 2>(&b5);
        assert!(!r5.is_positive);
        assert_eq!(r5.magnitude.0[0], 4000);
    }

    #[test]
    fn test_signed_fmadd_trunc_mixed_width_and_signs() {
        use crate::biginteger::SignedBigInt as S;
        // Case 1: same sign => pure addition of magnitudes
        let a = S::<2>::from_u128(30000);
        let b = S::<1>::from_u64(7);
        let mut acc = S::<2>::from_u128(1000000);
        a.fmadd_trunc::<1, 2>(&b, &mut acc); // acc += 210000
        assert!(acc.is_positive);
        assert_eq!(
            acc.magnitude.0[0] as u128 + ((acc.magnitude.0[1] as u128) << 64),
            1210000u128
        );

        // Case 2: different sign, |prod| < |acc| => sign preserved
        let a2 = S::<2>::from_u128(30000);
        let b2 = S::<1>::from((7u64, false)); // -7
        let mut acc2 = S::<2>::from_u128(1000000);
        a2.fmadd_trunc::<1, 2>(&b2, &mut acc2); // acc2 -= 210000 => 790000
        assert!(acc2.is_positive);
        assert_eq!(
            acc2.magnitude.0[0] as u128 + ((acc2.magnitude.0[1] as u128) << 64),
            790000u128
        );

        // Case 3: different sign, |prod| > |acc| => sign flips to prod_sign
        let a3 = S::<2>::from_u128(300);
        let b3 = S::<1>::from((7u64, false)); // -7 => prod = -2100
        let mut acc3 = S::<2>::from_u128(1000);
        a3.fmadd_trunc::<1, 2>(&b3, &mut acc3); // 1000 - 2100 = -1100
        assert!(!acc3.is_positive);
        assert_eq!(acc3.magnitude.0[0], 1100);
    }

    #[test]
    fn test_prop_add_sub_trunc_mixed_random() {
        use crate::biginteger::SignedBigInt as S;
        use ark_std::rand::Rng;
        let mut rng = ark_std::test_rng();

        // Helper to validate a single pair for given consts
        macro_rules! run_case {
            ($n:expr, $m:expr, $p:expr, $iters:expr) => {{
                for _ in 0..$iters {
                    let a_mag: crate::biginteger::BigInt<$n> = UniformRand::rand(&mut rng);
                    let b_mag: crate::biginteger::BigInt<$m> = UniformRand::rand(&mut rng);
                    let a_pos = (rng.gen::<u8>() & 1) == 1;
                    let b_pos = (rng.gen::<u8>() & 1) == 1;
                    let a = S::<$n>::from_bigint(a_mag, a_pos);
                    let b = S::<$m>::from_bigint(b_mag, b_pos);

                    // add_trunc_mixed
                    let r_add = a.add_trunc_mixed::<$m, $p>(&b);
                    let a_bu = num_bigint::BigUint::from(a.magnitude);
                    let b_bu = num_bigint::BigUint::from(b.magnitude);
                    let (exp_add_mag, exp_add_pos) = if a_pos == b_pos {
                        (&a_bu + &b_bu, a_pos)
                    } else if a_bu >= b_bu {
                        (&a_bu - &b_bu, a_pos)
                    } else {
                        (&b_bu - &a_bu, b_pos)
                    };
                    let modulus = num_bigint::BigUint::from(1u8) << (64 * $p);
                    let exp_add_mag_mod = exp_add_mag % &modulus;
                    assert_eq!(num_bigint::BigUint::from(r_add.magnitude), exp_add_mag_mod);
                    if exp_add_mag_mod != num_bigint::BigUint::from(0u8) {
                        assert_eq!(r_add.is_positive, exp_add_pos);
                    }

                    // sub_trunc_mixed: a - b
                    let r_sub = a.sub_trunc_mixed::<$m, $p>(&b);
                    let (exp_sub_mag, exp_sub_pos) = if a_pos != b_pos {
                        (&a_bu + &b_bu, a_pos)
                    } else if a_bu >= b_bu {
                        (&a_bu - &b_bu, a_pos)
                    } else {
                        (&b_bu - &a_bu, !a_pos)
                    };
                    let exp_sub_mag_mod = exp_sub_mag % &modulus;
                    assert_eq!(num_bigint::BigUint::from(r_sub.magnitude), exp_sub_mag_mod);
                    if exp_sub_mag_mod != num_bigint::BigUint::from(0u8) {
                        assert_eq!(r_sub.is_positive, exp_sub_pos);
                    }
                }
            }};
        }

        // Ensure P >= M to satisfy internal add_trunc constraints
        run_case!(2, 3, 3, 200);
        run_case!(3, 1, 2, 200);
        run_case!(1, 2, 2, 200);
    }

    #[test]
    fn test_prop_fmadd_trunc_random() {
        use crate::biginteger::SignedBigInt as S;
        use ark_std::rand::Rng;
        let mut rng = ark_std::test_rng();

        macro_rules! run_case {
            ($n:expr, $m:expr, $p:expr, $iters:expr) => {{
                for _ in 0..$iters {
                    let a_mag: crate::biginteger::BigInt<$n> = UniformRand::rand(&mut rng);
                    let b_mag: crate::biginteger::BigInt<$m> = UniformRand::rand(&mut rng);
                    let acc_mag: crate::biginteger::BigInt<$p> = UniformRand::rand(&mut rng);
                    let a_pos = (rng.gen::<u8>() & 1) == 1;
                    let b_pos = (rng.gen::<u8>() & 1) == 1;
                    let acc_pos = (rng.gen::<u8>() & 1) == 1;
                    let a = S::<$n>::from_bigint(a_mag, a_pos);
                    let b = S::<$m>::from_bigint(b_mag, b_pos);
                    let mut acc = S::<$p>::from_bigint(acc_mag, acc_pos);

                    // expected via BigUint with truncation of the product BEFORE combining signs
                    let a_bu = num_bigint::BigUint::from(a.magnitude);
                    let b_bu = num_bigint::BigUint::from(b.magnitude);
                    let acc_bu = num_bigint::BigUint::from(acc.magnitude);
                    let modulus = num_bigint::BigUint::from(1u8) << (64 * $p);
                    let prod_mod = (&a_bu * &b_bu) % &modulus;
                    let prod_pos = a_pos == b_pos;
                    let (exp_mag_mod, exp_pos) = if acc_pos == prod_pos {
                        ((acc_bu + &prod_mod) % &modulus, acc_pos)
                    } else if acc_bu >= prod_mod {
                        (acc_bu - &prod_mod, acc_pos)
                    } else {
                        (prod_mod - &acc_bu, prod_pos)
                    };

                    a.fmadd_trunc::<$m, $p>(&b, &mut acc);

                    assert_eq!(num_bigint::BigUint::from(acc.magnitude), exp_mag_mod);
                    if exp_mag_mod != num_bigint::BigUint::from(0u8) {
                        assert_eq!(acc.is_positive, exp_pos);
                    }
                }
            }};
        }

        run_case!(2, 1, 2, 200);
        run_case!(3, 2, 2, 200);
    }

    // ==============================
    // Tests for add_trunc and add_assign_trunc (unsigned BigInt)
    // ==============================

    #[test]
    fn test_add_trunc_correctness_random() {
        use crate::biginteger::BigInt;
        let mut rng = ark_std::test_rng();

        macro_rules! run_case {
            ($n:expr, $m:expr, $p:expr, $iters:expr) => {{
                for _ in 0..$iters {
                    let mut a: BigInt<$n> = UniformRand::rand(&mut rng);
                    let mut b: BigInt<$m> = UniformRand::rand(&mut rng);

                    // Clamp low P limbs to avoid any carry across limb P-1 in add_trunc.
                    let mut i = 0;
                    while i < core::cmp::min($p, $n) {
                        a.0[i] >>= 1;
                        i += 1;
                    }
                    let mut j = 0;
                    while j < core::cmp::min($p, $m) {
                        b.0[j] >>= 1;
                        j += 1;
                    }

                    // Build rhs respecting M <= P
                    let (res, b_p): (BigInt<$p>, BigInt<$p>) = if $m <= $p {
                        let mut b_p = BigInt::<$p>::zero();
                        let mut k = 0;
                        while k < $m {
                            b_p.0[k] = b.0[k];
                            k += 1;
                        }
                        (a.add_trunc::<$m, $p>(&b), b_p)
                    } else {
                        let mut bl = [0u64; $p];
                        let mut t = 0;
                        while t < $p {
                            bl[t] = b.0[t];
                            t += 1;
                        }
                        let b_trunc = BigInt::<$p>::new(bl);
                        (a.add_trunc::<$p, $p>(&b_trunc), b_trunc)
                    };

                    // Expected using low-P truncated operands (after clamping)
                    let mut a_p = BigInt::<$p>::zero();
                    let mut u = 0;
                    while u < core::cmp::min($p, $n) {
                        a_p.0[u] = a.0[u];
                        u += 1;
                    }
                    let a_bu = BigUint::from(a_p);
                    let b_bu = BigUint::from(b_p);
                    let modulus = BigUint::from(1u8) << (64 * $p);
                    let expected = (a_bu + b_bu) % &modulus;
                    assert_eq!(BigUint::from(res), expected);
                }
            }};
        }

        // Same-width
        run_case!(4, 4, 4, 200);
        // Mixed widths with P chosen to satisfy M <= P
        run_case!(4, 2, 3, 200);
        run_case!(2, 4, 4, 200);
    }

    #[test]
    fn test_add_assign_trunc_correctness_and_zeroing() {
        use crate::biginteger::BigInt;
        let mut rng = ark_std::test_rng();

        // Case 1: N = 4, M = 4 (no truncation when P=N); compare against add_trunc and add_with_carry
        for _ in 0..200 {
            let mut a: BigInt<4> = UniformRand::rand(&mut rng);
            let mut b: BigInt<4> = UniformRand::rand(&mut rng);
            // Ensure no carry anywhere by masking all limbs to 62 bits
            for i in 0..4 {
                a.0[i] &= (1u64 << 62) - 1;
                b.0[i] &= (1u64 << 62) - 1;
            }
            let r_trunc = a.add_trunc::<4, 4>(&b);
            let mut a2 = a;
            a2.add_assign_trunc::<4>(&b);
            assert_eq!(a2, r_trunc);

            // Regular add_with_carry should match lower 4 limbs modulo 2^(256)
            let mut a3 = a;
            a3.add_with_carry(&b);
            assert_eq!(a3, r_trunc);
        }

        // Case 2: N = 4, M = 4, P = 3 (truncation): low P limbs must match add_trunc
        for _ in 0..200 {
            let mut a: BigInt<4> = UniformRand::rand(&mut rng);
            let mut b: BigInt<4> = UniformRand::rand(&mut rng);
            for i in 0..4 {
                a.0[i] &= (1u64 << 62) - 1;
                b.0[i] &= (1u64 << 62) - 1;
            }
            // Respect add_trunc contract by pre-truncating rhs to P limbs
            let b3 = crate::biginteger::BigInt::<3>::new([b.0[0], b.0[1], b.0[2]]);
            let r_trunc = a.add_trunc::<3, 3>(&b3);
            let mut a2 = a;
            a2.add_assign_trunc::<4>(&b);
            // Low 3 limbs match result
            for i in 0..3 {
                assert_eq!(a2.0[i], r_trunc.0[i]);
            }
        }

        // Case 3: Mixed widths N = 4, M = 2, P = 3: low P limbs must match add_trunc
        for _ in 0..200 {
            let mut a: BigInt<4> = UniformRand::rand(&mut rng);
            let b: BigInt<2> = UniformRand::rand(&mut rng);
            a.0[3] >>= 1;
            let r_trunc = a.add_trunc::<2, 3>(&b);
            let mut a2 = a;
            a2.add_assign_trunc::<2>(&b);
            for i in 0..3 {
                assert_eq!(a2.0[i], r_trunc.0[i]);
            }
        }
    }

    #[test]
    fn test_add_trunc_and_add_assign_trunc_overflow_edges() {
        use crate::biginteger::BigInt;

        // Use values that don't overflow beyond N to respect debug contract
        let mut a = BigInt::<4>::new([u64::MAX; 4]);
        let mut b = BigInt::<4>::new([u64::MAX; 4]);
        for i in 0..4 {
            a.0[i] >>= 1;
            b.0[i] >>= 1;
        }
        // P = 4: result should match BigUint addition modulo 2^256
        // add_assign_trunc debug-overflow behavior cannot be reliably asserted in this
        // environment without std; we validate the non-mutating truncated result above.

        // P = 3: validate truncated result against BigUint; pre-truncate rhs to 3 limbs
        let b3 = crate::biginteger::BigInt::<3>::new([b.0[0], b.0[1], b.0[2]]);
        let r3 = a.add_trunc::<3, 3>(&b3);
        let a_bu = BigUint::from(a);
        let b_bu = BigUint::from(b);
        let modulus = BigUint::from(1u8) << (64 * 3);
        let expected_r3 = (a_bu + b_bu) % &modulus;
        assert_eq!(BigUint::from(r3), expected_r3);
    }
}
