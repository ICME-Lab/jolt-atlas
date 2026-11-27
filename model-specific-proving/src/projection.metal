#include <metal_stdlib>
using namespace metal;

kernel void split_digits(
    device const int *source [[buffer(0)]], device float *digits [[buffer(1)]],
    constant uint2 &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    int value = source[i];
    for (uint d = 0; d < p.y; ++d) {
        int digit = int((uint(value) + 64u) & 127u) - 64;
        digits[d * p.x + i] = float(digit);
        value = (value >> 7) + int((uint(value) & 127u) >= 64u);
    }
}

kernel void accumulate(
    device const float *partial [[buffer(0)]], device long *sum [[buffer(1)]],
    device const int *bias [[buffer(2)]], device int *output [[buffer(3)]],
    device uint *remainder [[buffer(4)]], device atomic_uint *failure [[buffer(5)]],
    constant uint4 &p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    uint flags = p.w;
    long term = long(partial[i]) * (1L << p.x);
    long raw = ((flags & 1) != 0) ? term : sum[i] + term;
    if ((flags & 2) == 0) {
        sum[i] = raw;
        return;
    }
    raw += long(bias[i % p.y]) * (1L << p.z);
    long rounded = raw >> p.z;
    if (rounded < -2147483648L || rounded > 2147483647L) {
        atomic_store_explicit(failure, 1u, memory_order_relaxed);
    }
    output[i] = int(rounded);
    remainder[i] = uint(raw) & ((1u << p.z) - 1u);
}
