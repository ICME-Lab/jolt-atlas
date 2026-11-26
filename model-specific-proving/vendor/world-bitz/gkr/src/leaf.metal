#include <metal_stdlib>
using namespace metal;

struct Args { uint trees; uint chunk; uint count; };

kernel void leaf_buckets(
    device const uint *packed [[buffer(0)]],
    device const uint4 *weights [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant Args &args [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint size [[threads_per_threadgroup]]
) {
    threadgroup atomic_uint bins[2 * 1024];
    uint shards = size / 32;
    for (uint i = lid; i < shards * 1024; i += size) {
        atomic_store_explicit(&bins[i], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint shard = lid / 32;
    uint start = block * args.chunk;
    for (uint i = lid; i < args.chunk; i += size) {
        uint at = start + i;
        uint nibble = (packed[at >> 3] >> (4 * (at & 7))) & 15;
        nibble |= ((packed[(at + args.count) >> 3] >> (4 * (at & 7))) & 15) << 4;
        if (nibble != 0) {
            uint4 value = weights[at & (args.trees - 1)];
            for (uint j = 0; j < 4; ++j) {
                atomic_fetch_xor_explicit(&bins[shard * 1024 + 4 * nibble + j], value[j], memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint lane = lid; lane < 1024; lane += size) {
        uint result = 0;
        for (uint s = 0; s < shards; ++s) {
            result ^= atomic_load_explicit(&bins[s * 1024 + lane], memory_order_relaxed);
        }
        output[block * 1024 + lane] = result;
    }
}

kernel void reduce_buckets(
    device const uint4 *partial [[buffer(0)]],
    device uint4 *output [[buffer(1)]],
    constant uint2 &shape [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    uint row = i / shape.y;
    uint bin = i % shape.y;
    uint4 sum = 0;
    for (uint b = 0; b < shape.x; ++b)
        sum ^= partial[shape.y * (row * shape.x + b) + bin];
    output[i] = sum;
}

kernel void leaf_moments(
    device const uint *packed [[buffer(0)]],
    device const uint4 *weights [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant Args &args [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup atomic_uint bins[2 * 81 * 4];
    for (uint i = lid; i < 648; i += 64)
        atomic_store_explicit(&bins[i], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint shard = (lid / 32) * 324;
    for (uint i = lid; i < args.chunk; i += 64) {
        uint at = block * args.chunk + i;
        uint left = 256, right = 256;
        for (uint stream = 0; stream < 4; ++stream) {
            uint pos = at + stream * args.count;
            uint nibble = (packed[pos >> 3] >> (4 * (pos & 7))) & 15;
            left |= (nibble & 3) << (2 * stream);
            right |= (nibble >> 2) << (2 * stream);
        }
        uint4 value = weights[at & (args.trees - 1)];
        for (uint l = left; l != 0; l &= l - 1) {
            uint a = ctz(l);
            for (uint r = a == 8 ? right & 255 : right; r != 0; r &= r - 1) {
                uint key = 4 * (9 * a + ctz(r));
                for (uint j = 0; j < 4; ++j)
                    atomic_fetch_xor_explicit(&bins[shard + key + j], value[j], memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < 324; i += 64)
        output[block * 324 + i] = atomic_load_explicit(&bins[i], memory_order_relaxed)
            ^ atomic_load_explicit(&bins[324 + i], memory_order_relaxed);
}
