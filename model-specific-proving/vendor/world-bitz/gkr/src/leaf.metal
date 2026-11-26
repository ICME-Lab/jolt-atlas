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
    constant uint &blocks [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    uint row = i >> 8;
    uint bin = i & 255;
    uint4 sum = 0;
    for (uint b = 0; b < blocks; ++b)
        sum ^= partial[256 * (row * blocks + b) + bin];
    output[i] = sum;
}
