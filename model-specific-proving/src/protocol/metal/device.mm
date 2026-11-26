#import <Metal/Metal.h>
#include <algorithm>
#include <memory>

struct Device {
    id<MTLDevice> gpu;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> rounds[3], tree, binary;
};

struct Job {
    Device *device;
    id<MTLBuffer> input, tiles, table, axes, read, write, partial;
    unsigned round = 0, reads = 1;
    id<MTLComputePipelineState> pipeline;
};

static id<MTLBuffer> upload(id<MTLDevice> gpu, const void *bytes, size_t len) {
    return [gpu newBufferWithBytes:bytes length:len options:MTLResourceStorageModeShared];
}

extern "C" void *atlas_metal_device(const void *library, size_t len) {
    @autoreleasepool {
        auto gpu = MTLCreateSystemDefaultDevice();
        if (!gpu || !gpu.hasUnifiedMemory) return nullptr;
        auto data = dispatch_data_create(library, len, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        auto lib = [gpu newLibraryWithData:data error:nullptr];
        if (!lib) return nullptr;
        auto queue = [gpu newCommandQueue];
        if (!queue) return nullptr;
        auto device = std::make_unique<Device>();
        device->gpu = gpu;
        device->queue = queue;
        for (unsigned kind=0;kind<3;kind++) {
            auto constants = [MTLFunctionConstantValues new];
            [constants setConstantValue:&kind type:MTLDataTypeUInt atIndex:0];
            auto kernel = [lib newFunctionWithName:@"round_sum" constantValues:constants error:nullptr];
            if (!kernel) return nullptr;
            auto pipeline = [gpu newComputePipelineStateWithFunction:kernel error:nullptr];
            if (!pipeline || pipeline.threadExecutionWidth != 32) return nullptr;
            device->rounds[kind] = pipeline;
        }
        auto tree = [lib newFunctionWithName:@"bitz_tree"];
        if (tree) device->tree = [gpu newComputePipelineStateWithFunction:tree error:nullptr];
        auto binary = [lib newFunctionWithName:@"bitz_round"];
        if (binary) device->binary = [gpu newComputePipelineStateWithFunction:binary error:nullptr];
        return device.release();
    }
}

extern "C" void *atlas_metal_job(Device *d, size_t cells, unsigned columns, unsigned reads, unsigned kind,
                                 const void *input, size_t input_bytes,
                                 const void *tiles, size_t tile_bytes,
                                 const void *table, size_t table_bytes, size_t axis_bytes) {
    @autoreleasepool {
        auto gpu = d->gpu;
        auto job = std::make_unique<Job>();
        job->device = d;
        job->reads = reads;
        job->pipeline = d->rounds[kind];
        columns *= reads;
        job->input = upload(gpu, input, input_bytes);
        job->tiles = upload(gpu, tiles, tile_bytes);
        job->table = upload(gpu, table, table_bytes);
        auto buffer = [&](size_t bytes) {
            return [gpu newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        };
        job->axes = buffer(axis_bytes);
        job->read = buffer(cells / 4 * columns * 16);
        job->write = buffer(cells / 2 * columns * 16);
        job->partial = buffer((cells / 2 + 63) / 64 * 10 * 16 * reads);
        if (!job->input || !job->tiles || !job->table || !job->axes ||
            !job->read || !job->write || !job->partial) return nullptr;
        return job.release();
    }
}

extern "C" bool atlas_metal_round(Job *job, const void *args, size_t args_len,
                                 const void *axis, size_t axis_len, unsigned count,
                                 unsigned threads, void *partial, size_t partial_len) {
    @autoreleasepool {
        memcpy(job->axes.contents, axis, axis_len);
        auto command = [job->device->queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:job->pipeline];
        [encoder setBuffer:job->round < 2 ? job->input : job->read offset:0 atIndex:0];
        [encoder setBuffer:job->write offset:0 atIndex:1];
        [encoder setBuffer:job->partial offset:0 atIndex:2];
        [encoder setBytes:args length:args_len atIndex:3];
        [encoder setBuffer:job->tiles offset:0 atIndex:4];
        [encoder setBuffer:job->axes offset:0 atIndex:5];
        [encoder setBuffer:job->table offset:0 atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake((count + threads - 1) / threads, job->reads, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return false;
        memcpy(partial, job->partial.contents, partial_len);
        if (job->round++ != 0) std::swap(job->read, job->write);
        return true;
    }
}

extern "C" void atlas_metal_snapshot(Job *job, void *data, size_t len) {
    memcpy(data, job->read.contents, len);
}
extern "C" void atlas_metal_drop(Job *job) { delete job; }

extern "C" bool atlas_metal_tree(Device *d, const void *values, unsigned count,
                                unsigned groups, bool include_output, void *upper) {
    @autoreleasepool {
        if (!d->tree) return false;
        auto initial = upload(d->gpu, values, size_t(count) * 16);
        auto tree = [d->gpu newBufferWithLength:size_t(count) * 16 options:MTLResourceStorageModeShared];
        if (!initial || !tree) return false;
        auto command = [d->queue commandBuffer];
        id<MTLBuffer> read = initial;
        size_t from = 0, to = 0;
        for (unsigned n = count / 2; n > groups || (include_output && n == groups); n /= 2) {
            auto encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:d->tree];
            [encoder setBuffer:read offset:from * 16 atIndex:0];
            [encoder setBuffer:tree offset:to * 16 atIndex:1];
            [encoder setBytes:&n length:sizeof(n) atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake((n + 127) / 128, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
            read = tree;
            from = to;
            to += n;
        }
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return false;
        memcpy(upper, tree.contents, to * 16);
        return true;
    }
}

struct BinaryJob {
    Device *device;
    id<MTLBuffer> read, write, low, high, partial;
};

extern "C" void *atlas_binary_start(Device *d, const void *left, const void *right, size_t bytes) {
    @autoreleasepool {
        if (!d->binary || d->binary.threadExecutionWidth != 32) return nullptr;
        auto buffer = [&](size_t len) {
            return [d->gpu newBufferWithLength:len options:MTLResourceStorageModeShared];
        };
        auto job = std::make_unique<BinaryJob>(BinaryJob{d,buffer(2*bytes),buffer(bytes),
            buffer(4096*16),buffer(std::max<size_t>(16,bytes/8192)),buffer((bytes/32+127)/128*32)});
        if(!job->read || !job->write || !job->low || !job->high || !job->partial) return nullptr;
        memcpy(job->read.contents,left,bytes);
        memcpy((char *)job->read.contents+bytes,right,bytes);
        return job.release();
    }
}

extern "C" bool atlas_binary_round(BinaryJob *job, const void *args, size_t args_bytes,
                                   unsigned count, bool binding, const void *low, size_t low_bytes,
                                   const void *high, size_t high_bytes, void *out) {
    @autoreleasepool {
        memcpy(job->low.contents,low,low_bytes);
        memcpy(job->high.contents,high,high_bytes);
        unsigned groups=(count+127)/128;
        auto command=[job->device->queue commandBuffer];
        auto encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:job->device->binary];
        [encoder setBuffer:job->read offset:0 atIndex:0];
        [encoder setBuffer:job->write offset:0 atIndex:1];
        [encoder setBuffer:job->partial offset:0 atIndex:2];
        [encoder setBytes:args length:args_bytes atIndex:3];
        [encoder setBuffer:job->low offset:0 atIndex:4];
        [encoder setBuffer:job->high offset:0 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(groups,1,1)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted) return false;
        const uint64_t *values=(const uint64_t *)job->partial.contents;
        auto *sum=(uint64_t *)out;
        memset(sum,0,32);
        for(unsigned j=0;j<groups;j++) for(unsigned k=0;k<4;k++) sum[k]^=values[4*j+k];
        if(binding) std::swap(job->read,job->write);
        return true;
    }
}
extern "C" void atlas_binary_snapshot(BinaryJob *job, void *left, void *right, size_t len) {
    memcpy(left,job->read.contents,len);
    memcpy(right,(char *)job->read.contents+len,len);
}
extern "C" void atlas_binary_drop(BinaryJob *job) {delete job;}
