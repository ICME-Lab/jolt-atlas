#import <Metal/Metal.h>

bool bitz_leaf_buckets(const void *library, size_t library_len, const void *packed,
                      const void *weights, uint32_t trees, uint32_t count, void *result, void *third) {
    static id<MTLDevice> device;
    static id<MTLCommandQueue> queue;
    static id<MTLComputePipelineState> buckets, reduce, moments;
    static dispatch_once_t once;
    @autoreleasepool {
        dispatch_once(&once, ^{
            device = MTLCreateSystemDefaultDevice();
            queue = [device newCommandQueue];
            dispatch_data_t data = dispatch_data_create(library, library_len, nil,
                                                        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            id<MTLLibrary> lib = [device newLibraryWithData:data error:NULL];
            id<MTLFunction> leaf = [lib newFunctionWithName:@"leaf_buckets"];
            id<MTLFunction> merge = [lib newFunctionWithName:@"reduce_buckets"];
            if (!queue || !leaf || !merge) return;
            buckets = [device newComputePipelineStateWithFunction:leaf error:NULL];
            reduce = [device newComputePipelineStateWithFunction:merge error:NULL];
            id<MTLFunction> moment = [lib newFunctionWithName:@"leaf_moments"];
            if (moment) moments = [device newComputePipelineStateWithFunction:moment error:NULL];
        });
        const uint32_t chunk = 16384, rows = count / trees;
        if (!buckets || !reduce || buckets.maxTotalThreadsPerThreadgroup < 64) return false;
        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        id<MTLBuffer> bits = [device newBufferWithBytes:packed length:count options:shared];
        id<MTLBuffer> eq = [device newBufferWithBytes:weights length:16ul * trees options:shared];
        id<MTLBuffer> partial = [device newBufferWithLength:4096ul * (count / chunk) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out = [device newBufferWithLength:4096ul * rows options:shared];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!bits || !eq || !partial || !out || !enc) return false;
        uint32_t args[] = {trees, chunk, count};
        [enc setComputePipelineState:buckets];
        [enc setBuffer:bits offset:0 atIndex:0];
        [enc setBuffer:eq offset:0 atIndex:1];
        [enc setBuffer:partial offset:0 atIndex:2];
        [enc setBytes:args length:sizeof(args) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(count / chunk, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        enc = [command computeCommandEncoder];
        if (!enc) return false;
        uint32_t shape[] = {trees / chunk, 256};
        [enc setComputePipelineState:reduce];
        [enc setBuffer:partial offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc setBytes:shape length:sizeof(shape) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(rows * 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        id<MTLBuffer> third_out = nil;
        if (third) {
            if (!moments) return false;
            uint32_t third_count = count / 2;
            id<MTLBuffer> third_partial =
                [device newBufferWithLength:1296ul * (third_count / chunk) options:MTLResourceStorageModePrivate];
            third_out = [device newBufferWithLength:1296ul * (rows / 2) options:shared];
            if (!third_partial || !third_out) return false;
            enc = [command computeCommandEncoder];
            if (!enc) return false;
            args[2] = third_count;
            [enc setComputePipelineState:moments];
            [enc setBuffer:bits offset:0 atIndex:0];
            [enc setBuffer:eq offset:0 atIndex:1];
            [enc setBuffer:third_partial offset:0 atIndex:2];
            [enc setBytes:args length:sizeof(args) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(third_count / chunk, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            enc = [command computeCommandEncoder];
            if (!enc) return false;
            shape[1] = 81;
            [enc setComputePipelineState:reduce];
            [enc setBuffer:third_partial offset:0 atIndex:0];
            [enc setBuffer:third_out offset:0 atIndex:1];
            [enc setBytes:shape length:sizeof(shape) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(81 * (rows / 2), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
        }
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return false;
        memcpy(result, out.contents, 4096ul * rows);
        if (third) memcpy(third, third_out.contents, 1296ul * (rows / 2));
        return true;
    }
}
