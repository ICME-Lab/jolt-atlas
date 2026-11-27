#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

static void dispatch(id<MTLComputeCommandEncoder> e, NSArray<id<MTLBuffer>> *buffers,
                     const uint32_t *args, NSUInteger arg_count, NSUInteger n) {
    for (NSUInteger i = 0; i < buffers.count; ++i) [e setBuffer:buffers[i] offset:0 atIndex:i];
    [e setBytes:args length:arg_count * sizeof(*args) atIndex:buffers.count];
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

bool atlas_metal_projection(const void *library, size_t library_size, const int32_t *input,
                           const int32_t *weights, const int32_t *bias,
                           const uint32_t *dims, int32_t *output, uint32_t *remainder) {
    @autoreleasepool {
        uint32_t rows = dims[0], inner = dims[1], cols = dims[2], bits = dims[3];
        uint32_t xd = dims[4], wd = dims[5];
        static dispatch_once_t once;
        static id<MTLDevice> dev;
        static id<MTLCommandQueue> queue;
        static id<MTLComputePipelineState> split, accumulate;
        dispatch_once(&once, ^{
            dev = MTLCreateSystemDefaultDevice();
            queue = [dev newCommandQueue];
            dispatch_data_t data = dispatch_data_create(library, library_size, nil,
                                                        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            id<MTLLibrary> lib = [dev newLibraryWithData:data error:NULL];
            split = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"split_digits"] error:NULL];
            accumulate = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"accumulate"] error:NULL];
        });
        if (!queue || !split || !accumulate) return false;

        size_t xlen = (size_t)rows * inner, wlen = (size_t)cols * inner, olen = (size_t)rows * cols;
        const MTLResourceOptions auto_shared = MTLResourceStorageModeShared;
        const MTLResourceOptions auto_private = MTLResourceStorageModePrivate;
        id<MTLBuffer> x = [dev newBufferWithBytes:input length:xlen*4 options:auto_shared];
        id<MTLBuffer> w = [dev newBufferWithBytes:weights length:wlen*4 options:auto_shared];
        id<MTLBuffer> b = [dev newBufferWithBytes:bias length:cols*4ul options:auto_shared];
        id<MTLBuffer> xp = [dev newBufferWithLength:xlen*4*xd options:auto_private];
        id<MTLBuffer> wp = [dev newBufferWithLength:wlen*4*wd options:auto_private];
        id<MTLBuffer> partial = [dev newBufferWithLength:olen*4 options:auto_private];
        id<MTLBuffer> sum = [dev newBufferWithLength:olen*8 options:auto_private];
        id<MTLBuffer> out = [dev newBufferWithLength:olen*4 options:auto_shared];
        id<MTLBuffer> rem = [dev newBufferWithLength:olen*4 options:auto_shared];
        uint32_t zero = 0;
        id<MTLBuffer> failure = [dev newBufferWithBytes:&zero length:4 options:auto_shared];
        if (!(x && w && b && xp && wp && partial && sum && out && rem && failure)) return false;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return false;
        [enc setComputePipelineState:split];
        uint32_t xargs[] = {(uint32_t)xlen, xd}, wargs[] = {(uint32_t)wlen, wd};
        dispatch(enc, @[x, xp], xargs, 2, xlen);
        dispatch(enc, @[w, wp], wargs, 2, wlen);
        [enc endEncoding];

        MPSMatrixDescriptor *ad = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:inner
                                          rowBytes:inner*4ul dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *bd = [MPSMatrixDescriptor matrixDescriptorWithRows:cols columns:inner
                                          rowBytes:inner*4ul dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *cd = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
                                          rowBytes:cols*4ul dataType:MPSDataTypeFloat32];
        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
            transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:cols
            interiorColumns:inner alpha:1.0 beta:0.0];
        MPSMatrix *c = [[MPSMatrix alloc] initWithBuffer:partial descriptor:cd];
        NSArray<id<MTLBuffer>> *buffers = @[partial, sum, b, out, rem, failure];
        for (uint32_t j=0; j<xd; j++) {
            MPSMatrix *a = [[MPSMatrix alloc] initWithBuffer:xp offset:j*xlen*4 descriptor:ad];
            for (uint32_t k=0; k<wd; k++) {
                MPSMatrix *m = [[MPSMatrix alloc] initWithBuffer:wp offset:k*wlen*4 descriptor:bd];
                [mm encodeToCommandBuffer:cmd leftMatrix:a rightMatrix:m resultMatrix:c];
                enc = [cmd computeCommandEncoder];
                if (!enc) return false;
                [enc setComputePipelineState:accumulate];
                uint32_t flags = (j==0 && k==0) | ((j+1==xd && k+1==wd) << 1);
                uint32_t config[] = {7*(j+k), cols, bits, flags};
                dispatch(enc, buffers, config, 4, olen);
                [enc endEncoding];
            }
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted || *(uint32_t *)failure.contents)
            return false;
        memcpy(output, out.contents, olen*4);
        memcpy(remainder, rem.contents, olen*4);
        return true;
    }
}
