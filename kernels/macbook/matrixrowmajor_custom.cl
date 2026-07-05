/*
 * Optimized Matrix-Vector Multiplication Kernels for Apple M4 Pro Max
 *
 * IMPORTANT: Apple M4 OpenCL supports max work-group size of 32!
 * These kernels dynamically adapt to the work-group size provided.
 *
 * Optimizations applied:
 * - Vectorized loads (vload4) for coalesced memory access
 * - Multiple accumulators to hide memory latency
 * - 4x loop unrolling
 * - Unrolled parallel reduction (macro-based, no loop overhead)
 * - restrict keyword for compiler optimization hints
 *
 * Bottleneck: MEMORY-BOUND (2 global reads per fma, AI ~0.25 FLOP/byte)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ============================================================================
// Common Definitions
// ============================================================================

#define FLOAT_BASE_INDEX 4      // TornadoVM float array header offset (16 bytes / 4 = 4 floats)
#define HALF_BASE_INDEX 8       // TornadoVM half array header offset (16 bytes / 2 = 8 halfs)
#define CHAR_BASE_INDEX 16      // TornadoVM char array header offset (16 bytes)
#define VEC_SIZE 4              // Vector width for vload4
#define BLOCK_SIZE 32           // Quantization block size for int8 weights

// Dynamic reduction macro - works with any work-group size up to 32
#define REDUCE_STEP_DYN(scratch, lid, localSize, sz) \
    if ((localSize) >= (sz)) { \
        if ((lid) < ((sz) >> 1)) { \
            (scratch)[(lid)] += (scratch)[(lid) + ((sz) >> 1)]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    }

// ============================================================================
// Kernel 1: matrixVectorGeneric (FP32 weights, local memory reduction)
// ============================================================================

__kernel void matrixVectorGeneric(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar * restrict x,
    __global uchar * restrict hb,
    __global uchar * restrict w,
    __private int n,
    __private int d,
    __private int localWorkGroupSize)
{
    __local float scratch[32];  // Max 32 for Apple M4

    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);  // Dynamic work-group size

    if (row >= d) return;

    // Get typed pointers with TornadoVM header offset
    __global float * restrict weights = ((__global float *)w) + FLOAT_BASE_INDEX;
    __global float * restrict input = ((__global float *)x) + FLOAT_BASE_INDEX;
    __global float * restrict output = ((__global float *)hb) + FLOAT_BASE_INDEX;

    // Row start in weight matrix
    __global float * restrict rowWeights = weights + row * n;

    // Multiple float4 accumulators to hide memory latency
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);

    // Vectorized loop parameters - adapt to actual work-group size
    const int vecStride = localSize * VEC_SIZE;       // e.g., 32 * 4 = 128
    const int unrollSpan = vecStride * 4;             // 4x unroll
    const int vecStart = lid * VEC_SIZE;
    const int vecLimit = (n / unrollSpan) * unrollSpan;

    // Main vectorized + unrolled loop
    for (int i = vecStart; i < vecLimit; i += unrollSpan) {
        float4 w0 = vload4(0, rowWeights + i);
        float4 x0 = vload4(0, input + i);
        acc0 = fma(w0, x0, acc0);

        float4 w1 = vload4(0, rowWeights + i + vecStride);
        float4 x1 = vload4(0, input + i + vecStride);
        acc1 = fma(w1, x1, acc1);

        float4 w2 = vload4(0, rowWeights + i + vecStride * 2);
        float4 x2 = vload4(0, input + i + vecStride * 2);
        acc2 = fma(w2, x2, acc2);

        float4 w3 = vload4(0, rowWeights + i + vecStride * 3);
        float4 x3 = vload4(0, input + i + vecStride * 3);
        acc3 = fma(w3, x3, acc3);
    }

    // Handle remaining elements (scalar loop)
    float scalarAcc = 0.0f;
    for (int i = vecLimit + lid; i < n; i += localSize) {
        scalarAcc = fma(rowWeights[i], input[i], scalarAcc);
    }

    // Combine all accumulators
    float4 combined = acc0 + acc1 + acc2 + acc3;
    float sum = combined.x + combined.y + combined.z + combined.w + scalarAcc;

    // Store to local memory for parallel reduction
    scratch[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Dynamic parallel reduction (supports up to 32 threads)
    REDUCE_STEP_DYN(scratch, lid, localSize, 32);
    REDUCE_STEP_DYN(scratch, lid, localSize, 16);
    REDUCE_STEP_DYN(scratch, lid, localSize, 8);
    REDUCE_STEP_DYN(scratch, lid, localSize, 4);
    REDUCE_STEP_DYN(scratch, lid, localSize, 2);

    // Thread 0 writes final result
    if (lid == 0) {
        output[row] = scratch[0];
    }
}

// ============================================================================
// Kernel 2: matrixVectorParallel (FP32 weights, no reduction - 1 thread per row)
// ============================================================================

__kernel void matrixVectorParallel(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar * restrict x,
    __global uchar * restrict hb,
    __global uchar * restrict w,
    __private int n,
    __private int d)
{
    const int row = get_global_id(0);

    if (row >= d) return;

    __global float * restrict weights = ((__global float *)w) + FLOAT_BASE_INDEX;
    __global float * restrict input = ((__global float *)x) + FLOAT_BASE_INDEX;
    __global float * restrict output = ((__global float *)hb) + FLOAT_BASE_INDEX;

    __global float * restrict rowWeights = weights + row * n;

    // Multiple float4 accumulators
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);

    const int vecLimit = (n / 16) * 16;  // 4 * 4 = 16 elements per iteration

    // Vectorized + unrolled main loop (16 elements per iteration)
    for (int i = 0; i < vecLimit; i += 16) {
        float4 w0 = vload4(0, rowWeights + i);
        float4 x0 = vload4(0, input + i);
        acc0 = fma(w0, x0, acc0);

        float4 w1 = vload4(0, rowWeights + i + 4);
        float4 x1 = vload4(0, input + i + 4);
        acc1 = fma(w1, x1, acc1);

        float4 w2 = vload4(0, rowWeights + i + 8);
        float4 x2 = vload4(0, input + i + 8);
        acc2 = fma(w2, x2, acc2);

        float4 w3 = vload4(0, rowWeights + i + 12);
        float4 x3 = vload4(0, input + i + 12);
        acc3 = fma(w3, x3, acc3);
    }

    // Scalar remainder
    float scalarAcc = 0.0f;
    for (int i = vecLimit; i < n; i++) {
        scalarAcc = fma(rowWeights[i], input[i], scalarAcc);
    }

    // Combine accumulators and write output
    float4 combined = acc0 + acc1 + acc2 + acc3;
    output[row] = combined.x + combined.y + combined.z + combined.w + scalarAcc;
}

// ============================================================================
// Kernel 3: matrixVectorGenericFP16 (FP16 weights, local memory reduction)
// ============================================================================

__kernel void matrixVectorGenericFP16(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar * restrict x,
    __global uchar * restrict hb,
    __global uchar * restrict w,
    __private int n,
    __private int d,
    __private int localWorkGroupSize)
{
    __local float scratch[32];

    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);

    if (row >= d) return;

    // FP16 weights have different header offset
    __global half * restrict weights = ((__global half *)w) + HALF_BASE_INDEX;
    __global float * restrict input = ((__global float *)x) + FLOAT_BASE_INDEX;
    __global float * restrict output = ((__global float *)hb) + FLOAT_BASE_INDEX;

    __global half * restrict rowWeights = weights + row * n;

    // Float4 accumulators (compute in FP32 for precision)
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);

    const int vecStride = localSize * VEC_SIZE;
    const int unrollSpan = vecStride * 4;
    const int vecStart = lid * VEC_SIZE;
    const int vecLimit = (n / unrollSpan) * unrollSpan;

    // Main loop: load half4, convert to float4, accumulate
    for (int i = vecStart; i < vecLimit; i += unrollSpan) {
        half4 wh0 = vload4(0, rowWeights + i);
        float4 w0 = convert_float4(wh0);
        float4 x0 = vload4(0, input + i);
        acc0 = fma(w0, x0, acc0);

        half4 wh1 = vload4(0, rowWeights + i + vecStride);
        float4 w1 = convert_float4(wh1);
        float4 x1 = vload4(0, input + i + vecStride);
        acc1 = fma(w1, x1, acc1);

        half4 wh2 = vload4(0, rowWeights + i + vecStride * 2);
        float4 w2 = convert_float4(wh2);
        float4 x2 = vload4(0, input + i + vecStride * 2);
        acc2 = fma(w2, x2, acc2);

        half4 wh3 = vload4(0, rowWeights + i + vecStride * 3);
        float4 w3 = convert_float4(wh3);
        float4 x3 = vload4(0, input + i + vecStride * 3);
        acc3 = fma(w3, x3, acc3);
    }

    // Scalar remainder
    float scalarAcc = 0.0f;
    for (int i = vecLimit + lid; i < n; i += localSize) {
        scalarAcc = fma(convert_float(rowWeights[i]), input[i], scalarAcc);
    }

    // Combine accumulators
    float4 combined = acc0 + acc1 + acc2 + acc3;
    float sum = combined.x + combined.y + combined.z + combined.w + scalarAcc;

    // Parallel reduction
    scratch[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    REDUCE_STEP_DYN(scratch, lid, localSize, 32);
    REDUCE_STEP_DYN(scratch, lid, localSize, 16);
    REDUCE_STEP_DYN(scratch, lid, localSize, 8);
    REDUCE_STEP_DYN(scratch, lid, localSize, 4);
    REDUCE_STEP_DYN(scratch, lid, localSize, 2);

    if (lid == 0) {
        output[row] = scratch[0];
    }
}

// ============================================================================
// Kernel 4: matrixVectorGenericFinal (INT8 quantized weights + FP16 scales)
// ============================================================================

__kernel void matrixVectorGenericFinal(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar * restrict x,
    __global uchar * restrict output_buf,
    __global uchar * restrict weightsQ,
    __global uchar * restrict weightScales,
    __private int dim1,     // n (vector length / columns)
    __private int dim0,     // d (number of rows)
    __private int localWorkGroupSize)
{
    __local float scratch[32];

    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);

    if (row >= dim0) return;

    // Pointers with appropriate TornadoVM header offsets
    __global float * restrict input = ((__global float *)x) + FLOAT_BASE_INDEX;
    __global float * restrict output = ((__global float *)output_buf) + FLOAT_BASE_INDEX;
    __global char * restrict weights = ((__global char *)weightsQ) + CHAR_BASE_INDEX;
    __global half * restrict scales = ((__global half *)weightScales) + HALF_BASE_INDEX;

    // Row-specific pointers
    __global char * restrict rowWeights = weights + row * dim1;
    __global half * restrict rowScales = scales + row * (dim1 / BLOCK_SIZE);

    // Multiple float4 accumulators for hiding memory latency
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);

    // Each thread processes 4 elements at a time with dynamic stride
    const int stride = localSize * VEC_SIZE;
    const int vecLimit = (dim1 / stride) * stride;

    // Main vectorized loop
    for (int i = lid * VEC_SIZE; i < vecLimit; i += stride) {
        // Determine quantization block and load scale
        int blockIdx = i / BLOCK_SIZE;
        float scale = convert_float(rowScales[blockIdx]);

        // Load 4 int8 weights using char4
        char4 wq = vload4(0, rowWeights + i);

        // Dequantize: convert int8 to float and multiply by scale
        float4 w;
        w.x = (float)((int)wq.x) * scale;
        w.y = (float)((int)wq.y) * scale;
        w.z = (float)((int)wq.z) * scale;
        w.w = (float)((int)wq.w) * scale;

        // Load float4 input
        float4 xv = vload4(0, input + i);

        // Accumulate
        acc0 = fma(w, xv, acc0);
    }

    // Handle remainder elements (scalar)
    float scalarAcc = 0.0f;
    for (int i = vecLimit + lid; i < dim1; i += localSize) {
        int blockIdx = i / BLOCK_SIZE;
        float scale = convert_float(rowScales[blockIdx]);
        float w = (float)((int)rowWeights[i]) * scale;
        scalarAcc = fma(w, input[i], scalarAcc);
    }

    // Combine all accumulators
    float sum = acc0.x + acc0.y + acc0.z + acc0.w +
                acc1.x + acc1.y + acc1.z + acc1.w + scalarAcc;

    // Parallel reduction
    scratch[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    REDUCE_STEP_DYN(scratch, lid, localSize, 32);
    REDUCE_STEP_DYN(scratch, lid, localSize, 16);
    REDUCE_STEP_DYN(scratch, lid, localSize, 8);
    REDUCE_STEP_DYN(scratch, lid, localSize, 4);
    REDUCE_STEP_DYN(scratch, lid, localSize, 2);

    if (lid == 0) {
        output[row] = scratch[0];
    }
}
