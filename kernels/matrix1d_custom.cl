/*
 * Optimized Matrix Multiplication Kernel for Apple M4 Pro Max
 *
 * KEY INSIGHT: Apple M4 has Unified Memory Architecture (UMA)
 * - Global memory latency is already low (~100 cycles vs ~400 on discrete GPU)
 * - Max work-group size is only 32 threads (too small for effective tiling)
 * - Local memory tiling adds barrier overhead that exceeds the benefit
 *
 * STRATEGY: Use register-level optimizations instead of local memory tiling
 * - Vectorized loads (float4)
 * - Multiple accumulators for ILP
 * - Loop unrolling
 * - No barriers (avoids synchronization overhead)
 *
 * Bottleneck: MEMORY-BOUND
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TornadoVM header offset
#define FLOAT_BASE_INDEX 4

/*
 * Main optimized kernel - vectorized without tiling
 * Best for Apple M4's unified memory architecture
 */
__kernel void matrixMultiplication(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar * restrict matrixA,
    __global uchar * restrict matrixB,
    __global uchar * restrict result,
    __private int size)
{
    __global float * restrict A = ((__global float *)matrixA) + FLOAT_BASE_INDEX;
    __global float * restrict B = ((__global float *)matrixB) + FLOAT_BASE_INDEX;
    __global float * restrict C = ((__global float *)result) + FLOAT_BASE_INDEX;

    const int col = get_global_id(0);  // Column in C
    const int row = get_global_id(1);  // Row in C

    if (col >= size || row >= size) return;

    // Pointer to row of A (contiguous access)
    __global float * restrict rowA = A + row * size;

    // Multiple accumulators for instruction-level parallelism
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    // Unrolled loop (4x unroll)
    const int limit4 = (size / 4) * 4;

    for (int k = 0; k < limit4; k += 4) {
        // Load 4 elements from A (contiguous, fast)
        float a0 = rowA[k];
        float a1 = rowA[k + 1];
        float a2 = rowA[k + 2];
        float a3 = rowA[k + 3];

        // Load 4 elements from B (strided access)
        float b0 = B[(k + 0) * size + col];
        float b1 = B[(k + 1) * size + col];
        float b2 = B[(k + 2) * size + col];
        float b3 = B[(k + 3) * size + col];

        // Accumulate (4 independent chains)
        acc0 = fma(a0, b0, acc0);
        acc1 = fma(a1, b1, acc1);
        acc2 = fma(a2, b2, acc2);
        acc3 = fma(a3, b3, acc3);
    }

    // Handle remainder
    float accRem = 0.0f;
    for (int k = limit4; k < size; k++) {
        accRem = fma(rowA[k], B[k * size + col], accRem);
    }

    // Combine and store
    C[row * size + col] = acc0 + acc1 + acc2 + acc3 + accRem;
}


/*
 * Alternative: 8x unrolled version for larger matrices
 */
__kernel void matrixMultiplication8x(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar * restrict matrixA,
    __global uchar * restrict matrixB,
    __global uchar * restrict result,
    __private int size)
{
    __global float * restrict A = ((__global float *)matrixA) + FLOAT_BASE_INDEX;
    __global float * restrict B = ((__global float *)matrixB) + FLOAT_BASE_INDEX;
    __global float * restrict C = ((__global float *)result) + FLOAT_BASE_INDEX;

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (col >= size || row >= size) return;

    __global float * restrict rowA = A + row * size;

    // 8 accumulators for maximum ILP
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const int limit8 = (size / 8) * 8;

    for (int k = 0; k < limit8; k += 8) {
        acc0 = fma(rowA[k+0], B[(k+0) * size + col], acc0);
        acc1 = fma(rowA[k+1], B[(k+1) * size + col], acc1);
        acc2 = fma(rowA[k+2], B[(k+2) * size + col], acc2);
        acc3 = fma(rowA[k+3], B[(k+3) * size + col], acc3);
        acc4 = fma(rowA[k+4], B[(k+4) * size + col], acc4);
        acc5 = fma(rowA[k+5], B[(k+5) * size + col], acc5);
        acc6 = fma(rowA[k+6], B[(k+6) * size + col], acc6);
        acc7 = fma(rowA[k+7], B[(k+7) * size + col], acc7);
    }

    // Remainder
    float accRem = 0.0f;
    for (int k = limit8; k < size; k++) {
        accRem = fma(rowA[k], B[k * size + col], accRem);
    }

    C[row * size + col] = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + accRem;
}


/*
 * Transposed B version - better memory access pattern
 * Requires B to be pre-transposed: B_T[col][k] = B[k][col]
 * Both A and B_T have contiguous row access
 */
__kernel void matrixMultiplicationTransposed(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar * restrict matrixA,
    __global uchar * restrict matrixB_T,  // Pre-transposed!
    __global uchar * restrict result,
    __private int size)
{
    __global float * restrict A = ((__global float *)matrixA) + FLOAT_BASE_INDEX;
    __global float * restrict B_T = ((__global float *)matrixB_T) + FLOAT_BASE_INDEX;
    __global float * restrict C = ((__global float *)result) + FLOAT_BASE_INDEX;

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (col >= size || row >= size) return;

    __global float * restrict rowA = A + row * size;
    __global float * restrict colB = B_T + col * size;  // Now contiguous!

    // Use float4 for vectorized loads (both are now contiguous)
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);

    const int limit8 = (size / 8) * 8;

    for (int k = 0; k < limit8; k += 8) {
        float4 a0 = vload4(0, rowA + k);
        float4 a1 = vload4(0, rowA + k + 4);
        float4 b0 = vload4(0, colB + k);
        float4 b1 = vload4(0, colB + k + 4);

        acc0 = fma(a0, b0, acc0);
        acc1 = fma(a1, b1, acc1);
    }

    // Remainder
    float accRem = 0.0f;
    for (int k = limit8; k < size; k++) {
        accRem = fma(rowA[k], colB[k], accRem);
    }

    float4 combined = acc0 + acc1;
    C[row * size + col] = combined.x + combined.y + combined.z + combined.w + accRem;
}
