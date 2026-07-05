#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// ============================================================================
// Optimized Matrix-Vector Row Major for RTX 4090
// Computes: y[row] = sum_k(W[row][k] * x[k])
// Dimensions: W[2048][8192], x[8192], y[2048]
// ============================================================================

#define ROWS     2048
#define N        8192
#define HEADER   4
#define WG_SIZE  256

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void matrixVectorGeneric(__global long *_kernel_context,
                                  __constant uchar *_constant_region,
                                  __local uchar *_local_region,
                                  __global int *_atomics,
                                  __global uchar *x,
                                  __global uchar *hb,
                                  __global uchar *w,
                                  __private int n,
                                  __private int d,
                                  __private int localWorkGroupSize)
{
    // Typed pointer views (TornadoVM uses +4 float offset for headers)
    __global const float * restrict x_f  = ((__global const float *)x) + HEADER;
    __global float * restrict hb_f       = ((__global float *)hb) + HEADER;
    __global const float * restrict w_f  = ((__global const float *)w) + HEADER;

    __local float scratch[WG_SIZE];

    const int rowId = get_group_id(0);
    if (rowId >= ROWS) {
        return;
    }

    const int lid = get_local_id(0);

    // Row base pointer - each row has N elements
    __global const float * restrict rowW = w_f + (rowId * N);

    // ========================================================================
    // Main accumulation loop with 4x unrolling and float4 vector loads
    // Each thread processes elements at stride = WG_SIZE
    // ========================================================================
    const int vecStride = WG_SIZE * 4;       // 1024 elements per unroll iteration
    const int vecLimit = N - (vecStride * 4 - 4);  // safe limit for 4x unroll
    int vecIdx = lid * 4;                     // starting index for this thread

    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);

    // 4x unrolled vectorized loop
    for (; vecIdx < vecLimit; vecIdx += vecStride * 4) {
        // Iteration 0
        float4 w0 = vload4(0, rowW + vecIdx);
        float4 x0 = vload4(0, x_f + vecIdx);
        acc0 = fma(w0, x0, acc0);

        // Iteration 1
        float4 w1 = vload4(0, rowW + vecIdx + vecStride);
        float4 x1 = vload4(0, x_f + vecIdx + vecStride);
        acc1 = fma(w1, x1, acc1);

        // Iteration 2
        float4 w2 = vload4(0, rowW + vecIdx + vecStride * 2);
        float4 x2 = vload4(0, x_f + vecIdx + vecStride * 2);
        acc2 = fma(w2, x2, acc2);

        // Iteration 3
        float4 w3 = vload4(0, rowW + vecIdx + vecStride * 3);
        float4 x3 = vload4(0, x_f + vecIdx + vecStride * 3);
        acc3 = fma(w3, x3, acc3);
    }

    // Merge accumulators
    float4 partial4 = acc0 + acc1 + acc2 + acc3;

    // Handle remaining vectorized elements (1x stride)
    for (; vecIdx < N - 3; vecIdx += vecStride) {
        float4 wv = vload4(0, rowW + vecIdx);
        float4 xv = vload4(0, x_f + vecIdx);
        partial4 = fma(wv, xv, partial4);
    }

    // Reduce float4 to scalar
    float partial = partial4.x + partial4.y + partial4.z + partial4.w;

    // Handle tail elements (when N not divisible by 4)
    int tailStart = (N & ~3) + lid;
    for (int t = tailStart; t < N; t += WG_SIZE) {
        partial = fma(rowW[t], x_f[t], partial);
    }

    // ========================================================================
    // Parallel reduction in local memory
    // ========================================================================
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Unrolled reduction tree for WG_SIZE=256
    if (lid < 128) { scratch[lid] += scratch[lid + 128]; } barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 64)  { scratch[lid] += scratch[lid + 64]; }  barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 32)  { scratch[lid] += scratch[lid + 32]; }  barrier(CLK_LOCAL_MEM_FENCE);

    // Warp-level reduction (no barrier needed within a warp on NVIDIA)
    if (lid < 16) { scratch[lid] += scratch[lid + 16]; }
    if (lid < 8)  { scratch[lid] += scratch[lid + 8]; }
    if (lid < 4)  { scratch[lid] += scratch[lid + 4]; }
    if (lid < 2)  { scratch[lid] += scratch[lid + 2]; }
    if (lid < 1)  { scratch[lid] += scratch[lid + 1]; }

    // Write result
    if (lid == 0) {
        hb_f[rowId] = scratch[0];
    }
}
