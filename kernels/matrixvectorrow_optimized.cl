#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// Optimized Matrix-Vector Row Major (32 threads, compatible with benchmark)
// Preserves exact TornadoVM signature and output

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
    // Typed pointers (TornadoVM +4 float header offset)
    __global const float * restrict x_f = ((__global const float *)x) + 4;
    __global float * restrict hb_f = ((__global float *)hb) + 4;
    __global const float * restrict w_f = ((__global const float *)w) + 4;

    __local float adf_3[32];

    int i_4 = get_group_id(0);
    if (i_4 >= 2048) return;

    int i_8 = get_local_id(0);

    // Row base pointer
    __global const float * restrict rowW = w_f + (i_4 * 8192);

    // 4x unrolled accumulation with float4 vectors
    // Each thread handles 4 floats per iteration, stride = 32*4 = 128
    const int vecStride = 128;  // 32 threads * 4 floats
    int vecIdx = i_8 * 4;

    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);

    // Main loop: 4x unrolled, processes 512 elements per iteration
    // 8192 / 512 = 16 full iterations
    for (; vecIdx < 8192 - 511; vecIdx += vecStride * 4) {
        float4 w0 = vload4(0, rowW + vecIdx);
        float4 x0 = vload4(0, x_f + vecIdx);
        acc0 = fma(w0, x0, acc0);

        float4 w1 = vload4(0, rowW + vecIdx + vecStride);
        float4 x1 = vload4(0, x_f + vecIdx + vecStride);
        acc1 = fma(w1, x1, acc1);

        float4 w2 = vload4(0, rowW + vecIdx + vecStride * 2);
        float4 x2 = vload4(0, x_f + vecIdx + vecStride * 2);
        acc2 = fma(w2, x2, acc2);

        float4 w3 = vload4(0, rowW + vecIdx + vecStride * 3);
        float4 x3 = vload4(0, x_f + vecIdx + vecStride * 3);
        acc3 = fma(w3, x3, acc3);
    }

    // Merge accumulators
    float4 partial4 = acc0 + acc1 + acc2 + acc3;

    // Handle remaining elements
    for (; vecIdx < 8192 - 3; vecIdx += vecStride) {
        float4 wv = vload4(0, rowW + vecIdx);
        float4 xv = vload4(0, x_f + vecIdx);
        partial4 = fma(wv, xv, partial4);
    }

    // Reduce float4 to scalar
    float f_9 = partial4.x + partial4.y + partial4.z + partial4.w;

    // Store partial sum to local memory
    adf_3[i_8] = f_9;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Unrolled parallel reduction for 32 threads
    if (i_8 < 16) { adf_3[i_8] += adf_3[i_8 + 16]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i_8 < 8)  { adf_3[i_8] += adf_3[i_8 + 8]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i_8 < 4)  { adf_3[i_8] += adf_3[i_8 + 4]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i_8 < 2)  { adf_3[i_8] += adf_3[i_8 + 2]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i_8 < 1)  { adf_3[i_8] += adf_3[i_8 + 1]; }

    // Write result
    if (i_8 == 0) {
        hb_f[i_4] = adf_3[0];
    }
}
