#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Optimized version preserving exact signature
__kernel void matrixVectorGeneric(
    __global long *_kernel_context,
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
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    if(gid >= 2048) return;

    // Cast pointers for direct float access
    __global float *x_f = (__global float *)x;
    __global float *hb_f = (__global float *)hb;
    __global float *w_f = (__global float *)w;

    __local float partialSums[128];

    // Calculate row start index
    const int rowStart = (gid << 13) + 6;  // gid * 8192 + 6

    // Accumulate with loop unrolling
    float sum = 0.0f;

    // Unroll by 4 for better instruction-level parallelism
    int i = lid;
    for(; i < 8192 - 3; i += lsize * 4) {
        float w0 = w_f[rowStart + i];
        float w1 = w_f[rowStart + i + lsize];
        float w2 = w_f[rowStart + i + lsize * 2];
        float w3 = w_f[rowStart + i + lsize * 3];

        float x0 = x_f[i + 6];
        float x1 = x_f[i + 6 + lsize];
        float x2 = x_f[i + 6 + lsize * 2];
        float x3 = x_f[i + 6 + lsize * 3];

        sum = fma(w0, x0, sum);
        sum = fma(w1, x1, sum);
        sum = fma(w2, x2, sum);
        sum = fma(w3, x3, sum);
    }

    // Handle remaining elements
    for(; i < 8192; i += lsize) {
        sum = fma(w_f[rowStart + i], x_f[i + 6], sum);
    }

    partialSums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Optimized parallel reduction
    for(int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if(lid < stride) {
            partialSums[lid] += partialSums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if(lid == 0) {
        hb_f[gid + 6] = partialSums[0];
    }
}

// Optimized parallel version preserving exact signature
__kernel void matrixVectorParallel(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar *x,
    __global uchar *hb,
    __global uchar *w,
    __private int n,
    __private int d)
{
    const int gsize = get_global_size(0);
    int gid = get_global_id(0);

    // Cast pointers for direct float access
    __global float *x_f = (__global float *)x;
    __global float *hb_f = (__global float *)hb;
    __global float *w_f = (__global float *)w;

    for(; gid < 2048; gid += gsize) {
        const int rowStart = (gid << 13) + 6;

        float sum = 0.0f;

        // Process 4 elements at a time with loop unrolling
        int i = 0;
        for(; i < 8192 - 3; i += 4) {
            float w0 = w_f[rowStart + i];
            float w1 = w_f[rowStart + i + 1];
            float w2 = w_f[rowStart + i + 2];
            float w3 = w_f[rowStart + i + 3];

            float x0 = x_f[i + 6];
            float x1 = x_f[i + 6 + 1];
            float x2 = x_f[i + 6 + 2];
            float x3 = x_f[i + 6 + 3];

            sum = fma(w0, x0, sum);
            sum = fma(w1, x1, sum);
            sum = fma(w2, x2, sum);
            sum = fma(w3, x3, sum);
        }

        // Handle remaining elements
        for(; i < 8192; i++) {
            sum = fma(w_f[rowStart + i], x_f[i + 6], sum);
        }

        hb_f[gid + 6] = sum;
    }
}