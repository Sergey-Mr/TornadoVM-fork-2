#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void computePi(__global long *_kernel_context,
                        __constant uchar *_constant_region,
                        __local uchar *_local_region,
                        __global int *_atomics,
                        __global uchar *input,
                        __global uchar *result)
{
    // Typed views
    __global float *in  = (__global float *)input;
    __global float *out = (__global float *)result;

    // Local scratch (same 256 size as original)
    __local float adf_2[256];

    const int gsize = get_global_size(0);
    const int gid   = get_global_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);

    // Initialize base accumulator slot once
    if (gid == 0) {
        out[6] = 0.0f;  // result + 24 bytes
    }

    // Each global thread processes terms: i = gid+1, gid+1+gsize, ...
    for (int i = gid + 1; i < 8192; i += gsize) {
        // Load previous partial from input (same +6 offset)
        float prev = in[i + 6];

        // Leibniz term: (-1)^(i+1) / (2*i - 1)
        // sign = +1 if (i+1) even, -1 if odd
        float sign = (((i + 1) & 1) ? -1.0f : 1.0f);
        float denom = (float)((i << 1) - 1);
        float term = sign / denom;

        float val = prev + term;

        // Store into local scratch
        adf_2[lid] = val;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Parallel reduction in local memory
        int stride = lsize >> 1;
        while (stride >= 1) {
            if (lid < stride) {
                adf_2[lid] += adf_2[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            stride >>= 1;
        }

        // Write one partial sum per group into result
        if (lid == 0) {
            int groupIndex = get_group_id(0) + 1;      // 1..N
            out[6 + groupIndex] = adf_2[0];            // result + 24 + 4*(group+1)
        }
        // No cross-group sync possible/needed; next i uses fresh data
    }
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void rAdd(__global long *_kernel_context,
                   __constant uchar *_constant_region,
                   __local uchar *_local_region,
                   __global int *_atomics,
                   __global uchar *array,
                   __private int size)
{
    __global float *arr = (__global float *)array;

    // Base index for partial sums (24 bytes = 6 floats)
    const int base = 6;

    float sum = 0.0f;

    // Sum 'size' floats: arr[base] .. arr[base + size - 1]
    for (int i = 0; i < size; ++i) {
        sum += arr[base + i];
    }

    // Store total back to arr[base] (same as original: *(array+24))
    arr[base] = sum;
}
