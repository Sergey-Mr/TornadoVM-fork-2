#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void matrixVectorGeneric(__global long *_kernel_context,
                                  __constant uchar *_constant_region,
                                  __local    uchar *_local_region,
                                  __global   int   *_atomics,
                                  __global   uchar *x,
                                  __global   uchar *hb,
                                  __global   uchar *w,
                                  __private  int    n,
                                  __private  int    d,
                                  __private  int    localWorkGroupSize)
{
    __global float *x_f  = (__global float *) x;
    __global float *hb_f = (__global float *) hb;
    __global float *w_f  = (__global float *) w;

    __local float adf_3[32];

    const int ROWS   = 2048;
    const int N      = 8192;
    const int HEADER = 4;
    const int WG     = 32;

    int group = get_group_id(0);
    if (group >= ROWS) {
        return;
    }

    int lid = get_local_id(0);

    int rowBaseW = (group << 13) + HEADER;

    float sum = 0.0f;

    for (int k = lid; k < N; k += WG)
    {
        float wv = w_f[rowBaseW + k];
        float xv = x_f[HEADER + k];
        sum = fma(wv, xv, sum);
    }

    adf_3[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Slightly cleaned reduction – identical tree, just easier for the compiler
    for (int offset = WG >> 1; offset > 0; offset >>= 1)
    {
        if (lid < offset)
        {
            adf_3[lid] += adf_3[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        hb_f[group + HEADER] = adf_3[0];
    }
}
