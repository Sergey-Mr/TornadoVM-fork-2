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
    ulong ul_0, ul_1, ul_2;
    float f_20, f_30, f_15, f_9, f_25, f_27, f_28, f_21;
    long  l_18, l_34, l_17, l_33, l_12, l_13;
    bool  b_24, b_5, b_31;
    int   i_4, i_7, i_23, i_8, i_6, i_22, i_11, i_10, i_26, i_16, i_32, i_29;

    // Original pointer captures (unused now, but kept for minimal diff)
    ul_0 = (ulong) x;
    ul_1 = (ulong) hb;
    ul_2 = (ulong) w;

    // NEW: typed views
    __global float *x_f  = (__global float *) x;
    __global float *hb_f = (__global float *) hb;
    __global float *w_f  = (__global float *) w;

    __local float adf_3[32];

    i_4 = get_group_id(0);
    b_5 = i_4 < 2048;
    if (b_5)
    {
        i_6 = i_4 << 13;
        i_7 = i_6 + 4;
        i_8 = get_local_id(0);

        // BLOCK 2 MERGES [1 3 ]
        f_9  = 0.0f;
        i_10 = i_8;
        for (; i_10 < 8192; )
        {
            // BLOCK 3
            i_11 = i_7 + i_10;
            // ul_14 = ul_2 + ((long)i_11 << 2);
            f_15 = w_f[i_11];

            i_16 = i_10 + 4;
            // ul_19 = ul_0 + ((long)i_16 << 2);
            f_20 = x_f[i_16];

            f_21 = fma(f_15, f_20, f_9);
            i_22 = i_10 + 32;
            f_9  = f_21;
            i_10 = i_22;
        }

        adf_3[i_8] = f_9;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduction: unchanged
        i_23 = 16;
        for (; i_23 >= 1; )
        {
            b_24 = i_8 < i_23;
            if (b_24)
            {
                f_25 = adf_3[i_8];
                i_26 = i_23 + i_8;
                f_27 = adf_3[i_26];
                f_28 = f_25 + f_27;
                adf_3[i_8] = f_28;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            i_29 = i_23 >> 1;
            i_23 = i_29;
        }

        f_30 = adf_3[0];
        b_31 = (i_8 == 0);
        if (b_31)
        {
            i_32 = i_4 + 4;
            // ul_35 = ul_1 + ((long)i_32 << 2);
            hb_f[i_32] = f_30;
        }
        return;
    }
    else
    {
        return;
    }
}
