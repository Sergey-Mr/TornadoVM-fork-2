#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// ============================================================================
// matrixVectorGeneric (FP32)
// ============================================================================
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
    // Reinterpret buffers as float arrays
    __global float *x_f  = (__global float *) x;
    __global float *hb_f = (__global float *) hb;
    __global float *w_f  = (__global float *) w;

    // Local reduction buffer
    __local float adf_3[32];

    const int group = get_group_id(0);

    // Original guard: only first 2048 groups are active
    if (group >= 2048) {
        return;
    }

    const int lid      = get_local_id(0);  // 0..31
    const int WG       = 32;               // local work-group size (fixed)
    const int N        = 8192;             // vector length
    const int HEADER   = 4;                // first 4 floats reserved (metadata)

    // Original:
    // i_6  = group << 13;            // group * 8192
    // i_7  = i_6 + 4;                // base index in w
    // Then w[i_7 + i_10], x[i_10 + 4]
    const int rowBaseW = (group << 13) + HEADER; // w row offset (float index)

    float sum = 0.0f;

    // for (i_10 = lid; i_10 < 8192; i_10 += 32)
    for (int k = lid; k < N; k += WG) {
        float wv = w_f[rowBaseW + k];  // w[group, k]
        float xv = x_f[HEADER + k];    // x[k]
        sum = fma(wv, xv, sum);
    }

    // Store partial sum in local memory
    adf_3[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction: offsets 16, 8, 4, 2, 1 (same pattern as original)
    for (int offset = 16; offset >= 1; offset >>= 1) {
        if (lid < offset) {
            adf_3[lid] = adf_3[lid] + adf_3[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Original:
    // i_32 = group + 4;
    // hb[i_32] = adf_3[0];
    if (lid == 0) {
        hb_f[group + HEADER] = adf_3[0];
    }
}


// ============================================================================
// matrixVectorParallel (FP32)
// ============================================================================
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void matrixVectorParallel(__global long *_kernel_context,
                                   __constant uchar *_constant_region,
                                   __local    uchar *_local_region,
                                   __global   int   *_atomics,
                                   __global   uchar *x,
                                   __global   uchar *hb,
                                   __global   uchar *w,
                                   __private  int    n,
                                   __private  int    d)
{
    __global float *x_f  = (__global float *) x;
    __global float *hb_f = (__global float *) hb;
    __global float *w_f  = (__global float *) w;

    const int gsize = get_global_size(0);
    const int gid   = get_global_id(0);

    const int N      = 8192;
    const int ROWS   = 2048;
    const int HEADER = 4;

    // Each work-item processes multiple rows in a grid-stride fashion:
    // for (i_5 = gid; i_5 < 2048; i_5 += gsize)
    for (int row = gid; row < ROWS; row += gsize) {

        // Original:
        // i_6 = row << 13;      // row * 8192
        // i_7 = i_6 + 4;        // base index in w
        const int rowBaseW = (row << 13) + HEADER;

        float sum = 0.0f;

        // for (i_9 = 0; i_9 < 8192; i_9++)
        for (int k = 0; k < N; ++k) {
            float wv = w_f[rowBaseW + k];  // w[row, k]
            float xv = x_f[HEADER + k];    // x[k]
            sum = fma(wv, xv, sum);
        }

        // Original:
        // i_22 = row + 4;
        // hb[i_22] = sum;
        hb_f[row + HEADER] = sum;
    }
}


// ============================================================================
// matrixVectorGenericFP16 (FP16 weights, FP32 x/hb)
// ============================================================================
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void matrixVectorGenericFP16(__global long *_kernel_context,
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
    __global half  *w_h  = (__global half  *) w;

    __local float adf_3[32];

    const int group = get_group_id(0);

    if (group >= 2048) {
        return;
    }

    const int lid      = get_local_id(0);  // 0..31
    const int WG       = 32;
    const int N        = 8192;
    const int X_HEADER = 4;
    const int W_HEADER = 8;               // from i_7 = (group<<13) + 8

    // Original:
    // i_6  = group << 13;         // group * 8192
    // i_7  = i_6 + 8;             // base index for half-weights
    const int rowBaseW = (group << 13) + W_HEADER;

    float sum = 0.0f;

    // for (i_10 = lid; i_10 < 8192; i_10 += 32)
    for (int k = lid; k < N; k += WG) {
        // w index: i_11 = i_7 + i_10; half at index (i_7 + i_10)
        half  wh = w_h[rowBaseW + k];
        float wv = convert_float((float) wh);

        // x index: i_16 = i_10 + 4
        float xv = x_f[X_HEADER + k];

        sum = fma(wv, xv, sum);
    }

    adf_3[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction with offsets 16,8,4,2,1 (unchanged)
    for (int offset = 16; offset >= 1; offset >>= 1) {
        if (lid < offset) {
            adf_3[lid] = adf_3[lid] + adf_3[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        // i_33 = group + 4;
        hb_f[group + 4] = adf_3[0];
    }
}


// ============================================================================
// matrixVectorGenericFinal (int8 weights + half scales, FP32 x/output)
// This is kept structurally identical but with clearer indexing.
// ============================================================================
#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  

__kernel void matrixVectorGenericFinal(__global long *_kernel_context,
                                       __constant uchar *_constant_region,
                                       __local    uchar *_local_region,
                                       __global   int   *_atomics,
                                       __global   uchar *x,
                                       __global   uchar *output,
                                       __global   uchar *weightsQ,
                                       __global   uchar *weightScales,
                                       __private  int    dim1,
                                       __private  int    dim0,
                                       __private  int    localWorkGroupSize)
{
    __global float *x_f   = (__global float *) x;
    __global float *out_f = (__global float *) output;
    __global char  *wq    = (__global char  *) weightsQ;
    __global half  *ws    = (__global half  *) weightScales;

    __local float adf_4[32];

    const int group = get_group_id(0);

    if (group >= 2048) {
        return;
    }

    const int lid = get_local_id(0);

    // Constants from original indexing:
    //
    // i_7  = group << 13;         // group * 8192
    // i_8  = i_7 + 19;
    // i_9  = i_7 + 18;
    // i_10 = i_7 + 17;
    // i_11 = i_7 + 16;
    //
    // i_12 = group << 8;          // group * 256
    // i_13 = i_12 + 8;
    const int ROW_OFF      = group << 13;   // 8192 * group
    const int W_BASE0      = ROW_OFF + 16;  // corresponds to i_11
    const int W_BASE1      = ROW_OFF + 17;  // i_10
    const int W_BASE2      = ROW_OFF + 18;  // i_9
    const int W_BASE3      = ROW_OFF + 19;  // i_8

    const int SCALE_BASE   = (group << 8) + 8;  // i_13

    const int X_HEADER     = 4;
    const int N            = 8192;
    const int WG           = 32;

    // i_15 = lid << 2;  (4 * lid)
    int idx = lid << 2;

    // Four partial accumulators (lanes)
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    // ---------------------------
    // First main loop (unrolled by 4)
    // for (i_20 = i_15; i_20 < 8189; i_20 += 128)
    // ---------------------------
    for (int pos = idx; pos < 8189; pos += (WG * 4)) {

        // Original:
        // block index = floor(pos / 32)
        int block      = pos >> 5;
        int scaleIndex = SCALE_BASE + block;

        half  hscale = ws[scaleIndex];
        float scale  = convert_float((float) hscale);

        // Lane 0
        //   weight index: i_30 = i_11 + pos  => W_BASE0 + pos
        //   x index     : (pos + 4)
        int wIdx0 = W_BASE0 + pos;
        int xIdx0 = pos + X_HEADER;
        int q0    = (int) wq[wIdx0];
        float x0  = x_f[xIdx0];
        acc0      = fma(scale * (float) q0, x0, acc0);

        // Lane 1
        //   weight index: i_39 = i_10 + pos  => W_BASE1 + pos
        //   x index     : (pos + 5)
        int wIdx1 = W_BASE1 + pos;
        int xIdx1 = pos + X_HEADER + 1;
        int q1    = (int) wq[wIdx1];
        float x1  = x_f[xIdx1];
        acc1      = fma(scale * (float) q1, x1, acc1);

        // Lane 2
        //   weight index: i_48 = i_9 + pos   => W_BASE2 + pos
        //   x index     : (pos + 6)
        int wIdx2 = W_BASE2 + pos;
        int xIdx2 = pos + X_HEADER + 2;
        int q2    = (int) wq[wIdx2];
        float x2  = x_f[xIdx2];
        acc2      = fma(scale * (float) q2, x2, acc2);

        // Lane 3
        //   weight index: i_57 = i_8 + pos   => W_BASE3 + pos
        //   x index     : (pos + 7)
        int wIdx3 = W_BASE3 + pos;
        int xIdx3 = pos + X_HEADER + 3;
        int q3    = (int) wq[wIdx3];
        float x3  = x_f[xIdx3];
        acc3      = fma(scale * (float) q3, x3, acc3);
    }

    // Combine the four partial accumulators
    float total = (acc0 + acc1) + (acc2 + acc3);

    // ---------------------------
    // Second loop (tail processing)
    // Original code:
    //   i_87 = lid + 8192;
    //   for (i_89 = i_87; i_89 < 8192; i_89 += 32) { ... }
    // For N = 8192, this loop is effectively empty, but we keep it for
    // semantic equivalence.
    // ---------------------------
    for (int pos = lid + N; pos < N; pos += WG) {

        int block      = pos >> 5;
        int scaleIndex = SCALE_BASE + block;

        half  hscale = ws[scaleIndex];
        float scale  = convert_float((float) hscale);

        // weight index: i_99 = pos + i_11 => W_BASE0 + pos
        int wIdx  = W_BASE0 + pos;
        int q     = (int) wq[wIdx];

        // x index: i_103 = pos + 4
        int xIdx  = pos + X_HEADER;
        float x0  = x_f[xIdx];

        total = fma(scale * (float) q, x0, total);
    }

    // Store partial result into local buffer for reduction
    adf_4[lid] = total;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction: same pattern as original (16,8,4,2,1)
    for (int offset = 16; offset >= 1; offset >>= 1) {
        if (lid < offset) {
            adf_4[lid] = adf_4[lid] + adf_4[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write final result to output[group + 4]
    if (lid == 0) {
        out_f[group + 4] = adf_4[0];
    }
}
