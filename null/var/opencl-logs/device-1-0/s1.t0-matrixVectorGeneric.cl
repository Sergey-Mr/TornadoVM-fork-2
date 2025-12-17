// Common extensions
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable


//============================================================
// 1) matrixVectorGeneric  (FP32, 32 threads cooperate on a row)
//============================================================
__attribute__((reqd_work_group_size(32, 1, 1)))
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
    // Assume original dimensions:
    const int N_ROWS = 2048;
    const int D_COLS = 8192;

    const int row  = get_group_id(0);
    if (row >= N_ROWS) return;

    const int lid  = get_local_id(0);
    const int lsz  = get_local_size(0); // should be 32

    __local float partial[32];

    // Reinterpret input buffers as typed pointers.
    __global const float *x_f = (__global const float *)x;
    __global const float *w_f = (__global const float *)w;
    __global       float *hb_f = (__global       float *)hb;

    // Original layout:
    // w index: ((row << 13) + 4) + j  (== row * 8192 + 4 + j)
    // x index: 4 + j
    const int w_row_base = (row << 13) + 4;  // row * 8192 + 4
    const int x_base     = 4;

    float sum = 0.0f;

    // Each thread processes strided elements to keep loads coalesced.
    // (j = lid, lid + lsz, lid + 2*lsz, ...)
    for (int j = lid; j < D_COLS; j += lsz) {
        float wv = w_f[w_row_base + j];
        float xv = x_f[x_base + j];
        sum = fma(wv, xv, sum);
    }

    // Write partial result into local memory.
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in local memory.
    // Known group size (32) lets the compiler unroll this nicely.
    for (int stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Thread 0 writes final result.
    if (lid == 0) {
        // Original code writes to hb[row + 4]
        hb_f[row + 4] = partial[0];
    }
}


//============================================================
// 2) matrixVectorParallel  (FP32, one work-item per row, strided over rows)
//============================================================
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
    const int N_ROWS = 2048;
    const int D_COLS = 8192;

    const int gid     = get_global_id(0);
    const int gsize   = get_global_size(0);

    __global const float *x_f  = (__global const float *)x;
    __global       float *hb_f = (__global       float *)hb;
    __global const float *w_f  = (__global const float *)w;

    const int x_base = 4;

    // Each global thread processes multiple rows in a grid-stride loop.
    for (int row = gid; row < N_ROWS; row += gsize) {

        // Original: i_6 = row << 13; i_7 = i_6 + 4;
        // => base index for w: row * 8192 + 4
        const int w_row_base = (row << 13) + 4;

        float sum = 0.0f;

        // Simple sequential dot product for this row.
        // Loads are sequential in both w and x and coalesced across rows.
        for (int j = 0; j < D_COLS; ++j) {
            float wv = w_f[w_row_base + j];
            float xv = x_f[x_base + j];
            sum = fma(wv, xv, sum);
        }

        // Original code: write to hb[row + 4]
        hb_f[row + 4] = sum;
    }
}


//============================================================
// 3) matrixVectorGenericFP16  (FP16 weights, FP32 accumulation)
//============================================================
__attribute__((reqd_work_group_size(32, 1, 1)))
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
    const int N_ROWS = 2048;
    const int D_COLS = 8192;

    const int row = get_group_id(0);
    if (row >= N_ROWS) return;

    const int lid = get_local_id(0);
    const int lsz = get_local_size(0); // expected 32

    __local float partial[32];

    __global const float *x_f  = (__global const float *)x;
    __global const half  *w_h  = (__global const half  *)w;
    __global       float *hb_f = (__global       float *)hb;

    // Original:
    // i_6 = row << 13; i_7 = i_6 + 8;
    // => base index for w (in half elements): row * 8192 + 8
    const int w_row_base = (row << 13) + 8;
    const int x_base     = 4;

    float sum = 0.0f;

    for (int j = lid; j < D_COLS; j += lsz) {
        // w index is w_row_base + j (half elements)
        half  w_half = w_h[w_row_base + j];
        float wv     = convert_float(w_half);
        float xv     = x_f[x_base + j];
        sum = fma(wv, xv, sum);
    }

    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Local reduction
    for (int stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        hb_f[row + 4] = partial[0];
    }
}


//============================================================
// 4) matrixVectorGenericFinal  (int8 weights + FP16 scales, FP32 accum)
//     Kept logic identical but cleaned indexing & types.
//============================================================
__attribute__((reqd_work_group_size(32, 1, 1)))
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
    const int N_ROWS = 2048;   // as in original
    const int D_COLS = 8192;   // logical dimension

    const int row = get_group_id(0);
    if (row >= N_ROWS) return;

    const int lid = get_local_id(0);
    const int lane4 = lid << 2;   // 4 * lid

    __local float partial[32];

    __global const float *x_f   = (__global const float *)x;
    __global       float *out_f = (__global       float *)output;
    __global const char  *w_q   = (__global const char  *)weightsQ;
    __global const half  *s_h   = (__global const half  *)weightScales;

    // Original layout reproduction
    //  - For weightsQ:
    //      base = (row << 13) = row * 8192
    //      we use four interleaved streams starting at +16..+19
    const int w_base      = (row << 13);
    const int w_base_0    = w_base + 16;
    const int w_base_1    = w_base + 17;
    const int w_base_2    = w_base + 18;
    const int w_base_3    = w_base + 19;

    //  - For weightScales (half):
    //      base_scales = (row << 8) + 8  (== row*256 + 8)
    //      each scale used for a group of 32 columns (i >> 5)
    const int s_row_base  = (row << 8) + 8;

    //  - For x:
    const int x_base = 4;

    // Four accumulators as in original
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    // First loop: processes chunks of 4 columns per thread, spaced by 128
    // i_20 starts at 4*lid (lane4) and increments by 128 (=4*32)
    for (int idx = lane4; idx < 8189; idx += 128) {

        // scale index: floor(idx / 32)
        const int scale_idx = s_row_base + (idx >> 5);
        float scale = convert_float(s_h[scale_idx]);

        // x indices (4 contiguous floats)
        float x0 = x_f[x_base + idx + 0];
        float x1 = x_f[x_base + idx + 1];
        float x2 = x_f[x_base + idx + 2];
        float x3 = x_f[x_base + idx + 3];

        // quantized weights for 4 "lanes"
        char q0 = w_q[w_base_0 + idx];
        char q1 = w_q[w_base_1 + idx];
        char q2 = w_q[w_base_2 + idx];
        char q3 = w_q[w_base_3 + idx];

        // dequantize and accumulate (matching original fma order)
        acc0 = fma(scale * (float)q0, x0, acc0);
        acc1 = fma(scale * (float)q1, x1, acc1);
        acc2 = fma(scale * (float)q2, x2, acc2);
        acc3 = fma(scale * (float)q3, x3, acc3);
    }

    // Combine four accumulators as in original
    float sum = (((acc0 + acc1) + acc2) + acc3);

    // Second loop (originally prepared for >8192 dims).
    // For your 8192-dim case this loop never executes, but
    // we keep it for correctness with the same bounds.
    int idx2 = lid + 8192;
    for (; idx2 < 8192; idx2 += 32) {
        const int scale_idx2 = s_row_base + (idx2 >> 5);
        float scale2 = convert_float(s_h[scale_idx2]);

        float x_val = x_f[x_base + idx2];
        char  q_val = w_q[w_base_0 + idx2]; // original used +i_11

        sum = fma(scale2 * (float)q_val, x_val, sum);
    }

    // Local reduction across 32 threads
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        // Original: output[row + 4]
        out_f[row + 4] = partial[0];
    }
}
