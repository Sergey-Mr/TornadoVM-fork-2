#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// Tornado FloatArray uses a 16-byte (4 float) header before raw data
#define FLOAT_BASE_INDEX 4

// Tune for RTX 4090: 16x16 tile is a good default (256 threads / block)
#define TS 16

inline __global const float *restrict get_ro_float_ptr(__global const uchar *ptr) {
    return ((__global const float *)ptr) + FLOAT_BASE_INDEX;
}
inline __global float *restrict get_rw_float_ptr(__global uchar *ptr) {
    return ((__global float *)ptr) + FLOAT_BASE_INDEX;
}

// Work-group computes a TSxTS tile of C using shared (local) memory tiling.
__attribute__((reqd_work_group_size(TS, TS, 1)))
__kernel void matrixMultiplication(__global long *_kernel_context,
                                   __constant uchar *_constant_region,
                                   __local uchar *_local_region,
                                   __global int *_atomics,
                                   __global uchar *A,
                                   __global uchar *B,
                                   __global uchar *C,
                                   __private int size)
{
    (void)_constant_region;
    (void)_local_region;
    (void)_atomics;
    (void)size; // keep signature; original uses _kernel_context[0]

    const int N = (int)_kernel_context[0];

    __global const float *restrict a = get_ro_float_ptr(A);
    __global const float *restrict b = get_ro_float_ptr(B);
    __global float *restrict c = get_rw_float_ptr(C);

    // Global element this thread is responsible for (same mapping as original, but faster)
    const int col0 = get_global_id(0);
    const int row0 = get_global_id(1);

    // Strided coverage (preserves original behavior if global sizes < N)
    const int colStep = get_global_size(0);
    const int rowStep = get_global_size(1);

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    __local float As[TS][TS];
    __local float Bs[TS][TS];

    for (int row = row0; row < N; row += rowStep) {
        for (int col = col0; col < N; col += colStep) {

            float acc = 0.0f;

            // Tile over K dimension
            for (int k0 = 0; k0 < N; k0 += TS) {

                // Cooperative, coalesced loads into local memory
                const int aCol = k0 + lx;
                const int bRow = k0 + ly;

                As[ly][lx] = (aCol < N) ? a[row * N + aCol] : 0.0f;
                Bs[ly][lx] = (bRow < N) ? b[bRow * N + col] : 0.0f;

                barrier(CLK_LOCAL_MEM_FENCE);

                // Compute partial dot product for this C(row,col)
                #pragma unroll
                for (int k = 0; k < TS; k += 4) {
                    acc = fma(As[ly][k + 0], Bs[k + 0][lx], acc);
                    acc = fma(As[ly][k + 1], Bs[k + 1][lx], acc);
                    acc = fma(As[ly][k + 2], Bs[k + 2][lx], acc);
                    acc = fma(As[ly][k + 3], Bs[k + 3][lx], acc);
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            c[row * N + col] = acc;
        }
    }
}
