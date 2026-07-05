#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TS 16

__attribute__((reqd_work_group_size(TS, TS, 1)))
__kernel void matrixMultiplication(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar * restrict A,
    __global uchar * restrict B,
    __global uchar * restrict C,
    __private int size)
{
    #define FLOAT_BASE_INDEX 4

    __global float * restrict matA = ((__global float *)A) + FLOAT_BASE_INDEX;
    __global float * restrict matB = ((__global float *)B) + FLOAT_BASE_INDEX;
    __global float * restrict matC = ((__global float *)C) + FLOAT_BASE_INDEX;

    __local float tileA[TS][TS];
    __local float tileB[TS][TS];

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    float acc = 0.0f;
    const int numTiles = (size + TS - 1) / TS;

    for (int t = 0; t < numTiles; t++) {
        const int tileStart = t * TS;

        const int aCol = tileStart + lx;
        if (gy < size && aCol < size) {
            tileA[ly][lx] = matA[gy * size + aCol];
        } else {
            tileA[ly][lx] = 0.0f;
        }

        const int bRow = tileStart + ly;
        if (bRow < size && gx < size) {
            tileB[ly][lx] = matB[bRow * size + gx];
        } else {
            tileB[ly][lx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TS; k++) {
            acc = fma(tileA[ly][k], tileB[k][lx], acc);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gy < size && gx < size) {
        matC[gy * size + gx] = acc;
    }
}
