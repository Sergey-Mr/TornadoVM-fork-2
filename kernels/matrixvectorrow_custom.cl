#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// Tornado FloatArray uses a 16-byte (4 float) header before the raw data.
// We materialise typed views once to avoid repeating pointer arithmetic.
#define FLOAT_BASE_INDEX 4
#define MAX_LOCAL_SIZE   512
#define REQD_LOCAL_SIZE  128
#define VEC_UNROLL       4

inline __global const float *restrict get_ro_float_ptr(__global const uchar *ptr) {
    return ((__global const float *)ptr) + FLOAT_BASE_INDEX;
}

inline __global float *restrict get_rw_float_ptr(__global uchar *ptr) {
    return ((__global float *)ptr) + FLOAT_BASE_INDEX;
}

inline void reduce_group(__local float *scratch, int localId, int localSize) {
#define REDUCE_STEP(SZ)                                                                               \
    if (localSize >= (SZ)) {                                                                          \
        int offset = (SZ) >> 1;                                                                       \
        if (localId < offset) {                                                                       \
            scratch[localId] += scratch[localId + offset];                                             \
        }                                                                                             \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                 \
    }

    REDUCE_STEP(512);
    REDUCE_STEP(256);
    REDUCE_STEP(128);
    REDUCE_STEP(64);
    REDUCE_STEP(32);
    REDUCE_STEP(16);
    REDUCE_STEP(8);
    REDUCE_STEP(4);
    REDUCE_STEP(2);

#undef REDUCE_STEP
}

__attribute__((reqd_work_group_size(REQD_LOCAL_SIZE, 1, 1)))
__kernel void matrixVectorGeneric(__global long *_kernel_context,
                                  __constant uchar *_constant_region,
                                  __local uchar *_local_region,
                                  __global int *_atomics,
                                  __global uchar *x,
                                  __global uchar *hb,
                                  __global uchar *w,
                                  __private int n,
                                  __private int d,
                                  __private int localWorkGroupSize) {
    (void)_kernel_context;
    (void)_constant_region;
    (void)_local_region;
    (void)_atomics;
    (void)localWorkGroupSize;

    __global const float *restrict input = get_ro_float_ptr(x);
    __global float *restrict output = get_rw_float_ptr(hb);
    __global const float *restrict weights = get_ro_float_ptr(w);

    const int rowId = get_group_id(0);
    if (rowId >= d) {
        return;
    }

    const int localId = get_local_id(0);
    const int localSize = get_local_size(0);

    __local float scratch[MAX_LOCAL_SIZE];

    // Each work item traverses the row with a stride equal to the work-group size.
    const __global float *rowWeights = weights + (rowId * n);
    const int vecSpan = n & ~3; // process multiples of 4 using vector loads
    const int vecStride = localSize << 2; // localSize * 4
    const int unrollSpan = vecStride * VEC_UNROLL;
    const int maxStart = (vecSpan >= 4) ? (vecSpan - 4) : 0;
    int vecLimit = maxStart - ((VEC_UNROLL - 1) * vecStride) + 1;
    if (vecLimit < 0) {
        vecLimit = 0;
    }
    int vecIndex = localId << 2; // localId * 4

    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);

    for (; vecIndex < vecLimit; vecIndex += unrollSpan) {
        float4 wVals0 = vload4(0, rowWeights + vecIndex);
        float4 xVals0 = vload4(0, input + vecIndex);
        acc0 = fma(wVals0, xVals0, acc0);

        float4 wVals1 = vload4(0, rowWeights + vecIndex + vecStride);
        float4 xVals1 = vload4(0, input + vecIndex + vecStride);
        acc1 = fma(wVals1, xVals1, acc1);

        float4 wVals2 = vload4(0, rowWeights + vecIndex + (vecStride << 1));
        float4 xVals2 = vload4(0, input + vecIndex + (vecStride << 1));
        acc2 = fma(wVals2, xVals2, acc2);

        float4 wVals3 = vload4(0, rowWeights + vecIndex + (vecStride * 3));
        float4 xVals3 = vload4(0, input + vecIndex + (vecStride * 3));
        acc3 = fma(wVals3, xVals3, acc3);
    }

    float4 partial4 = acc0 + acc1 + acc2 + acc3;

    for (; vecIndex < vecSpan; vecIndex += vecStride) {
        float4 wVals = vload4(0, rowWeights + vecIndex);
        float4 xVals = vload4(0, input + vecIndex);
        partial4 = fma(wVals, xVals, partial4);
    }

    float partial = partial4.x + partial4.y + partial4.z + partial4.w;

    // Handle leftover elements when n is not divisible by 4.
    int tailIndex = vecSpan + localId;
    for (; tailIndex < n; tailIndex += localSize) {
        partial = fma(rowWeights[tailIndex], input[tailIndex], partial);
    }

    scratch[localId] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    reduce_group(scratch, localId, localSize);

    if (localId == 0) {
        output[rowId] = scratch[0];
    }
}

__kernel void matrixVectorParallel(__global long *_kernel_context,
                                   __constant uchar *_constant_region,
                                   __local uchar *_local_region,
                                   __global int *_atomics,
                                   __global uchar *x,
                                   __global uchar *hb,
                                   __global uchar *w,
                                   __private int n,
                                   __private int d) {
    (void)_kernel_context;
    (void)_constant_region;
    (void)_local_region;
    (void)_atomics;

    __global const float *restrict input = get_ro_float_ptr(x);
    __global float *restrict output = get_rw_float_ptr(hb);
    __global const float *restrict weights = get_ro_float_ptr(w);

    const int globalId = get_global_id(0);
    const int globalSize = get_global_size(0);

    for (int row = globalId; row < d; row += globalSize) {
        const __global float *rowWeights = weights + (row * n);
        int idx = 0;
        const int vecSpan = n & ~3;
        const int vecLimit = vecSpan - 16;

        float4 acc0 = (float4)(0.0f);
        float4 acc1 = (float4)(0.0f);
        float4 acc2 = (float4)(0.0f);
        float4 acc3 = (float4)(0.0f);

        for (; idx <= vecLimit; idx += 16) {
            float4 w0 = vload4(0, rowWeights + idx);
            float4 x0 = vload4(0, input + idx);
            acc0 = fma(w0, x0, acc0);

            float4 w1 = vload4(0, rowWeights + idx + 4);
            float4 x1 = vload4(0, input + idx + 4);
            acc1 = fma(w1, x1, acc1);

            float4 w2 = vload4(0, rowWeights + idx + 8);
            float4 x2 = vload4(0, input + idx + 8);
            acc2 = fma(w2, x2, acc2);

            float4 w3 = vload4(0, rowWeights + idx + 12);
            float4 x3 = vload4(0, input + idx + 12);
            acc3 = fma(w3, x3, acc3);
        }

        float4 sumVec = acc0 + acc1 + acc2 + acc3;

        for (; idx < vecSpan; idx += 4) {
            float4 wVals = vload4(0, rowWeights + idx);
            float4 xVals = vload4(0, input + idx);
            sumVec = fma(wVals, xVals, sumVec);
        }

        float sum = sumVec.x + sumVec.y + sumVec.z + sumVec.w;

        for (; idx < n; ++idx) {
            sum = fma(rowWeights[idx], input[idx], sum);
        }

        output[row] = sum;
    }
}
