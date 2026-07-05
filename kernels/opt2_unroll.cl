#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// Optimization 2: Loop unrolling only (4x unroll on k-loop)
// Reduces loop overhead and enables better instruction pipelining

__kernel void matrixMultiplication(__global long *_kernel_context, 
                                   __constant uchar *_constant_region, 
                                   __local uchar *_local_region, 
                                   __global int *_atomics, 
                                   __global uchar *A, 
                                   __global uchar *B, 
                                   __global uchar *C, 
                                   __private int size)
{
  const int N = (int)_kernel_context[0];
  const int gx = get_global_size(0);
  const int gy = get_global_size(1);
  const int idx = get_global_id(0);
  const int idy = get_global_id(1);
  
  __global const float *a = ((__global const float *)A) + 4;
  __global const float *b = ((__global const float *)B) + 4;
  __global float *c = ((__global float *)C) + 4;
  
  for (int row = idy; row < N; row += gy) {
    const int rowN = row * N;
    for (int col = idx; col < N; col += gx) {
      float acc = 0.0f;
      
      // Main loop: 4x unrolled
      const int kEnd = N - (N % 4);
      #pragma unroll 4
      for (int k = 0; k < kEnd; k += 4) {
        acc = fma(a[rowN + k + 0], b[(k + 0) * N + col], acc);
        acc = fma(a[rowN + k + 1], b[(k + 1) * N + col], acc);
        acc = fma(a[rowN + k + 2], b[(k + 2) * N + col], acc);
        acc = fma(a[rowN + k + 3], b[(k + 3) * N + col], acc);
      }
      
      // Remainder loop (handles N not divisible by 4)
      for (int k = kEnd; k < N; k++) {
        acc = fma(a[rowN + k], b[k * N + col], acc);
      }
      
      c[rowN + col] = acc;
    }
  }
}
