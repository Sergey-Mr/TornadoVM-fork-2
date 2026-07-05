#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// Optimization 3: Explicit work-group size only
// Allows compiler to optimize register allocation and scheduling
// 16x16 = 256 threads per workgroup (good for NVIDIA GPUs)

__attribute__((reqd_work_group_size(16, 16, 1)))
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
      for (int k = 0; k < N; k++) {
        float aVal = a[rowN + k];
        float bVal = b[k * N + col];
        acc = fma(aVal, bVal, acc);
      }
      c[rowN + col] = acc;
    }
  }
}
