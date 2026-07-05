#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// Optimization 4: Local memory tiling (NO unrolling inside tile loop)
// - Uses shared memory to reduce global memory traffic by ~16x
// - Cooperative coalesced loads into local memory
// - This isolates the tiling benefit from unrolling

#define TS 16

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
  const int N = (int)_kernel_context[0];
  
  __global const float *a = ((__global const float *)A) + 4;
  __global const float *b = ((__global const float *)B) + 4;
  __global float *c = ((__global float *)C) + 4;
  
  // Global position
  const int col0 = get_global_id(0);
  const int row0 = get_global_id(1);
  
  // Stride for grid-stride loop
  const int colStep = get_global_size(0);
  const int rowStep = get_global_size(1);
  
  // Local position within workgroup
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  
  // Local memory tiles
  __local float As[TS][TS];
  __local float Bs[TS][TS];
  
  for (int row = row0; row < N; row += rowStep) {
    for (int col = col0; col < N; col += colStep) {
      float acc = 0.0f;
      
      // Tile over K dimension
      for (int k0 = 0; k0 < N; k0 += TS) {
        // Cooperative coalesced loads into local memory
        const int aCol = k0 + lx;
        const int bRow = k0 + ly;
        
        // Load tile of A (row-major, coalesced)
        As[ly][lx] = (aCol < N) ? a[row * N + aCol] : 0.0f;
        
        // Load tile of B (row-major, coalesced)
        Bs[ly][lx] = (bRow < N) ? b[bRow * N + col] : 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial dot product - NO UNROLLING
        for (int k = 0; k < TS; k++) {
          acc = fma(As[ly][k], Bs[k][lx], acc);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      
      c[row * N + col] = acc;
    }
  }
}
