  #pragma OPENCL EXTENSION cl_khr_fp16 : enable

  // Tile size - 16x16 works well on Apple GPUs
  // Can try 8x8 if register pressure is an issue
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
      // TornadoVM float array header offset
      #define FLOAT_BASE_INDEX 4

      // Get typed pointers with header offset
      __global float * restrict matA = ((__global float *)A) + FLOAT_BASE_INDEX;
      __global float * restrict matB = ((__global float *)B) + FLOAT_BASE_INDEX;
      __global float * restrict matC = ((__global float *)C) + FLOAT_BASE_INDEX;

      // Local memory tiles
      __local float tileA[TS][TS];
      __local float tileB[TS][TS];

      // Thread indices
      const int lx = get_local_id(0);
      const int ly = get_local_id(1);
      const int gx = get_global_id(0);  // Column of C
      const int gy = get_global_id(1);  // Row of C

      // Accumulator
      float acc = 0.0f;

      // Number of tiles
      const int numTiles = (size + TS - 1) / TS;

      // Loop over tiles
      for (int t = 0; t < numTiles; t++) {
          // Tile starting column for A, starting row for B
          const int tileStart = t * TS;

          // Cooperative load: each thread loads one element into each tile
          // Load A[gy][tileStart + lx]
          const int aCol = tileStart + lx;
          if (gy < size && aCol < size) {
              tileA[ly][lx] = matA[gy * size + aCol];
          } else {
              tileA[ly][lx] = 0.0f;
          }

          // Load B[tileStart + ly][gx]
          const int bRow = tileStart + ly;
          if (bRow < size && gx < size) {
              tileB[ly][lx] = matB[bRow * size + gx];
          } else {
              tileB[ly][lx] = 0.0f;
          }

          // Synchronize to ensure tile is fully loaded
          barrier(CLK_LOCAL_MEM_FENCE);

          // Compute partial dot product from this tile
          #pragma unroll
          for (int k = 0; k < TS; k++) {
              acc = fma(tileA[ly][k], tileB[k][lx], acc);
          }

          // Synchronize before loading next tile
          barrier(CLK_LOCAL_MEM_FENCE);
      }

      // Write result
      if (gy < size && gx < size) {
          matC[gy * size + gx] = acc;
      }
  }


