#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  

__kernel void matrixMultiplication(__global long *_kernel_context,
                                   __constant uchar *_constant_region,
                                   __local    uchar *_local_region,
                                   __global   int   *_atomics,
                                   __global   uchar *matrixA,
                                   __global   uchar *matrixB,
                                   __global   uchar *result,
                                   __private  int    size)
{
  // Reinterpret raw uchar* buffers as float* for arithmetic
  __global float *A = (__global float *) matrixA;
  __global float *B = (__global float *) matrixB;
  __global float *C = (__global float *) result;

  // We keep the hard-coded 512 used in the original kernel
  // (the 'size' argument was not used there either).
  const int N = 512;

  int global_size_x = get_global_size(0);
  int global_size_y = get_global_size(1);
  int gx            = get_global_id(0);
  int gy            = get_global_id(1);

  // Grid-stride loop over rows (same as i_7 loop)
  for (int row = gy; row < N; row += global_size_y)
  {
    // From original:
    //   i_8  = row << 9;      // row * 512
    //   i_9  = i_8 + 4;       // base index for this row in A and C
    int row_base = (row << 9) + 4;   // row * 512 + 4

    // Grid-stride loop over columns (same as i_10 loop)
    for (int col = gx; col < N; col += global_size_x)
    {
      // From original:
      //   i_11 = col + 4;      // used as column index into B
      int col_plus_4 = col + 4;

      float sum = 0.0f;

      // Inner product over k (same as i_13 loop, 0..511)
      for (int k = 0; k < N; ++k)
      {
        // Original A access:
        //   i_14 = i_9 + i_13;
        //   f_18 = *((float*)(matrixA + (i_14 << 2)));
        //
        // => A[row_base + k]
        float a = A[row_base + k];

        // Original B access:
        //   i_19 = i_13 << 9;          // k * 512
        //   i_20 = i_19 + i_11;        // k*512 + (col+4)
        //   f_24 = *((float*)(matrixB + (i_20 << 2)));
        //
        // => B[(k << 9) + col_plus_4]
        int  b_index = (k << 9) + col_plus_4;
        float b = B[b_index];

        // Same FMA pattern as original: fma(a, b, sum)
        sum = fma(a, b, sum);
      }

      // Original C write:
      //   i_27 = col + i_9;           // row_base + col
      //   *((float*)(result + (i_27 << 2))) = f_12;
      //
      // => C[row_base + col] = sum
      int c_index = row_base + col;
      C[c_index] = sum;
    }
  }
}
