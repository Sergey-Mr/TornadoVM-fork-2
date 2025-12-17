#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  

__kernel void computeMatrixVector(__global long *_kernel_context,
                                  __constant uchar *_constant_region,
                                  __local    uchar *_local_region,
                                  __global   int   *_atomics,
                                  __global   uchar *matrix,
                                  __global   uchar *vector,
                                  __global   uchar *output)
{
  // Reinterpret raw uchar* buffers as typed pointers
  __global uint  *matrix_u = (__global uint  *) matrix;
  __global uint  *vector_u = (__global uint  *) vector;
  __global uint  *output_u = (__global uint  *) output;

  __global float *matrix_f = (__global float *) matrix;
  __global float *vector_f = (__global float *) vector;
  __global float *output_f = (__global float *) output;

  // Extract the same metadata offsets as original code:
  //
  // original:
  //   ui_14 = *((__global uint *) (matrix + 24));
  //   ui_22 = *((__global uint *) (vector + 16));
  //   ui_32 = *((__global uint *) (output + 16));
  //
  // Using uint* indexing: 24/4 = 6, 16/4 = 4.
  uint ui_14 = matrix_u[6];  // matrix + 24 bytes
  uint ui_22 = vector_u[4];  // vector + 16 bytes
  uint ui_32 = output_u[4];  // output + 16 bytes

  // Convert those byte offsets (in units of 8 bytes) into float indices.
  //
  // original base address for floats: buf + (ui_* << 3) bytes.
  // (ui_ * 8 bytes) / 4 bytes per float = ui_*2 float elements.
  __global float *matrix_base = matrix_f + ((int)ui_14 << 1);   // == (float*)(matrix + (ui_14<<3))
  __global float *vector_base = vector_f + ((int)ui_22 << 1);   // == (float*)(vector + (ui_22<<3))
  __global float *output_base = output_f + ((int)ui_32 << 1);   // == (float*)(output + (ui_32<<3))

  int global_size = get_global_size(0);
  int gid         = get_global_id(0);

  // Outer loop over rows: grid-stride, same as original i_8 loop.
  for (int row = gid; row < 8192; row += global_size)
  {
    // original:
    //   i_9 = row << 13;  // row * 8192
    //   i_10 = i_9 + 4;   // base index for this row
    int row_base = (row << 13) + 4;

    float sum = 0.0f;

    // Inner loop over columns: SAME bounds & indexing as original.
    //
    // original per-iteration:
    //   i_16 = i_10 + i_12;
    //   f_20 = *(float*)(matrix + (ui_14<<3) + (i_16<<2));
    //   i_24 = i_12 + 4;
    //   f_28 = *(float*)(vector + (ui_22<<3) + (i_24<<2));
    //
    // which is equivalent to:
    //   f_20 = matrix_base[row_base + col];
    //   f_28 = vector_base[col + 4];
    for (int col = 0; col < 8192; ++col)
    {
      float m = matrix_base[row_base + col];
      float v = vector_base[col + 4];
      sum = fma(m, v, sum);   // same FMA order as original
    }

    // original:
    //   i_34 = row + 4;
    //   *(float*)(output + (ui_32<<3) + (i_34<<2)) = sum;
    //
    // which is equivalent to:
    //   output_base[row + 4] = sum;
    output_base[row + 4] = sum;
  }
}
