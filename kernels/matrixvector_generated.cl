#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  
__kernel void computeMatrixVector(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *matrix, __global uchar *vector, __global uchar *output)
{
  long l_26, l_25, l_36, l_35, l_18, l_17; 
  float f_11, f_28, f_29, f_20; 
  uint ui_22, ui_14, ui_32; 
  ulong ul_13, ul_15, ul_19, ul_21, ul_23, ul_27, ul_31, ul_0, ul_1, ul_33, ul_2, ul_4, ul_5, ul_37, ul_6; 
  int i_8, i_24, i_7, i_10, i_9, i_12, i_30, i_16, i_34, i_3, i_38; 

  // BLOCK 0
  ul_0  =  (ulong) matrix;
  ul_1  =  (ulong) vector;
  ul_2  =  (ulong) output;
  i_3  =  get_global_size(0);
  ul_4  =  ul_2 + 16L;
  ul_5  =  ul_1 + 16L;
  ul_6  =  ul_0 + 24L;
  i_7  =  get_global_id(0);
  // BLOCK 1 MERGES [0 5 ]
  i_8  =  i_7;
  for(;i_8 < 8192;)
  {
    // BLOCK 2
    i_9  =  i_8 << 13;
    i_10  =  i_9 + 4;
    // BLOCK 3 MERGES [2 4 ]
    f_11  =  0.0F;
    i_12  =  0;
    for(;i_12 < 8192;)
    {
      // BLOCK 4
      ul_13  =  ul_0 + 24L;
      ui_14  =  *((__global uint *) ul_13);
      ul_15  =  ul_0 + ((ulong) ui_14 << 3);
      i_16  =  i_10 + i_12;
      l_17  =  (long) i_16;
      l_18  =  l_17 << 2;
      ul_19  =  ul_15 + l_18;
      f_20  =  *((__global float *) ul_19);
      ul_21  =  ul_1 + 16L;
      ui_22  =  *((__global uint *) ul_21);
      ul_23  =  ul_1 + ((ulong) ui_22 << 3);
      i_24  =  i_12 + 4;
      l_25  =  (long) i_24;
      l_26  =  l_25 << 2;
      ul_27  =  ul_23 + l_26;
      f_28  =  *((__global float *) ul_27);
      f_29  =  fma(f_20, f_28, f_11);
      i_30  =  i_12 + 1;
      f_11  =  f_29;
      i_12  =  i_30;
    }  // B4
    // BLOCK 5
    ul_31  =  ul_2 + 16L;
    ui_32  =  *((__global uint *) ul_31);
    ul_33  =  ul_2 + ((ulong) ui_32 << 3);
    i_34  =  i_8 + 4;
    l_35  =  (long) i_34;
    l_36  =  l_35 << 2;
    ul_37  =  ul_33 + l_36;
    *((__global float *) ul_37)  =  f_11;
    i_38  =  i_3 + i_8;
    i_8  =  i_38;
  }  // B5
  // BLOCK 6
  return;
}  //  kernel

