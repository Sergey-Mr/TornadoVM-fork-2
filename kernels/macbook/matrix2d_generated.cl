#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
__kernel void matrixMultiplication(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *A, __global uchar *B, __global uchar *C, __private int size)
{
  ulong ul_2, ul_18, ul_1, ul_0, ul_31, ul_24; 
  float f_25, f_19, f_13, f_26; 
  int i_33, i_4, i_3, i_6, i_5, i_8, i_7, i_28, i_27, i_32, i_20, i_21, i_10, i_9, i_12, i_11, i_14, i_15; 
  long l_29, l_30, l_23, l_22, l_16, l_17; 

  // BLOCK 0
  ul_0  =  (ulong) A;
  ul_1  =  (ulong) B;
  ul_2  =  (ulong) C;
  i_3  =  get_global_size(0);
  i_4  =  get_global_size(1);
  i_5  =  get_global_id(0);
  i_6  =  get_global_id(1);
  i_7  =  _kernel_context[0];
  // BLOCK 1 MERGES [0 8 ]
  i_8  =  i_6;
  for(;i_8 < i_7;)
  {
    // BLOCK 2
    i_9  =  i_8 * i_7;
    i_10  =  i_9 + 4;
    // BLOCK 3 MERGES [2 7 ]
    i_11  =  i_5;
    for(;i_11 < i_7;)
    {
      // BLOCK 4
      i_12  =  i_11 + 4;
      // BLOCK 5 MERGES [4 6 ]
      f_13  =  0.0F;
      i_14  =  0;
      for(;i_14 < i_7;)
      {
        // BLOCK 6
        i_15  =  i_10 + i_14;
        l_16  =  (long) i_15;
        l_17  =  l_16 << 2;
        ul_18  =  ul_0 + l_17;
        f_19  =  *((__global float *) ul_18);
        i_20  =  i_14 * i_7;
        i_21  =  i_20 + i_12;
        l_22  =  (long) i_21;
        l_23  =  l_22 << 2;
        ul_24  =  ul_1 + l_23;
        f_25  =  *((__global float *) ul_24);
        f_26  =  fma(f_19, f_25, f_13);
        i_27  =  i_14 + 1;
        f_13  =  f_26;
        i_14  =  i_27;
      }  // B6
      // BLOCK 7
      i_28  =  i_11 + i_10;
      l_29  =  (long) i_28;
      l_30  =  l_29 << 2;
      ul_31  =  ul_2 + l_30;
      *((__global float *) ul_31)  =  f_13;
      i_32  =  i_3 + i_11;
      i_11  =  i_32;
    }  // B7
    // BLOCK 8
    i_33  =  i_4 + i_8;
    i_8  =  i_33;
  }  // B8
  // BLOCK 9
  return;
}  //  kernel
