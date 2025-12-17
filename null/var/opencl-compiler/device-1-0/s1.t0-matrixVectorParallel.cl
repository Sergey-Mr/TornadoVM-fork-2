#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  
__kernel void matrixVectorParallel(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *x, __global uchar *hb, __global uchar *w, __private int n, __private int d)
{
  ulong ul_2, ul_18, ul_0, ul_1, ul_13, ul_25; 
  int i_10, i_26, i_9, i_15, i_4, i_3, i_6, i_22, i_5, i_21, i_7; 
  float f_20, f_19, f_14, f_8; 
  long l_24, l_11, l_23, l_16, l_17, l_12; 

  // BLOCK 0
  ul_0  =  (ulong) x;
  ul_1  =  (ulong) hb;
  ul_2  =  (ulong) w;
  i_3  =  get_global_size(0);
  i_4  =  get_global_id(0);
  // BLOCK 1 MERGES [0 5 ]
  i_5  =  i_4;
  for(;i_5 < 2048;)
  {
    // BLOCK 2
    i_6  =  i_5 << 13;
    i_7  =  i_6 + 4;
    // BLOCK 3 MERGES [2 4 ]
    f_8  =  0.0F;
    i_9  =  0;
    for(;i_9 < 8192;)
    {
      // BLOCK 4
      i_10  =  i_7 + i_9;
      l_11  =  (long) i_10;
      l_12  =  l_11 << 2;
      ul_13  =  ul_2 + l_12;
      f_14  =  *((__global float *) ul_13);
      i_15  =  i_9 + 4;
      l_16  =  (long) i_15;
      l_17  =  l_16 << 2;
      ul_18  =  ul_0 + l_17;
      f_19  =  *((__global float *) ul_18);
      f_20  =  fma(f_14, f_19, f_8);
      i_21  =  i_9 + 1;
      f_8  =  f_20;
      i_9  =  i_21;
    }  // B4
    // BLOCK 5
    i_22  =  i_5 + 4;
    l_23  =  (long) i_22;
    l_24  =  l_23 << 2;
    ul_25  =  ul_1 + l_24;
    *((__global float *) ul_25)  =  f_8;
    i_26  =  i_3 + i_5;
    i_5  =  i_26;
  }  // B5
  // BLOCK 6
  return;
}  //  kernel
