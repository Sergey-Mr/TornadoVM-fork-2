#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void matrixVectorGeneric(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *x, __global uchar *hb, __global uchar *w, __private int n, __private int d, __private int localWorkGroupSize)
{
  long l_18, l_34, l_17, l_33, l_13, l_12;
  ulong ul_19, ul_35, ul_0, ul_2, ul_1, ul_14;
  bool b_24, b_31, b_5;
  int i_6, i_22, i_4, i_16, i_32, i_29, i_11, i_10, i_26, i_7, i_23, i_8;
  float f_20, f_21, f_28, f_27, f_9, f_25, f_15, f_30;

  // BLOCK 0
  ul_0  =  (ulong) x;
  ul_1  =  (ulong) hb;
  ul_2  =  (ulong) w;
  __local float adf_3[128];
  i_4  =  get_group_id(0);
  b_5  =  i_4 < 2048;
  if(b_5)
  {
    // BLOCK 1
    i_6  =  i_4 << 13;
    i_7  =  i_6 + 6;
    i_8  =  get_local_id(0);
    // BLOCK 2 MERGES [1 3 ]
    f_9  =  0.0F;
    i_10  =  i_8;
    for(;i_10 < 8192;)
    {
      // BLOCK 3
      i_11  =  i_7 + i_10;
      l_12  =  (long) i_11;
      l_13  =  l_12 << 2;
      ul_14  =  ul_2 + l_13;
      f_15  =  *((__global float *) ul_14);
      i_16  =  i_10 + 6;
      l_17  =  (long) i_16;
      l_18  =  l_17 << 2;
      ul_19  =  ul_0 + l_18;
      f_20  =  *((__global float *) ul_19);
      f_21  =  fma(f_15, f_20, f_9);
      i_22  =  i_10 + 128;
      f_9  =  f_21;
      i_10  =  i_22;
    }  // B3
    // BLOCK 4
    adf_3[i_8]  =  f_9;
    barrier(CLK_LOCAL_MEM_FENCE);
    // BLOCK 5 MERGES [4 9 ]
    i_23  =  64;
    for(;i_23 >= 1;)
    {
      // BLOCK 6
      b_24  =  i_8 < i_23;
      if(b_24)
      {
        // BLOCK 7
        f_25  =  adf_3[i_8];
        i_26  =  i_23 + i_8;
        f_27  =  adf_3[i_26];
        f_28  =  f_25 + f_27;
        adf_3[i_8]  =  f_28;
      }  // B7
      else
      {
        // BLOCK 8
      }  // B8
      // BLOCK 9 MERGES [8 7 ]
      barrier(CLK_LOCAL_MEM_FENCE);
      i_29  =  i_23 >> 1;
      i_23  =  i_29;
    }  // B9
    // BLOCK 10
    f_30  =  adf_3[0];
    b_31  =  i_8 == 0;
    if(b_31)
    {
      // BLOCK 11
      i_32  =  i_4 + 6;
      l_33  =  (long) i_32;
      l_34  =  l_33 << 2;
      ul_35  =  ul_1 + l_34;
      *((__global float *) ul_35)  =  f_30;
    }  // B11
    else
    {
      // BLOCK 12
    }  // B12
    // BLOCK 13 MERGES [12 11 ]
    return;
  }  // B1
  else
  {
    // BLOCK 14
    return;
  }  // B14
}  //  kernel

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void matrixVectorParallel(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *x, __global uchar *hb, __global uchar *w, __private int n, __private int d)
{
  long l_11, l_24, l_23, l_17, l_16, l_12;
  ulong ul_0, ul_2, ul_18, ul_1, ul_13, ul_25;
  int i_5, i_21, i_6, i_22, i_3, i_4, i_15, i_9, i_10, i_26, i_7;
  float f_20, f_19, f_8, f_14;

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
    i_7  =  i_6 + 6;
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
      i_15  =  i_9 + 6;
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
    i_22  =  i_5 + 6;
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

