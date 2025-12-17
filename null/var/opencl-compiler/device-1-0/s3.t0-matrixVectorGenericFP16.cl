#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  
__kernel void matrixVectorGenericFP16(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *x, __global uchar *hb, __global uchar *w, __private int n, __private int d, __private int localWorkGroupSize)
{
  ulong ul_2, ul_19, ul_0, ul_1, ul_14, ul_36; 
  int i_10, i_11, i_27, i_30, i_16, i_33, i_4, i_6, i_8, i_24, i_7, i_23; 
  float f_20, f_21, f_31, f_28, f_29, f_26, f_9, f_22; 
  bool b_32, b_5, b_25; 
  half half_15; 
  long l_17, l_18, l_34, l_35, l_12, l_13; 

  // BLOCK 0
  ul_0  =  (ulong) x;
  ul_1  =  (ulong) hb;
  ul_2  =  (ulong) w;
  __local float adf_3[32];
  i_4  =  get_group_id(0);
  b_5  =  i_4 < 2048;
  if(b_5)
  {
    // BLOCK 1
    i_6  =  i_4 << 13;
    i_7  =  i_6 + 8;
    i_8  =  get_local_id(0);
    // BLOCK 2 MERGES [1 3 ]
    f_9  =  0.0F;
    i_10  =  i_8;
    for(;i_10 < 8192;)
    {
      // BLOCK 3
      i_11  =  i_7 + i_10;
      l_12  =  (long) i_11;
      l_13  =  l_12 << 1;
      ul_14  =  ul_2 + l_13;
      half_15  =  *((__global half *) ul_14);
      i_16  =  i_10 + 4;
      l_17  =  (long) i_16;
      l_18  =  l_17 << 2;
      ul_19  =  ul_0 + l_18;
      f_20  =  *((__global float *) ul_19);
      f_21  =  convert_float((float) half_15);
      f_22  =  fma(f_21, f_20, f_9);
      i_23  =  i_10 + 32;
      f_9  =  f_22;
      i_10  =  i_23;
    }  // B3
    // BLOCK 4
    adf_3[i_8]  =  f_9;
    barrier(CLK_LOCAL_MEM_FENCE);
    // BLOCK 5 MERGES [4 9 ]
    i_24  =  16;
    for(;i_24 >= 1;)
    {
      // BLOCK 6
      b_25  =  i_8 < i_24;
      if(b_25)
      {
        // BLOCK 7
        f_26  =  adf_3[i_8];
        i_27  =  i_24 + i_8;
        f_28  =  adf_3[i_27];
        f_29  =  f_26 + f_28;
        adf_3[i_8]  =  f_29;
      }  // B7
      else
      {
        // BLOCK 8
      }  // B8
      // BLOCK 9 MERGES [8 7 ]
      barrier(CLK_LOCAL_MEM_FENCE);
      i_30  =  i_24 >> 1;
      i_24  =  i_30;
    }  // B9
    // BLOCK 10
    f_31  =  adf_3[0];
    b_32  =  i_8 == 0;
    if(b_32)
    {
      // BLOCK 11
      i_33  =  i_4 + 4;
      l_34  =  (long) i_33;
      l_35  =  l_34 << 2;
      ul_36  =  ul_1 + l_35;
      *((__global float *) ul_36)  =  f_31;
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
