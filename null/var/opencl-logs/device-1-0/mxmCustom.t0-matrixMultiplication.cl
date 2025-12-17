#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  
__kernel void matrixMultiplicationCustom(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *A, __global uchar *B, __global uchar *C, __private int size)
{
  long l_21, l_22, l_40, l_41, l_30, l_31; 
  int i_20, i_14, i_13, i_16, i_28, i_4, i_3, i_35, i_29, i_10, i_9, i_12, i_44, i_11, i_43, i_8, i_39; 
  uint ui_26, ui_37, ui_18; 
  ulong ul_1, ul_0, ul_32, ul_2, ul_5, ul_36, ul_7, ul_6, ul_38, ul_42, ul_17, ul_19, ul_23, ul_25, ul_27; 
  float f_24, f_34, f_15, f_33; 

  // BLOCK 0
  ul_0  =  (ulong) A;
  ul_1  =  (ulong) B;
  ul_2  =  (ulong) C;
  i_3  =  get_global_size(0);
  i_4  =  get_global_size(1);
  ul_5  =  ul_2 + 24L;
  ul_6  =  ul_1 + 24L;
  ul_7  =  ul_0 + 24L;
  i_8  =  get_global_id(0);
  i_9  =  get_global_id(1);
  // BLOCK 1 MERGES [0 8 ]
  i_10  =  i_9;
  for(;i_10 < 512;)
  {
    // BLOCK 2
    i_11  =  i_10 << 9;
    i_12  =  i_11 + 4;
    // BLOCK 3 MERGES [2 7 ]
    i_13  =  i_8;
    for(;i_13 < 512;)
    {
      // BLOCK 4
      i_14  =  i_13 + 4;
      // BLOCK 5 MERGES [4 6 ]
      f_15  =  0.0F;
      i_16  =  0;
      for(;i_16 < 512;)
      {
        // BLOCK 6
        ul_17  =  ul_0 + 24L;
        ui_18  =  *((__global uint *) ul_17);
        ul_19  =  ul_0 + ((ulong) ui_18 << 3);
        i_20  =  i_12 + i_16;
        l_21  =  (long) i_20;
        l_22  =  l_21 << 2;
        ul_23  =  ul_19 + l_22;
        f_24  =  *((__global float *) ul_23);
        ul_25  =  ul_1 + 24L;
        ui_26  =  *((__global uint *) ul_25);
        ul_27  =  ul_1 + ((ulong) ui_26 << 3);
        i_28  =  i_16 << 9;
        i_29  =  i_28 + i_14;
        l_30  =  (long) i_29;
        l_31  =  l_30 << 2;
        ul_32  =  ul_27 + l_31;
        f_33  =  *((__global float *) ul_32);
        f_34  =  fma(f_24, f_33, f_15);
        i_35  =  i_16 + 1;
        f_15  =  f_34;
        i_16  =  i_35;
      }  // B6
      // BLOCK 7
      ul_36  =  ul_2 + 24L;
      ui_37  =  *((__global uint *) ul_36);
      ul_38  =  ul_2 + ((ulong) ui_37 << 3);
      i_39  =  i_13 + i_12;
      l_40  =  (long) i_39;
      l_41  =  l_40 << 2;
      ul_42  =  ul_38 + l_41;
      *((__global float *) ul_42)  =  f_15;
      i_43  =  i_3 + i_13;
      i_13  =  i_43;
    }  // B7
    // BLOCK 8
    i_44  =  i_4 + i_10;
    i_10  =  i_44;
  }  // B8
  // BLOCK 9
  return;
}  //  kernel
