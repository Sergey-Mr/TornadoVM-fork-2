#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
__kernel void mandelbrotTornado(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __private int size, __global uchar *output)
{
  ulong ul_0, ul_40; 
  float f_34, f_33, f_32, f_31, f_30, f_29, f_28, f_27, f_26, f_25, f_24, f_20, f_19, f_18, f_17, f_15, f_12, f_11, f_10, f_9, f_35; 
  short sh_44; 
  int i_16, i_21, i_22, i_23, i_1, i_2, i_3, i_4, i_36, i_5, i_37, i_6, i_7, i_8, i_41, i_42, i_43, i_13, i_45, i_14, i_46; 
  long l_39, l_38; 

  // BLOCK 0
  ul_0  =  (ulong) output;
  i_1  =  get_global_size(0);
  i_2  =  get_global_size(1);
  i_3  =  get_global_id(0);
  i_4  =  get_global_id(1);
  // BLOCK 1 MERGES [0 11 ]
  i_5  =  i_4;
  for(;i_5 < 1024;)
  {
    // BLOCK 2
    i_6  =  i_5 << 10;
    i_7  =  i_6 + 12;
    // BLOCK 3 MERGES [2 10 ]
    i_8  =  i_3;
    for(;i_8 < 1024;)
    {
      // BLOCK 4
      // BLOCK 5 MERGES [4 9 ]
      f_9  =  0.0F;
      f_10  =  0.0F;
      f_11  =  0.0F;
      f_12  =  0.0F;
      i_13  =  0;
      i_14  =  0;
      for(;i_14 < 10000;)
      {
        // BLOCK 6
        f_15  =  f_11 + f_12;
        i_16  =  isless(4.0F, f_15);
        if(i_16 == 1)
        {
          // BLOCK 7
          f_17  =  f_9;
          f_18  =  f_10;
          f_19  =  f_11;
          f_20  =  f_12;
          i_21  =  i_13;
          i_22  =  10000;
        }  // B7
        else
        {
          // BLOCK 8
          i_23  =  i_13 + 1;
          f_24  =  f_9 * 2.0F;
          f_25  =  (float) i_5;
          f_26  =  f_25 * 0.001953125F;
          f_27  =  f_26 - 1.0F;
          f_28  =  fma(f_10, f_24, f_27);
          f_29  =  f_28 * f_28;
          f_30  =  (float) i_8;
          f_31  =  f_30 * 0.001953125F;
          f_32  =  f_31 - 1.5F;
          f_33  =  f_11 - f_12;
          f_34  =  f_32 + f_33;
          f_35  =  f_34 * f_34;
          f_17  =  f_34;
          f_18  =  f_28;
          f_19  =  f_35;
          f_20  =  f_29;
          i_21  =  i_23;
          i_22  =  i_14;
        }  // B8
        // BLOCK 9 MERGES [8 7 ]
        i_36  =  i_22 + 1;
        f_9  =  f_17;
        f_10  =  f_18;
        f_11  =  f_19;
        f_12  =  f_20;
        i_13  =  i_21;
        i_14  =  i_36;
      }  // B9
      // BLOCK 10
      i_37  =  i_7 + i_8;
      l_38  =  (long) i_37;
      l_39  =  l_38 << 1;
      ul_40  =  ul_0 + l_39;
      i_41  =  i_13 << 8;
      i_42  =  i_41 - i_13;
      i_43  =  i_42 / 10000;
      sh_44  =  (short) i_43;
      *((__global short *) ul_40)  =  sh_44;
      i_45  =  i_1 + i_8;
      i_8  =  i_45;
    }  // B10
    // BLOCK 11
    i_46  =  i_2 + i_5;
    i_5  =  i_46;
  }  // B11
  // BLOCK 12
  return;
}  //  kernel

