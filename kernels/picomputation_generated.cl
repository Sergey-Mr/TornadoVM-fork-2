#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
__kernel void computePi(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *input, __global uchar *result)
{
  float f_29, f_12, f_31, f_21, f_20, f_36, f_22, f_17, f_16, f_32; 
  ulong ul_42, ul_11, ul_3, ul_0, ul_1; 
  int i_8, i_7, i_6, i_38, i_5, i_37, i_4, i_34, i_33, i_30, i_27, i_26, i_25, i_24, i_23, i_19, i_18, i_15, i_14, i_13, i_43; 
  long l_40, l_39, l_10, l_9, l_41; 
  bool b_35, b_28; 

  // BLOCK 0
  ul_0  =  (ulong) input;
  ul_1  =  (ulong) result;
  __local float adf_2[256];
  ul_3  =  ul_1 + 24L;
  *((__global float *) ul_3)  =  0.0F;
  i_4  =  get_global_size(0);
  i_5  =  get_global_id(0);
  i_6  =  i_5 + 1;
  // BLOCK 1 MERGES [0 11 ]
  i_7  =  i_6;
  for(;i_7 < 8192;)
  {
    // BLOCK 2
    i_8  =  i_7 + 6;
    l_9  =  (long) i_8;
    l_10  =  l_9 << 2;
    ul_11  =  ul_0 + l_10;
    f_12  =  *((__global float *) ul_11);
    i_13  =  get_local_id(0);
    i_14  =  get_local_size(0);
    i_15  =  i_7 + 1;
    f_16  =  (float) i_15;
    f_17  =  pow(-1.0F, f_16);
    i_18  =  i_7 << 1;
    i_19  =  i_18 + -1;
    f_20  =  (float) i_19;
    f_21  =  f_17 / f_20;
    f_22  =  f_21 + f_12;
    adf_2[i_13]  =  f_22;
    i_23  =  i_14 >> 31;
    i_24  =  i_23 + i_14;
    i_25  =  i_24 >> 1;
    // BLOCK 3 MERGES [2 7 ]
    i_26  =  i_25;
    for(;i_26 >= 1;)
    {
      // BLOCK 4
      barrier(CLK_LOCAL_MEM_FENCE);
      i_27  =  i_26 >> 1;
      b_28  =  i_13 < i_26;
      if(b_28)
      {
        // BLOCK 5
        f_29  =  adf_2[i_13];
        i_30  =  i_26 + i_13;
        f_31  =  adf_2[i_30];
        f_32  =  f_29 + f_31;
        adf_2[i_13]  =  f_32;
      }  // B5
      else
      {
        // BLOCK 6
      }  // B6
      // BLOCK 7 MERGES [6 5 ]
      i_33  =  i_27;
      i_26  =  i_33;
    }  // B7
    // BLOCK 8
    barrier(CLK_GLOBAL_MEM_FENCE);
    i_34  =  i_4 + i_7;
    b_35  =  i_13 == 0;
    if(b_35)
    {
      // BLOCK 9
      f_36  =  adf_2[0];
      i_37  =  get_group_id(0);
      i_38  =  i_37 + 1;
      l_39  =  (long) i_38;
      l_40  =  l_39 << 2;
      l_41  =  l_40 + 24L;
      ul_42  =  ul_1 + l_41;
      *((__global float *) ul_42)  =  f_36;
    }  // B9
    else
    {
      // BLOCK 10
    }  // B10
    // BLOCK 11 MERGES [10 9 ]
    i_43  =  i_34;
    i_7  =  i_43;
  }  // B11
  // BLOCK 12
  return;
}  //  kernel

#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
__kernel void rAdd(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *array, __private int size)
{
  float f_44, f_46, f_40, f_42, f_52, f_54, f_48, f_50, f_28, f_30, f_24, f_26, f_36, f_38, f_32, f_34, f_12, f_14, f_8, f_10, f_20, f_22, f_16, f_18, f_4, f_6, f_2, f_93, f_92, f_95, f_94, f_89, f_88, f_91, f_90, f_97, f_96, f_98, f_77, f_76, f_79, f_78, f_73, f_72, f_75, f_74, f_85, f_84, f_87, f_86, f_81, f_80, f_83, f_82, f_60, f_62, f_56, f_58, f_69, f_68, f_71, f_70, f_64, f_67, f_66; 
  ulong ul_29, ul_27, ul_25, ul_23, ul_21, ul_19, ul_17, ul_15, ul_13, ul_11, ul_9, ul_7, ul_5, ul_3, ul_0, ul_1, ul_65, ul_63, ul_61, ul_59, ul_57, ul_55, ul_53, ul_51, ul_49, ul_47, ul_45, ul_43, ul_41, ul_39, ul_37, ul_35, ul_33, ul_31; 

  // BLOCK 0
  ul_0  =  (ulong) array;
  ul_1  =  ul_0 + 24L;
  f_2  =  *((__global float *) ul_1);
  ul_3  =  ul_0 + 28L;
  f_4  =  *((__global float *) ul_3);
  ul_5  =  ul_0 + 32L;
  f_6  =  *((__global float *) ul_5);
  ul_7  =  ul_0 + 36L;
  f_8  =  *((__global float *) ul_7);
  ul_9  =  ul_0 + 40L;
  f_10  =  *((__global float *) ul_9);
  ul_11  =  ul_0 + 44L;
  f_12  =  *((__global float *) ul_11);
  ul_13  =  ul_0 + 48L;
  f_14  =  *((__global float *) ul_13);
  ul_15  =  ul_0 + 52L;
  f_16  =  *((__global float *) ul_15);
  ul_17  =  ul_0 + 56L;
  f_18  =  *((__global float *) ul_17);
  ul_19  =  ul_0 + 60L;
  f_20  =  *((__global float *) ul_19);
  ul_21  =  ul_0 + 64L;
  f_22  =  *((__global float *) ul_21);
  ul_23  =  ul_0 + 68L;
  f_24  =  *((__global float *) ul_23);
  ul_25  =  ul_0 + 72L;
  f_26  =  *((__global float *) ul_25);
  ul_27  =  ul_0 + 76L;
  f_28  =  *((__global float *) ul_27);
  ul_29  =  ul_0 + 80L;
  f_30  =  *((__global float *) ul_29);
  ul_31  =  ul_0 + 84L;
  f_32  =  *((__global float *) ul_31);
  ul_33  =  ul_0 + 88L;
  f_34  =  *((__global float *) ul_33);
  ul_35  =  ul_0 + 92L;
  f_36  =  *((__global float *) ul_35);
  ul_37  =  ul_0 + 96L;
  f_38  =  *((__global float *) ul_37);
  ul_39  =  ul_0 + 100L;
  f_40  =  *((__global float *) ul_39);
  ul_41  =  ul_0 + 104L;
  f_42  =  *((__global float *) ul_41);
  ul_43  =  ul_0 + 108L;
  f_44  =  *((__global float *) ul_43);
  ul_45  =  ul_0 + 112L;
  f_46  =  *((__global float *) ul_45);
  ul_47  =  ul_0 + 116L;
  f_48  =  *((__global float *) ul_47);
  ul_49  =  ul_0 + 120L;
  f_50  =  *((__global float *) ul_49);
  ul_51  =  ul_0 + 124L;
  f_52  =  *((__global float *) ul_51);
  ul_53  =  ul_0 + 128L;
  f_54  =  *((__global float *) ul_53);
  ul_55  =  ul_0 + 132L;
  f_56  =  *((__global float *) ul_55);
  ul_57  =  ul_0 + 136L;
  f_58  =  *((__global float *) ul_57);
  ul_59  =  ul_0 + 140L;
  f_60  =  *((__global float *) ul_59);
  ul_61  =  ul_0 + 144L;
  f_62  =  *((__global float *) ul_61);
  ul_63  =  ul_0 + 148L;
  f_64  =  *((__global float *) ul_63);
  ul_65  =  ul_0 + 152L;
  f_66  =  *((__global float *) ul_65);
  f_67  =  f_2 + f_4;
  f_68  =  f_67 + f_6;
  f_69  =  f_68 + f_8;
  f_70  =  f_69 + f_10;
  f_71  =  f_70 + f_12;
  f_72  =  f_71 + f_14;
  f_73  =  f_72 + f_16;
  f_74  =  f_73 + f_18;
  f_75  =  f_74 + f_20;
  f_76  =  f_75 + f_22;
  f_77  =  f_76 + f_24;
  f_78  =  f_77 + f_26;
  f_79  =  f_78 + f_28;
  f_80  =  f_79 + f_30;
  f_81  =  f_80 + f_32;
  f_82  =  f_81 + f_34;
  f_83  =  f_82 + f_36;
  f_84  =  f_83 + f_38;
  f_85  =  f_84 + f_40;
  f_86  =  f_85 + f_42;
  f_87  =  f_86 + f_44;
  f_88  =  f_87 + f_46;
  f_89  =  f_88 + f_48;
  f_90  =  f_89 + f_50;
  f_91  =  f_90 + f_52;
  f_92  =  f_91 + f_54;
  f_93  =  f_92 + f_56;
  f_94  =  f_93 + f_58;
  f_95  =  f_94 + f_60;
  f_96  =  f_95 + f_62;
  f_97  =  f_96 + f_64;
  f_98  =  f_97 + f_66;
  *((__global float *) ul_1)  =  f_98;
  return;
}  //  kernel

