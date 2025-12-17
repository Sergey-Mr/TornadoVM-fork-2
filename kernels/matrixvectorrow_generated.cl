#pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
  __kernel void matrixVectorGeneric(__global long *_kernel_context, 
  __constant uchar *_constant_region, __local uchar *_local_region, __global
   int *_atomics, __global uchar *x, __global uchar *hb, __global uchar *w, 
  __private int n, __private int d, __private int localWorkGroupSize)
  {
    ulong ul_14, ul_0, ul_1, ul_2, ul_19, ul_35; 
    float f_20, f_30, f_15, f_9, f_25, f_27, f_28, f_21; 
    long l_18, l_34, l_17, l_33, l_12, l_13; 
    bool b_24, b_5, b_31; 
    int i_4, i_7, i_23, i_8, i_6, i_22, i_11, i_10, i_26, i_16, i_32, i_29; 

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
      i_7  =  i_6 + 4;
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
        i_16  =  i_10 + 4;
        l_17  =  (long) i_16;
        l_18  =  l_17 << 2;
        ul_19  =  ul_0 + l_18;
        f_20  =  *((__global float *) ul_19);
        f_21  =  fma(f_15, f_20, f_9);
        i_22  =  i_10 + 32;
        f_9  =  f_21;
        i_10  =  i_22;
      }  // B3
      // BLOCK 4
      adf_3[i_8]  =  f_9;
      barrier(CLK_LOCAL_MEM_FENCE);
      // BLOCK 5 MERGES [4 9 ]
      i_23  =  16;
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
        i_32  =  i_4 + 4;
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

  #pragma OPENCL EXTENSION cl_khr_fp64 : enable  
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable  
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  
  __kernel void matrixVectorParallel(__global long *_kernel_context, 
  __constant uchar *_constant_region, __local uchar *_local_region, __global
   int *_atomics, __global uchar *x, __global uchar *hb, __global uchar *w, 
  __private int n, __private int d)
  {
    ulong ul_13, ul_25, ul_0, ul_1, ul_2, ul_18; 
    float f_19, f_20, f_14, f_8; 
    long l_16, l_17, l_11, l_12, l_23, l_24; 
    int i_3, i_4, i_7, i_5, i_21, i_6, i_22, i_9, i_10, i_26, i_15; 

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

  #pragma OPENCL EXTENSION cl_khr_fp64 : enable  
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable  
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  
  __kernel void matrixVectorGenericFP16(__global long *_kernel_context, 
  __constant uchar *_constant_region, __local uchar *_local_region, __global
   int *_atomics, __global uchar *x, __global uchar *hb, __global uchar *w, 
  __private int n, __private int d, __private int localWorkGroupSize)
  {
    ulong ul_14, ul_36, ul_0, ul_1, ul_2, ul_19; 
    float f_20, f_29, f_31, f_9, f_26, f_28, f_21, f_22; 
    half half_15; 
    long l_18, l_34, l_35, l_17, l_12, l_13; 
    bool b_25, b_5, b_32; 
    int i_4, i_33, i_7, i_23, i_8, i_24, i_6, i_11, i_27, i_10, i_16, i_30; 

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

  #pragma OPENCL EXTENSION cl_khr_fp64 : enable  
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable  
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  
  __kernel void matrixVectorGenericFinal(__global long *_kernel_context, 
  __constant uchar *_constant_region, __local uchar *_local_region, __global
   int *_atomics, __global uchar *x, __global uchar *output, __global uchar 
  *weightsQ, __global uchar *weightScales, __private int dim1, __private int
   dim0, __private int localWorkGroupSize)
  {
    char ch_102, ch_42, ch_60, ch_51, ch_33; 
    ulong ul_28, ul_126, ul_59, ul_55, ul_50, ul_46, ul_41, ul_106, ul_37, 
  ul_101, ul_0, ul_32, ul_64, ul_1, ul_97, ul_2, ul_3; 
    float f_65, f_66, f_68, f_73, f_74, f_76, f_69, f_70, f_72, f_116, 
  f_110, f_47, f_111, f_112, f_121, f_118, f_119, f_56, f_107, f_108, f_38, 
  f_17, f_81, f_18, f_82, f_19, f_84, f_77, f_78, f_16, f_80, f_85, f_86, 
  f_88; 
    half half_29, half_98; 
    long l_53, l_49, l_26, l_58, l_27, l_124, l_125, l_54, l_35, l_36, 
  l_100, l_62, l_31, l_63, l_95, l_96, l_44, l_45, l_40, l_104, l_105; 
    bool b_122, b_115, b_6; 
    int i_52, i_113, i_114, i_120, i_117, i_43, i_48, i_109, i_67, i_7, 
  i_71, i_8, i_5, i_123, i_57, i_61, i_83, i_20, i_23, i_87, i_24, i_21, 
  i_22, i_11, i_75, i_12, i_9, i_10, i_15, i_79, i_13, i_14, i_99, i_34, 
  i_39, i_103, i_91, i_92, i_25, i_89, i_90, i_93, i_30, i_94; 

    // BLOCK 0
    ul_0  =  (ulong) x;
    ul_1  =  (ulong) output;
    ul_2  =  (ulong) weightsQ;
    ul_3  =  (ulong) weightScales;
    __local float adf_4[32];
    i_5  =  get_group_id(0);
    b_6  =  i_5 < 2048;
    if(b_6)
    {
      // BLOCK 1
      i_7  =  i_5 << 13;
      i_8  =  i_7 + 19;
      i_9  =  i_7 + 18;
      i_10  =  i_7 + 17;
      i_11  =  i_7 + 16;
      i_12  =  i_5 << 8;
      i_13  =  i_12 + 8;
      i_14  =  get_local_id(0);
      i_15  =  i_14 << 2;
      // BLOCK 2 MERGES [1 3 ]
      f_16  =  0.0F;
      f_17  =  0.0F;
      f_18  =  0.0F;
      f_19  =  0.0F;
      i_20  =  i_15;
      for(;i_20 < 8189;)
      {
        // BLOCK 3
        i_21  =  i_20 >> 31;
        i_22  =  i_21 >> 27;
        i_23  =  i_22 + i_20;
        i_24  =  i_23 >> 5;
        i_25  =  i_24 + i_13;
        l_26  =  (long) i_25;
        l_27  =  l_26 << 1;
        ul_28  =  ul_3 + l_27;
        half_29  =  *((__global half *) ul_28);
        i_30  =  i_11 + i_20;
        l_31  =  (long) i_30;
        ul_32  =  ul_2 + l_31;
        ch_33  =  *((__global char *) ul_32);
        i_34  =  i_20 + 4;
        l_35  =  (long) i_34;
        l_36  =  l_35 << 2;
        ul_37  =  ul_0 + l_36;
        f_38  =  *((__global float *) ul_37);
        i_39  =  i_10 + i_20;
        l_40  =  (long) i_39;
        ul_41  =  ul_2 + l_40;
        ch_42  =  *((__global char *) ul_41);
        i_43  =  i_20 + 5;
        l_44  =  (long) i_43;
        l_45  =  l_44 << 2;
        ul_46  =  ul_0 + l_45;
        f_47  =  *((__global float *) ul_46);
        i_48  =  i_9 + i_20;
        l_49  =  (long) i_48;
        ul_50  =  ul_2 + l_49;
        ch_51  =  *((__global char *) ul_50);
        i_52  =  i_20 + 6;
        l_53  =  (long) i_52;
        l_54  =  l_53 << 2;
        ul_55  =  ul_0 + l_54;
        f_56  =  *((__global float *) ul_55);
        i_57  =  i_8 + i_20;
        l_58  =  (long) i_57;
        ul_59  =  ul_2 + l_58;
        ch_60  =  *((__global char *) ul_59);
        i_61  =  i_20 + 7;
        l_62  =  (long) i_61;
        l_63  =  l_62 << 2;
        ul_64  =  ul_0 + l_63;
        f_65  =  *((__global float *) ul_64);
        f_66  =  convert_float((float) half_29);
        i_67  =  (int) ch_60;
        f_68  =  (float) i_67;
        f_69  =  f_66 * f_68;
        f_70  =  fma(f_69, f_65, f_19);
        i_71  =  (int) ch_51;
        f_72  =  (float) i_71;
        f_73  =  f_72 * f_66;
        f_74  =  fma(f_73, f_56, f_18);
        i_75  =  (int) ch_42;
        f_76  =  (float) i_75;
        f_77  =  f_76 * f_66;
        f_78  =  fma(f_77, f_47, f_17);
        i_79  =  (int) ch_33;
        f_80  =  (float) i_79;
        f_81  =  f_80 * f_66;
        f_82  =  fma(f_81, f_38, f_16);
        i_83  =  i_20 + 128;
        f_16  =  f_82;
        f_17  =  f_78;
        f_18  =  f_74;
        f_19  =  f_70;
        i_20  =  i_83;
      }  // B3
      // BLOCK 4
      f_84  =  f_16 + f_17;
      f_85  =  f_84 + f_18;
      f_86  =  f_85 + f_19;
      i_87  =  i_14 + 8192;
      // BLOCK 5 MERGES [4 6 ]
      f_88  =  f_86;
      i_89  =  i_87;
      for(;i_89 < 8192;)
      {
        // BLOCK 6
        i_90  =  i_89 >> 31;
        i_91  =  i_90 >> 27;
        i_92  =  i_91 + i_89;
        i_93  =  i_92 >> 5;
        i_94  =  i_93 + i_13;
        l_95  =  (long) i_94;
        l_96  =  l_95 << 1;
        ul_97  =  ul_3 + l_96;
        half_98  =  *((__global half *) ul_97);
        i_99  =  i_89 + i_11;
        l_100  =  (long) i_99;
        ul_101  =  ul_2 + l_100;
        ch_102  =  *((__global char *) ul_101);
        i_103  =  i_89 + 4;
        l_104  =  (long) i_103;
        l_105  =  l_104 << 2;
        ul_106  =  ul_0 + l_105;
        f_107  =  *((__global float *) ul_106);
        f_108  =  convert_float((float) half_98);
        i_109  =  (int) ch_102;
        f_110  =  (float) i_109;
        f_111  =  f_108 * f_110;
        f_112  =  fma(f_111, f_107, f_88);
        i_113  =  i_89 + 32;
        f_88  =  f_112;
        i_89  =  i_113;
      }  // B6
      // BLOCK 7
      adf_4[i_14]  =  f_88;
      barrier(CLK_LOCAL_MEM_FENCE);
      // BLOCK 8 MERGES [7 12 ]
      i_114  =  16;
      for(;i_114 >= 1;)
      {
        // BLOCK 9
        b_115  =  i_14 < i_114;
        if(b_115)
        {
          // BLOCK 10
          f_116  =  adf_4[i_14];
          i_117  =  i_114 + i_14;
          f_118  =  adf_4[i_117];
          f_119  =  f_116 + f_118;
          adf_4[i_14]  =  f_119;
        }  // B10
        else
        {
          // BLOCK 11
        }  // B11
        // BLOCK 12 MERGES [11 10 ]
        barrier(CLK_LOCAL_MEM_FENCE);
        i_120  =  i_114 >> 1;
        i_114  =  i_120;
      }  // B12
      // BLOCK 13
      f_121  =  adf_4[0];
      b_122  =  i_14 == 0;
      if(b_122)
      {
        // BLOCK 14
        i_123  =  i_5 + 4;
        l_124  =  (long) i_123;
        l_125  =  l_124 << 2;
        ul_126  =  ul_1 + l_125;
        *((__global float *) ul_126)  =  f_121;
      }  // B14
      else
      {
        // BLOCK 15
      }  // B15
      // BLOCK 16 MERGES [15 14 ]
      return;
    }  // B1
    else
    {
      // BLOCK 17
      return;
    }  // B17
  }  //  kernel
