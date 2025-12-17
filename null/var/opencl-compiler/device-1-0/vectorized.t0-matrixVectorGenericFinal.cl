#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  
__kernel void matrixVectorGenericFinal(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *x, __global uchar *output, __global uchar *weightsQ, __global uchar *weightScales, __private int dim1, __private int dim0, __private int localWorkGroupSize)
{
  ulong ul_50, ul_46, ul_59, ul_55, ul_2, ul_3, ul_0, ul_32, ul_64, ul_1, ul_97, ul_126, ul_28, ul_106, ul_41, ul_37, ul_101; 
  int i_90, i_25, i_89, i_92, i_91, i_30, i_94, i_93, i_20, i_83, i_22, i_21, i_24, i_23, i_87, i_10, i_9, i_12, i_11, i_75, i_14, i_13, i_15, i_79, i_67, i_5, i_8, i_7, i_71, i_57, i_123, i_61, i_114, i_113, i_52, i_117, i_120, i_43, i_109, i_48, i_34, i_99, i_39, i_103; 
  float f_88, f_86, f_116, f_112, f_110, f_47, f_111, f_108, f_107, f_38, f_68, f_69, f_66, f_65, f_56, f_121, f_118, f_119, f_84, f_85, f_18, f_82, f_19, f_16, f_80, f_17, f_81, f_78, f_76, f_77, f_74, f_72, f_73, f_70; 
  bool b_115, b_6, b_122; 
  half half_98, half_29; 
  char ch_102, ch_33, ch_51, ch_60, ch_42; 
  long l_26, l_58, l_27, l_53, l_54, l_96, l_35, l_124, l_125, l_62, l_31, l_63, l_95, l_40, l_104, l_105, l_36, l_100, l_49, l_44, l_45; 

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
