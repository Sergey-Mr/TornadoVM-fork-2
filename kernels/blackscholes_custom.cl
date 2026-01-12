#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  

__kernel void blackScholesKernel(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *input, __global uchar *callResult, __global uchar *putResult)
{
  float f_15, f_16, f_13, f_14, f_11, f_12, f_10, f_21, f_19, f_20, f_17, f_18, f_107, f_105, f_106, f_95, f_96, f_93, f_94, f_91, f_92, f_89, f_90, f_103, f_104, f_101, f_102, f_99, f_98, f_79, f_80, f_77, f_78, f_75, f_76, f_73, f_74, f_87, f_88, f_85, f_86, f_83, f_81, f_82, f_64, f_61, f_62, f_59, f_60, f_57, f_58, f_71, f_72, f_69, f_70, f_67, f_68, f_65, f_47, f_48, f_45, f_46, f_44, f_41, f_42, f_55, f_56, f_53, f_54, f_51, f_52, f_49, f_50, f_31, f_32, f_29, f_30, f_27, f_28, f_25, f_26, f_39, f_40, f_37, f_38, f_35, f_36, f_33, f_34; 
  ulong ul_100, ul_1, ul_2, ul_66, ul_0, ul_9; 
  int i_97, i_3, i_108, i_63, i_43, i_4, i_84, i_5, i_6; 
  long l_7, l_8; 

  // BLOCK 0
  ul_0  =  (ulong) input;
  ul_1  =  (ulong) callResult;
  ul_2  =  (ulong) putResult;
  i_3  =  get_global_size(0);
  i_4  =  get_global_id(0);
  // BLOCK 1 MERGES [0 14 ]
  i_5  =  i_4;
  for(;i_5 < 1024;)
  {
    // BLOCK 2
    i_6  =  i_5 + 4;
    l_7  =  (long) i_6;
    l_8  =  l_7 << 2;
    ul_9  =  ul_0 + l_8;
    f_10  =  *((__global float *) ul_9);
    f_11  =  1.0F - f_10;
    f_12  =  fma(f_11, 10.0F, f_10);
    f_13  =  f_11 * 0.05F;
    f_14  =  fma(f_10, 0.01F, f_13);
    f_15  =  f_11 * 0.1F;
    f_16  =  fma(f_10, 0.01F, f_15);
    f_17  =  f_16 * f_16;
    f_18  =  f_17 * 0.5F;
    f_19  =  f_14 + f_18;
    f_20  =  f_11 * 100.0F;
    f_21  =  fma(f_10, 10.0F, f_20);
    // REMOVED: log(1.0) = 0 dead code
    f_25  =  (f_12 * f_19) / f_16;
    f_26  =  native_sqrt(f_12);
    f_27  =  f_25 * f_26;
    f_28  =  fabs(f_27);
    f_29  =  fma(f_28, 0.2316419F, 1.0F);
    f_30  =  1.0F / f_29;
    f_31  =  -f_27;
    f_32  =  f_31 * f_27;
    f_33  =  f_32 * 0.5F;
    f_34  =  exp(f_33);
    f_35  =  f_34 * 0.3989423F;
    f_36  =  f_30 * f_35;
    f_37  =  fma(f_30, 1.3302745F, -1.8212559F);
    f_38  =  fma(f_30, f_37, 1.7814779F);
    f_39  =  fma(f_30, f_38, -0.35656378F);
    f_40  =  fma(f_30, f_39, 0.31938154F);
    f_41  =  f_36 * f_40;
    f_42  =  1.0F - f_41;
    i_43  =  isless(f_27, 0.0F);
    if(i_43 == 1)
    {
      // BLOCK 3
      f_44  =  1.0F - f_42;
      f_45  =  f_44;
    }  // B3
    else
    {
      // BLOCK 4
      f_45  =  f_42;
    }  // B4
    // BLOCK 5 MERGES [3 4 ]
    f_46  =  f_16 * f_26;
    f_47  =  f_27 - f_46;
    f_48  =  fabs(f_47);
    f_49  =  fma(f_48, 0.2316419F, 1.0F);
    f_50  =  1.0F / f_49;
    f_51  =  -f_47;
    f_52  =  f_51 * f_47;
    f_53  =  f_52 * 0.5F;
    f_54  =  exp(f_53);
    f_55  =  f_54 * 0.3989423F;
    f_56  =  f_50 * f_55;
    f_57  =  fma(f_50, 1.3302745F, -1.8212559F);
    f_58  =  fma(f_50, f_57, 1.7814779F);
    f_59  =  fma(f_50, f_58, -0.35656378F);
    f_60  =  fma(f_50, f_59, 0.31938154F);
    f_61  =  f_56 * f_60;
    f_62  =  1.0F - f_61;
    i_63  =  isless(f_47, 0.0F);
    if(i_63 == 1)
    {
      // BLOCK 6
      f_64  =  1.0F - f_62;
      f_65  =  f_64;
    }  // B6
    else
    {
      // BLOCK 7
      f_65  =  f_62;
    }  // B7
    // BLOCK 8 MERGES [6 7 ]
    ul_66  =  ul_1 + l_8;
    f_67  =  f_45 * f_21;
    f_68  =  f_12 * -1.0F;
    f_69  =  f_68 * f_14;
    f_70  =  exp(f_69);
    f_71  =  f_70 * f_21;
    f_72  =  f_71 * f_65;
    f_73  =  f_67 - f_72;
    *((__global float *) ul_66)  =  f_73;
    f_74  =  fabs(f_51);
    f_75  =  fma(f_74, 0.2316419F, 1.0F);
    f_76  =  1.0F / f_75;
    f_77  =  f_76 * f_55;
    f_78  =  fma(f_76, 1.3302745F, -1.8212559F);
    f_79  =  fma(f_76, f_78, 1.7814779F);
    f_80  =  fma(f_76, f_79, -0.35656378F);
    f_81  =  fma(f_76, f_80, 0.31938154F);
    f_82  =  f_77 * f_81;
    f_83  =  1.0F - f_82;
    i_84  =  isless(f_51, 0.0F);
    if(i_84 == 1)
    {
      // BLOCK 9
      f_85  =  1.0F - f_83;
      f_86  =  f_85;
    }  // B9
    else
    {
      // BLOCK 10
      f_86  =  f_83;
    }  // B10
    // BLOCK 11 MERGES [9 10 ]
    f_87  =  fabs(f_31);
    f_88  =  fma(f_87, 0.2316419F, 1.0F);
    f_89  =  1.0F / f_88;
    f_90  =  f_89 * f_35;
    f_91  =  fma(f_89, 1.3302745F, -1.8212559F);
    f_92  =  fma(f_89, f_91, 1.7814779F);
    f_93  =  fma(f_89, f_92, -0.35656378F);
    f_94  =  fma(f_89, f_93, 0.31938154F);
    f_95  =  f_90 * f_94;
    f_96  =  1.0F - f_95;
    i_97  =  isless(f_31, 0.0F);
    if(i_97 == 1)
    {
      // BLOCK 12
      f_98  =  1.0F - f_96;
      f_99  =  f_98;
    }  // B12
    else
    {
      // BLOCK 13
      f_99  =  f_96;
    }  // B13
    // BLOCK 14 MERGES [12 13 ]
    ul_100  =  ul_2 + l_8;
    // OPTIMIZATION: Reuse f_70 instead of recomputing exp(-r*T)
    f_104  =  f_70 * f_21;
    f_105  =  f_104 * f_86;
    f_106  =  f_99 * f_21;
    f_107  =  f_105 - f_106;
    *((__global float *) ul_100)  =  f_107;
    i_108  =  i_3 + i_5;
    i_5  =  i_108;
  }  // B14
  // BLOCK 15
  return;
}
