#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  

__kernel void processHeadsFlashAttention(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *q, __global uchar *key_cache, __global uchar *value_cache, __global uchar *xb, __private int nHeads, __private int headSize, __private int kvDim, __private int kvMul, __global uchar *positionHolder, __private int layer, __private int contextLength)
{
  ulong ul_2, ul_1, ul_49, ul_113, ul_4, ul_52, ul_3, ul_14, ul_0, ul_24; 
  long l_47, l_111, l_23, l_22, l_48, l_112; 
  float f_64, f_66, f_69, f_71, f_74, f_75, f_78, f_79, f_81, f_83, f_84, f_86, f_87, f_25, f_89, f_90, f_27, f_92, f_93, f_94, f_97, f_34, f_35, f_99, f_100, f_102, f_106, f_107, f_109, f_50, f_114, f_53, f_59, f_61, f_63; 
  int i_55, i_56, i_54, i_51, i_62, i_60, i_57, i_58, i_72, i_70, i_67, i_68, i_65, i_80, i_76, i_73, i_88, i_85, i_82, i_95, i_96, i_91, i_103, i_104, i_101, i_98, i_110, i_108, i_105, i_115, i_15, i_16, i_12, i_21, i_19, i_20, i_17, i_18, i_31, i_32, i_29, i_30, i_28, i_26, i_39, i_40, i_37, i_38, i_36, i_33, i_45, i_46, i_43, i_44, i_41, i_42; 
  bool b_13, b_77; 

  // BLOCK 0
  ul_0  =  (ulong) q;
  ul_1  =  (ulong) key_cache;
  ul_2  =  (ulong) value_cache;
  ul_3  =  (ulong) xb;
  ul_4  =  (ulong) positionHolder;
  __private float ul_5[128];
  __private float* ul_6 = ul_5;
  __local float adf_7[128];
  __local float adf_8[2048];
  __local float adf_9[2048];
  __local float adf_10[16];
  __local float adf_11[1];
  i_12  =  get_group_id(0);
  b_13  =  i_12 < 32;
  if(b_13)
  {
    // BLOCK 1
    ul_14  =  ul_4 + 16L;
    i_15  =  *((__global int *) ul_14);
    
    // OPTIMIZATION: Replace 128 individual assignments with loop
    for (int _init = 0; _init < 128; _init++) {
        ul_6[_init] = 0.0F;
    }
    
    i_16  =  get_local_size(0);
    i_17  =  i_12 << 7;
    i_18  =  i_17 + 4;
    i_19  =  get_local_id(0);
    // BLOCK 2 MERGES [1 3 ]
    i_20  =  i_19;
    for(;i_20 < 128;)
    {
      // BLOCK 3
      i_21  =  i_18 + i_20;
      l_22  =  (long) i_21;
      l_23  =  l_22 << 2;
      ul_24  =  ul_0 + l_23;
      f_25  =  *((__global float *) ul_24);
      adf_7[i_20]  =  f_25;
      i_26  =  i_16 + i_20;
      i_20  =  i_26;
    }  // B3
    // BLOCK 4
    barrier(CLK_LOCAL_MEM_FENCE);
    f_27  =  -INFINITY;  // OPTIMIZATION: Use -INFINITY instead of -1.0F / 0.0F
    i_28  =  i_12 >> 31;
    i_29  =  i_28 >> 30;
    i_30  =  i_29 + i_12;
    i_31  =  i_30 >> 2;
    i_32  =  i_31 << 7;
    i_33  =  i_32 + 4;
    // BLOCK 5 MERGES [4 41 ]
    f_34  =  f_27;
    f_35  =  0.0F;
    i_36  =  0;
    for(;i_15 >= i_36;)
    {
      // BLOCK 6
      i_37  =  i_36 + 15;
      i_38  =  min(i_37, i_15);
      i_39  =  i_36 + i_19;
      // BLOCK 7 MERGES [6 11 ]
      i_40  =  i_39;
      for(;i_38 >= i_40;)
      {
        // BLOCK 8
        i_41  =  i_40 - i_36;
        i_42  =  i_41 << 7;
        i_43  =  i_40 << 10;
        i_44  =  i_43 + i_33;
        
        // OPTIMIZATION: Vectorized K/V loading with float4
        for (i_45 = 0; i_45 < 128; i_45 += 4) {
            i_46  =  i_44 + i_45;
            l_47  =  (long) i_46;
            l_48  =  l_47 << 2;
            ul_49  =  ul_1 + l_48;
            
            float4 k_vec = vload4(0, (__global float *) ul_49);
            float4 v_vec = vload4(0, (__global float *) (ul_2 + l_48));
            
            i_51  =  i_42 + i_45;
            adf_8[i_51]     = k_vec.x;
            adf_8[i_51 + 1] = k_vec.y;
            adf_8[i_51 + 2] = k_vec.z;
            adf_8[i_51 + 3] = k_vec.w;
            
            adf_9[i_51]     = v_vec.x;
            adf_9[i_51 + 1] = v_vec.y;
            adf_9[i_51 + 2] = v_vec.z;
            adf_9[i_51 + 3] = v_vec.w;
        }
        
        // BLOCK 11
        i_55  =  i_40 + i_16;
        i_40  =  i_55;
      }  // B11
      // BLOCK 12
      barrier(CLK_LOCAL_MEM_FENCE);
      // BLOCK 13 MERGES [12 17 ]
      i_56  =  i_39;
      for(;i_38 >= i_56;)
      {
        // BLOCK 14
        i_57  =  i_56 - i_36;
        i_58  =  i_57 << 7;
        
        // OPTIMIZATION: Vectorized dot product
        float4 sum4 = (float4)(0.0f);
        for (i_60 = 0; i_60 < 128; i_60 += 4) {
            float4 q_vec = (float4)(adf_7[i_60], adf_7[i_60+1], adf_7[i_60+2], adf_7[i_60+3]);
            i_62  =  i_58 + i_60;
            float4 k_vec = (float4)(adf_8[i_62], adf_8[i_62+1], adf_8[i_62+2], adf_8[i_62+3]);
            sum4 = fma(q_vec, k_vec, sum4);
        }
        f_59 = sum4.x + sum4.y + sum4.z + sum4.w;
        
        // BLOCK 17
        f_66  =  f_59 * 0.0883883476F;  // OPTIMIZATION: multiply by 1/sqrt(128) instead of divide
        adf_10[i_57]  =  f_66;
        i_67  =  i_56 + i_16;
        i_56  =  i_67;
      }  // B17
      // BLOCK 18
      barrier(CLK_LOCAL_MEM_FENCE);
      i_68  =  i_38 - i_36;
      // BLOCK 19 MERGES [18 23 ]
      f_69  =  f_27;
      i_70  =  0;
      for(;i_68 >= i_70;)
      {
        // BLOCK 20
        f_71  =  adf_10[i_70];
        i_72  =  i_70 + 1;
        f_75  =  fmax(f_69, f_71);  // OPTIMIZATION: use fmax instead of isless branch
        // BLOCK 23 MERGES [22 21 ]
        i_76  =  i_72;
        f_69  =  f_75;
        i_70  =  i_76;
      }  // B23
      // BLOCK 24
      b_77  =  i_19 == 0;
      if(b_77)
      {
        // BLOCK 25
        adf_11[0]  =  f_69;
      }  // B25
      // BLOCK 27 MERGES [26 25 ]
      barrier(CLK_LOCAL_MEM_FENCE);
      f_78  =  adf_11[0];
      f_79  =  fmax(f_34, f_78);
      i_80  =  isequal(f_34, f_79);
      if(i_80 == 1)
      {
        // BLOCK 28
        f_81  =  f_35;
      }  // B28
      else
      {
        // BLOCK 29
        i_82  =  isequal(f_34, f_27);
        if(i_82 == 1)
        {
          // BLOCK 30
          f_81  =  f_35;
        }  // B30
        else
        {
          // BLOCK 31
          f_83  =  f_34 - f_79;
          f_84  =  native_exp(f_83);  // OPTIMIZATION: native_exp
          
          // OPTIMIZATION: Unrolled rescale loop
          for (i_85 = 0; i_85 < 128; i_85 += 4) {
              ul_6[i_85]     *= f_84;
              ul_6[i_85 + 1] *= f_84;
              ul_6[i_85 + 2] *= f_84;
              ul_6[i_85 + 3] *= f_84;
          }
          
          // BLOCK 34
          f_89  =  f_35 * f_84;
          f_81  =  f_89;
        }  // B31
      }  // B29
      // BLOCK 35 MERGES [28 30 34 ]
      f_90  =  f_81;
      // BLOCK 36 MERGES [35 40 ]
      i_91  =  0;
      for(;i_68 >= i_91;)
      {
        // BLOCK 37
        f_92  =  adf_10[i_91];
        f_93  =  f_92 - f_79;
        f_94  =  native_exp(f_93);  // OPTIMIZATION: native_exp
        i_95  =  i_91 << 7;
        
        // OPTIMIZATION: Unrolled output accumulation
        for (i_96 = 0; i_96 < 128; i_96 += 4) {
            i_98  =  i_95 + i_96;
            ul_6[i_96]     = fma(f_94, adf_9[i_98],     ul_6[i_96]);
            ul_6[i_96 + 1] = fma(f_94, adf_9[i_98 + 1], ul_6[i_96 + 1]);
            ul_6[i_96 + 2] = fma(f_94, adf_9[i_98 + 2], ul_6[i_96 + 2]);
            ul_6[i_96 + 3] = fma(f_94, adf_9[i_98 + 3], ul_6[i_96 + 3]);
        }
        
        // BLOCK 40
        f_102  =  f_90 + f_94;
        i_103  =  i_91 + 1;
        f_90  =  f_102;
        i_91  =  i_103;
      }  // B40
      // BLOCK 41
      barrier(CLK_LOCAL_MEM_FENCE);
      i_104  =  i_36 + 16;
      f_34  =  f_79;
      f_35  =  f_90;
      i_36  =  i_104;
    }  // B41
    // BLOCK 42
    i_105  =  isless(0.0F, f_35);
    if(i_105 == 1)
    {
      // BLOCK 43
      f_106  =  1.0F / f_35;
      f_107  =  f_106;
    }  // B43
    else
    {
      // BLOCK 44
      f_107  =  0.0F;
    }  // B44
    // BLOCK 45 MERGES [43 44 ]
    // BLOCK 46 MERGES [45 47 ]
    i_108  =  i_19;
    for(;i_108 < 128;)
    {
      // BLOCK 47
      f_109  =  ul_6[i_108];
      i_110  =  i_108 + i_18;
      l_111  =  (long) i_110;
      l_112  =  l_111 << 2;
      ul_113  =  ul_3 + l_112;
      f_114  =  f_107 * f_109;
      *((__global float *) ul_113)  =  f_114;
      i_115  =  i_108 + i_16;
      i_108  =  i_115;
    }  // B47
    // BLOCK 48
    return;
  }  // B1
  else
  {
    // BLOCK 49
    return;
  }  // B49
}
