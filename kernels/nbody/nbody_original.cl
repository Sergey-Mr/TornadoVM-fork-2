#pragma OPENCL EXTENSION cl_khr_fp64 : enable                                                     
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable                                                     
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable                                       
  __kernel void nBody(__global long *_kernel_context, __constant uchar *_constant_region,           
  __local uchar *_local_region, __global int *_atomics, __private int numBodies, __global uchar     
  *refPos, __global uchar *refVel, __private float delT, __private float espSqr)                    
  {                                                                                                 
  long l_27, l_26, l_34, l_33, l_12, l_11, l_40, l_41, l_20, l_19, l_16, l_48, l_15, l_47;          
  ulong ul_21, ul_69, ul_42, ul_90, ul_28, ul_13, ul_79, ul_0, ul_1, ul_17, ul_49, ul_35;           
  double d_56, d_57;                                                                                
  int i_9, i_10, i_7, i_39, i_8, i_6, i_67, i_100, i_32, i_25, i_24, i_22, i_18, i_14, i_46;        
  float f_85, f_84, f_87, f_86, f_81, f_80, f_83, f_82, f_93, f_92, f_95, f_94, f_89, f_88,         
  f_91, f_97, f_96, f_99, f_98, f_53, f_52, f_55, f_54, f_51, f_50, f_61, f_60, f_63, f_62,         
  f_59, f_58, f_68, f_71, f_70, f_65, f_64, f_66, f_77, f_76, f_78, f_73, f_72, f_75, f_74,         
  f_23, f_29, f_31, f_30, f_37, f_36, f_38, f_45, f_44, f_43;                                       
                                                                                                    
  // BLOCK 0                                                                                        
  ul_0  =  (ulong) refPos;                                                                          
  ul_1  =  (ulong) refVel;                                                                          
  __private float ul_2[3];                                                                          
  __private float* ul_3 = ul_2;                                                                     
  __private float ul_4[3];                                                                          
  __private float* ul_5 = ul_4;                                                                     
  i_6  =  get_global_size(0);                                                                       
  i_7  =  get_global_id(0);                                                                         
  // BLOCK 1 MERGES [0 5 ]                                                                          
  i_8  =  i_7;                                                                                      
  for(;i_8 < 2048;)                                                                                 
  {                                                                                                 
  // BLOCK 2                                                                                        
  ul_3[0]  =  0.0F;                                                                                 
  ul_3[1]  =  0.0F;                                                                                 
  ul_3[2]  =  0.0F;                                                                                 
  i_9  =  i_8 << 2;                                                                                 
  i_10  =  i_9 + 6;                                                                                 
  l_11  =  (long) i_10;                                                                             
  l_12  =  l_11 << 2;                                                                               
  ul_13  =  ul_0 + l_12;                                                                            
  i_14  =  i_9 + 5;                                                                                 
  l_15  =  (long) i_14;                                                                             
  l_16  =  l_15 << 2;                                                                               
  ul_17  =  ul_0 + l_16;                                                                            
  i_18  =  i_9 + 4;                                                                                 
  l_19  =  (long) i_18;                                                                             
  l_20  =  l_19 << 2;                                                                               
  ul_21  =  ul_0 + l_20;                                                                            
  // BLOCK 3 MERGES [2 4 ]                                                                          
  i_22  =  0;                                                                                       
  f_23  =  0.0F;                                                                                    
  for(;i_22 < 2048;)                                                                                
  {                                                                                                 
  // BLOCK 4                                                                                        
  i_24  =  i_22 << 2;                                                                               
  i_25  =  i_24 + 4;                                                                                
  l_26  =  (long) i_25;                                                                             
  l_27  =  l_26 << 2;                                                                               
  ul_28  =  ul_0 + l_27;                                                                            
  f_29  =  *((__global float *) ul_28);                                                             
  f_30  =  *((__global float *) ul_21);                                                             
  f_31  =  f_29 - f_30;                                                                             
  ul_5[0]  =  f_31;                                                                                 
  i_32  =  i_24 + 5;                                                                                
  l_33  =  (long) i_32;                                                                             
  l_34  =  l_33 << 2;                                                                               
  ul_35  =  ul_0 + l_34;                                                                            
  f_36  =  *((__global float *) ul_35);                                                             
  f_37  =  *((__global float *) ul_17);                                                             
  f_38  =  f_36 - f_37;                                                                             
  ul_5[1]  =  f_38;                                                                                 
  i_39  =  i_24 + 6;                                                                                
  l_40  =  (long) i_39;                                                                             
  l_41  =  l_40 << 2;                                                                               
  ul_42  =  ul_0 + l_41;                                                                            
  f_43  =  *((__global float *) ul_42);                                                             
  f_44  =  *((__global float *) ul_13);                                                             
  f_45  =  f_43 - f_44;                                                                             
  ul_5[2]  =  f_45;                                                                                 
  i_46  =  i_24 + 7;                                                                                
  l_47  =  (long) i_46;                                                                             
  l_48  =  l_47 << 2;                                                                               
  ul_49  =  ul_0 + l_48;                                                                            
  f_50  =  *((__global float *) ul_49);                                                             
  f_51  =  ul_3[0];                                                                                 
  f_52  =  fma(f_31, f_31, 0.0F);                                                                   
  f_53  =  fma(f_38, f_38, f_52);                                                                   
  f_54  =  fma(f_45, f_45, f_53);                                                                   
  f_55  =  f_54 + 500.0F;                                                                           
  d_56  =  (double) f_55;                                                                           
  d_57  =  rsqrt(d_56);                                                                             
  f_58  =  (float) d_57;                                                                            
  f_59  =  f_58 * f_58;                                                                             
  f_60  =  f_59 * f_58;                                                                             
  f_61  =  f_60 * f_50;                                                                             
  f_62  =  fma(f_61, f_31, f_51);                                                                   
  ul_3[0]  =  f_62;                                                                                 
  f_63  =  ul_3[1];                                                                                 
  f_64  =  fma(f_61, f_38, f_63);                                                                   
  ul_3[1]  =  f_64;                                                                                 
  f_65  =  ul_3[2];                                                                                 
  f_66  =  fma(f_61, f_45, f_65);                                                                   
  ul_3[2]  =  f_66;                                                                                 
  i_67  =  i_22 + 1;                                                                                
  i_22  =  i_67;                                                                                    
  f_23  =  f_62;                                                                                    
  }  // B4                                                                                          
  // BLOCK 5                                                                                        
  f_68  =  *((__global float *) ul_21);                                                             
  ul_69  =  ul_1 + l_20;                                                                            
  f_70  =  *((__global float *) ul_69);                                                             
  f_71  =  f_23 * 0.5F;                                                                             
  f_72  =  f_71 * 0.005F;                                                                           
  f_73  =  fma(f_70, 0.005F, f_68);                                                                 
  f_74  =  fma(f_72, 0.005F, f_73);                                                                 
  *((__global float *) ul_21)  =  f_74;                                                             
  f_75  =  *((__global float *) ul_69);                                                             
  f_76  =  ul_3[0];                                                                                 
  f_77  =  fma(f_76, 0.005F, f_75);                                                                 
  *((__global float *) ul_69)  =  f_77;                                                             
  f_78  =  *((__global float *) ul_17);                                                             
  ul_79  =  ul_1 + l_16;                                                                            
  f_80  =  *((__global float *) ul_79);                                                             
  f_81  =  ul_3[1];                                                                                 
  f_82  =  f_81 * 0.5F;                                                                             
  f_83  =  f_82 * 0.005F;                                                                           
  f_84  =  fma(f_80, 0.005F, f_78);                                                                 
  f_85  =  fma(f_83, 0.005F, f_84);                                                                 
  *((__global float *) ul_17)  =  f_85;                                                             
  f_86  =  *((__global float *) ul_79);                                                             
  f_87  =  ul_3[1];                                                                                 
  f_88  =  fma(f_87, 0.005F, f_86);                                                                 
  *((__global float *) ul_79)  =  f_88;                                                             
  f_89  =  *((__global float *) ul_13);                                                             
  ul_90  =  ul_1 + l_12;                                                                            
  f_91  =  *((__global float *) ul_90);                                                             
  f_92  =  ul_3[2];                                                                                 
  f_93  =  f_92 * 0.5F;                                                                             
  f_94  =  f_93 * 0.005F;                                                                           
  f_95  =  fma(f_91, 0.005F, f_89);                                                                 
  f_96  =  fma(f_94, 0.005F, f_95);                                                                 
  *((__global float *) ul_13)  =  f_96;                                                             
  f_97  =  *((__global float *) ul_90);                                                             
  f_98  =  ul_3[2];                                                                                 
  f_99  =  fma(f_98, 0.005F, f_97);                                                                 
  *((__global float *) ul_90)  =  f_99;                                                             
  i_100  =  i_6 + i_8;                                                                              
  i_8  =  i_100;                                                                                    
  }  // B5                                                                                          
  // BLOCK 6                                                                                        
  return;                                                                                           
  }  //  kernel

