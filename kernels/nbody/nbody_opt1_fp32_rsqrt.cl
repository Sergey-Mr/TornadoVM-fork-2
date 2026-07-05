#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// ============================================================================
// Optimization 1: FP32 rsqrt ONLY
// Changes double-precision rsqrt to single-precision
// Everything else matches generated kernel structure
// ============================================================================

__kernel void nBody(__global long *_kernel_context, 
                    __constant uchar *_constant_region, 
                    __local uchar *_local_region, 
                    __global int *_atomics, 
                    __private int numBodies, 
                    __global uchar *refPos, 
                    __global uchar *refVel, 
                    __private float delT, 
                    __private float espSqr)
{
    ulong ul_0 = (ulong) refPos;
    ulong ul_1 = (ulong) refVel;
    
    __private float ul_2[3];
    __private float* ul_3 = ul_2;
    __private float ul_4[3];
    __private float* ul_5 = ul_4;
    
    int i_6 = get_global_size(0);
    int i_7 = get_global_id(0);
    
    for (int i_8 = i_7; i_8 < 2048; i_8 += i_6) {
        ul_3[0] = 0.0f;
        ul_3[1] = 0.0f;
        ul_3[2] = 0.0f;
        
        int i_9 = i_8 << 2;
        int i_10 = i_9 + 6;
        long l_12 = (long)i_10 << 2;
        ulong ul_13 = ul_0 + l_12;
        
        int i_14 = i_9 + 5;
        long l_16 = (long)i_14 << 2;
        ulong ul_17 = ul_0 + l_16;
        
        int i_18 = i_9 + 4;
        long l_20 = (long)i_18 << 2;
        ulong ul_21 = ul_0 + l_20;
        
        float f_23 = 0.0f;
        for (int i_22 = 0; i_22 < 2048; i_22++) {
            int i_24 = i_22 << 2;
            int i_25 = i_24 + 4;
            long l_27 = (long)i_25 << 2;
            ulong ul_28 = ul_0 + l_27;
            float f_29 = *((__global float *) ul_28);
            float f_30 = *((__global float *) ul_21);
            float f_31 = f_29 - f_30;
            ul_5[0] = f_31;
            
            int i_32 = i_24 + 5;
            long l_34 = (long)i_32 << 2;
            ulong ul_35 = ul_0 + l_34;
            float f_36 = *((__global float *) ul_35);
            float f_37 = *((__global float *) ul_17);
            float f_38 = f_36 - f_37;
            ul_5[1] = f_38;
            
            int i_39 = i_24 + 6;
            long l_41 = (long)i_39 << 2;
            ulong ul_42 = ul_0 + l_41;
            float f_43 = *((__global float *) ul_42);
            float f_44 = *((__global float *) ul_13);
            float f_45 = f_43 - f_44;
            ul_5[2] = f_45;
            
            int i_46 = i_24 + 7;
            long l_48 = (long)i_46 << 2;
            ulong ul_49 = ul_0 + l_48;
            float f_50 = *((__global float *) ul_49);
            
            float f_51 = ul_3[0];
            float f_52 = fma(f_31, f_31, 0.0f);
            float f_53 = fma(f_38, f_38, f_52);
            float f_54 = fma(f_45, f_45, f_53);
            float f_55 = f_54 + 500.0f;
            
            // OPTIMIZATION: Single-precision rsqrt instead of double
            float f_58 = rsqrt(f_55);
            
            float f_59 = f_58 * f_58;
            float f_60 = f_59 * f_58;
            float f_61 = f_60 * f_50;
            float f_62 = fma(f_61, f_31, f_51);
            ul_3[0] = f_62;
            
            float f_63 = ul_3[1];
            float f_64 = fma(f_61, f_38, f_63);
            ul_3[1] = f_64;
            
            float f_65 = ul_3[2];
            float f_66 = fma(f_61, f_45, f_65);
            ul_3[2] = f_66;
            
            f_23 = f_62;
        }
        
        // Position and velocity updates (unchanged)
        float f_68 = *((__global float *) ul_21);
        ulong ul_69 = ul_1 + l_20;
        float f_70 = *((__global float *) ul_69);
        float f_71 = f_23 * 0.5f;
        float f_72 = f_71 * 0.005f;
        float f_73 = fma(f_70, 0.005f, f_68);
        float f_74 = fma(f_72, 0.005f, f_73);
        *((__global float *) ul_21) = f_74;
        
        float f_75 = *((__global float *) ul_69);
        float f_76 = ul_3[0];
        float f_77 = fma(f_76, 0.005f, f_75);
        *((__global float *) ul_69) = f_77;
        
        float f_78 = *((__global float *) ul_17);
        ulong ul_79 = ul_1 + l_16;
        float f_80 = *((__global float *) ul_79);
        float f_81 = ul_3[1];
        float f_82 = f_81 * 0.5f;
        float f_83 = f_82 * 0.005f;
        float f_84 = fma(f_80, 0.005f, f_78);
        float f_85 = fma(f_83, 0.005f, f_84);
        *((__global float *) ul_17) = f_85;
        
        float f_86 = *((__global float *) ul_79);
        float f_87 = ul_3[1];
        float f_88 = fma(f_87, 0.005f, f_86);
        *((__global float *) ul_79) = f_88;
        
        float f_89 = *((__global float *) ul_13);
        ulong ul_90 = ul_1 + l_12;
        float f_91 = *((__global float *) ul_90);
        float f_92 = ul_3[2];
        float f_93 = f_92 * 0.5f;
        float f_94 = f_93 * 0.005f;
        float f_95 = fma(f_91, 0.005f, f_89);
        float f_96 = fma(f_94, 0.005f, f_95);
        *((__global float *) ul_13) = f_96;
        
        float f_97 = *((__global float *) ul_90);
        float f_98 = ul_3[2];
        float f_99 = fma(f_98, 0.005f, f_97);
        *((__global float *) ul_90) = f_99;
    }
}
