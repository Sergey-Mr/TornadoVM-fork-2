#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// ============================================================================
// Optimization 6: Scalar accumulators ONLY
// Uses scalar floats instead of private arrays to guarantee register usage
// Still uses FP64 rsqrt like generated kernel
// ============================================================================

#define FLOAT_OFFSET 4

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
    __global float *pos = ((__global float *)refPos) + FLOAT_OFFSET;
    __global float *vel = ((__global float *)refVel) + FLOAT_OFFSET;
    
    int gsize = get_global_size(0);
    int gid = get_global_id(0);
    
    for (int i = gid; i < 2048; i += gsize) {
        // OPTIMIZATION: Scalar accumulators instead of private arrays
        float accX = 0.0f;
        float accY = 0.0f;
        float accZ = 0.0f;
        
        int idx = i << 2;
        
        for (int j = 0; j < 2048; j++) {
            int jdx = j << 2;
            
            // Load positions (still reading body i each iteration like generated)
            float myX = pos[idx + 0];
            float myY = pos[idx + 1];
            float myZ = pos[idx + 2];
            
            float otherX = pos[jdx + 0];
            float otherY = pos[jdx + 1];
            float otherZ = pos[jdx + 2];
            float otherMass = pos[jdx + 3];
            
            float rx = otherX - myX;
            float ry = otherY - myY;
            float rz = otherZ - myZ;
            
            float distSqr = fma(rx, rx, fma(ry, ry, fma(rz, rz, 500.0f)));
            
            // Original FP64 rsqrt (unchanged)
            double d_dist = (double) distSqr;
            double d_inv = rsqrt(d_dist);
            float invDist = (float) d_inv;
            
            float invDist3 = invDist * invDist * invDist;
            float s = invDist3 * otherMass;
            
            // OPTIMIZATION: Accumulate directly into scalars (guaranteed registers)
            accX = fma(s, rx, accX);
            accY = fma(s, ry, accY);
            accZ = fma(s, rz, accZ);
        }
        
        // Position and velocity updates using scalar accumulators
        float myX = pos[idx + 0];
        float myY = pos[idx + 1];
        float myZ = pos[idx + 2];
        
        float vx = vel[idx + 0];
        float vy = vel[idx + 1];
        float vz = vel[idx + 2];
        
        float halfDt2 = 0.5f * 0.005f * 0.005f;
        pos[idx + 0] = fma(vx, 0.005f, fma(accX, halfDt2, myX));
        pos[idx + 1] = fma(vy, 0.005f, fma(accY, halfDt2, myY));
        pos[idx + 2] = fma(vz, 0.005f, fma(accZ, halfDt2, myZ));
        
        vel[idx + 0] = fma(accX, 0.005f, vx);
        vel[idx + 1] = fma(accY, 0.005f, vy);
        vel[idx + 2] = fma(accZ, 0.005f, vz);
    }
}
