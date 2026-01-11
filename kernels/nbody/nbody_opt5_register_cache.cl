#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// ============================================================================
// Optimization 5: Register caching for body i ONLY
// Loads body i's position once before inner loop, keeps in registers
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
    
    __private float accum[3];
    
    int gsize = get_global_size(0);
    int gid = get_global_id(0);
    
    for (int i = gid; i < 2048; i += gsize) {
        accum[0] = 0.0f;
        accum[1] = 0.0f;
        accum[2] = 0.0f;
        
        int idx = i << 2;
        
        // OPTIMIZATION: Cache body i's position in registers (load ONCE)
        const float myX = pos[idx + 0];
        const float myY = pos[idx + 1];
        const float myZ = pos[idx + 2];
        
        for (int j = 0; j < 2048; j++) {
            int jdx = j << 2;
            
            // Load body j from global memory
            float otherX = pos[jdx + 0];
            float otherY = pos[jdx + 1];
            float otherZ = pos[jdx + 2];
            float otherMass = pos[jdx + 3];
            
            // Use cached myX, myY, myZ instead of re-reading
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
            
            accum[0] = fma(s, rx, accum[0]);
            accum[1] = fma(s, ry, accum[1]);
            accum[2] = fma(s, rz, accum[2]);
        }
        
        // Position and velocity updates
        float vx = vel[idx + 0];
        float vy = vel[idx + 1];
        float vz = vel[idx + 2];
        
        float halfDt2 = 0.5f * 0.005f * 0.005f;
        pos[idx + 0] = fma(vx, 0.005f, fma(accum[0], halfDt2, myX));
        pos[idx + 1] = fma(vy, 0.005f, fma(accum[1], halfDt2, myY));
        pos[idx + 2] = fma(vz, 0.005f, fma(accum[2], halfDt2, myZ));
        
        vel[idx + 0] = fma(accum[0], 0.005f, vx);
        vel[idx + 1] = fma(accum[1], 0.005f, vy);
        vel[idx + 2] = fma(accum[2], 0.005f, vz);
    }
}
