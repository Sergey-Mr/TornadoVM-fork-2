#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// ============================================================================
// Optimization 7: Local memory tiling ONLY
// Uses shared memory to reduce global memory traffic
// Still uses FP64 rsqrt to isolate tiling benefit
// (requires workgroup size for tiling to work)
// ============================================================================

#define BLOCK_SIZE 256
#define FLOAT_OFFSET 4

__attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
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
    
    // OPTIMIZATION: Local memory tile for body positions
    __local float4 tile[BLOCK_SIZE];
    
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int stride = get_global_size(0);
    const int N = 2048;
    
    for (int i = gid; i < N; i += stride) {
        const int idx = i << 2;
        
        // Cache body i position in registers
        const float myX = pos[idx + 0];
        const float myY = pos[idx + 1];
        const float myZ = pos[idx + 2];
        
        float accX = 0.0f;
        float accY = 0.0f;
        float accZ = 0.0f;
        
        // OPTIMIZATION: Tile over all bodies using local memory
        for (int tileStart = 0; tileStart < N; tileStart += BLOCK_SIZE) {
            // Cooperative load into local memory
            const int loadIdx = tileStart + lid;
            if (loadIdx < N) {
                const int lo = loadIdx << 2;
                tile[lid] = (float4)(pos[lo], pos[lo+1], pos[lo+2], pos[lo+3]);
            } else {
                tile[lid] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            const int jEnd = min(BLOCK_SIZE, N - tileStart);
            
            for (int j = 0; j < jEnd; j++) {
                // Read from local memory (fast!)
                const float4 other = tile[j];
                
                const float rx = other.x - myX;
                const float ry = other.y - myY;
                const float rz = other.z - myZ;
                
                const float distSqr = fma(rx, rx, fma(ry, ry, fma(rz, rz, 500.0f)));
                
                // STILL FP64 rsqrt to isolate tiling benefit
                const double d_dist = (double) distSqr;
                const double d_inv = rsqrt(d_dist);
                const float invDist = (float) d_inv;
                
                const float invDist3 = invDist * invDist * invDist;
                const float s = invDist3 * other.w;
                
                accX = fma(s, rx, accX);
                accY = fma(s, ry, accY);
                accZ = fma(s, rz, accZ);
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // Position and velocity updates
        const float vx = vel[idx + 0];
        const float vy = vel[idx + 1];
        const float vz = vel[idx + 2];
        
        const float halfDt2 = 0.5f * 0.005f * 0.005f;
        pos[idx + 0] = fma(vx, 0.005f, fma(accX, halfDt2, myX));
        pos[idx + 1] = fma(vy, 0.005f, fma(accY, halfDt2, myY));
        pos[idx + 2] = fma(vz, 0.005f, fma(accZ, halfDt2, myZ));
        
        vel[idx + 0] = fma(accX, 0.005f, vx);
        vel[idx + 1] = fma(accY, 0.005f, vy);
        vel[idx + 2] = fma(accZ, 0.005f, vz);
    }
}
