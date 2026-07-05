#pragma OPENCL EXTENSION cl_khr_fp64 : enable                                                     
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable                                                     
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable                                       
                                                                                                    
  __kernel void nBody(                                                                              
      __global long *_kernel_context,                                                               
      __constant uchar *_constant_region,                                                           
      __local uchar *_local_region,                                                                 
      __global int *_atomics,                                                                       
      __private int numBodies,                                                                      
      __global uchar * restrict refPos,                                                             
      __global uchar * restrict refVel,                                                             
      __private float delT,                                                                         
      __private float espSqr)                                                                       
  {                                                                                                 
    // Get proper float pointers with +4 offset (TornadoVM header)                                  
    __global float * restrict pos = ((__global float *)refPos) + 4;                                 
    __global float * restrict vel = ((__global float *)refVel) + 4;                                 
                                                                                                    
    const int gid = get_global_id(0);                                                               
    const int gsize = get_global_size(0);                                                           
                                                                                                    
    // Process bodies assigned to this work-item                                                    
    for (int i = gid; i < numBodies; i += gsize)                                                    
    {                                                                                               
      // Cache current body position in registers (avoid repeated global reads)                     
      const int i4 = i << 2;                                                                        
      const float myPosX = pos[i4];                                                                 
      const float myPosY = pos[i4 + 1];                                                             
      const float myPosZ = pos[i4 + 2];                                                             
                                                                                                    
      // Use scalar accumulators instead of private array (better register allocation)              
      float accX = 0.0f;                                                                            
      float accY = 0.0f;                                                                            
      float accZ = 0.0f;                                                                            
                                                                                                    
      // Inner loop - compute gravitational forces from all bodies                                  
      #pragma unroll 4                                                                              
      for (int j = 0; j < numBodies; j++)                                                           
      {                                                                                             
        const int j4 = j << 2;                                                                      
                                                                                                    
        // Load other body position and mass                                                        
        const float otherPosX = pos[j4];                                                            
        const float otherPosY = pos[j4 + 1];                                                        
        const float otherPosZ = pos[j4 + 2];                                                        
        const float otherMass = pos[j4 + 3];                                                        
                                                                                                    
        // Compute distance vector                                                                  
        const float dx = otherPosX - myPosX;                                                        
        const float dy = otherPosY - myPosY;                                                        
        const float dz = otherPosZ - myPosZ;                                                        
                                                                                                    
        // Compute distance squared + softening                                                     
        const float distSqr = fma(dx, dx, fma(dy, dy, fma(dz, dz, espSqr)));                        
                                                                                                    
        // Compute inverse distance cubed: 1 / (dist^3) = rsqrt^3                                   
        // Using native_rsqrt for float precision (faster than double rsqrt)                        
        const float invDist = native_rsqrt(distSqr);                                                
        const float invDist3 = invDist * invDist * invDist;                                         
                                                                                                    
        // Force contribution (mass * invDist^3)                                                    
        const float force = otherMass * invDist3;                                                   
                                                                                                    
        // Accumulate acceleration                                                                  
        accX = fma(force, dx, accX);                                                                
        accY = fma(force, dy, accY);                                                                
        accZ = fma(force, dz, accZ);                                                                
      }                                                                                             
                                                                                                    
      // Update position and velocity using Verlet integration                                      
      // pos += vel * dt + 0.5 * acc * dt^2                                                         
      // vel += acc * dt                                                                            
                                                                                                    
      const float halfDtSqr = 0.5f * delT * delT;                                                   
                                                                                                    
      // Update X                                                                                   
      const float oldVelX = vel[i4];                                                                
      pos[i4] = fma(oldVelX, delT, fma(accX, halfDtSqr, myPosX));                                   
      vel[i4] = fma(accX, delT, oldVelX);                                                           
                                                                                                    
      // Update Y                                                                                   
      const float oldVelY = vel[i4 + 1];                                                            
      pos[i4 + 1] = fma(oldVelY, delT, fma(accY, halfDtSqr, myPosY));                               
      vel[i4 + 1] = fma(accY, delT, oldVelY);                                                       
                                                                                                    
      // Update Z                                                                                   
      const float oldVelZ = vel[i4 + 2];                                                            
      pos[i4 + 2] = fma(oldVelZ, delT, fma(accZ, halfDtSqr, myPosZ));                               
      vel[i4 + 2] = fma(accZ, delT, oldVelZ);                                                       
    }                                                                                               
  } 

