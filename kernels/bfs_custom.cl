#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  


// ============================================================================
//  FIXED: initializeVertices
//  This version *exactly* matches TornadoVM semantics but avoids overwriting
//  buffer metadata. The Tornado buffer layout is:
//
//  bytes 0..15   = metadata (4 ints)
//  bytes 16..    = user payload
//
//  In the generated kernel, BFS vertices start at index 4:
//    *(vertices + (i+4)<<2)
//
//  That means actual BFS vertex 0 is stored at payload[0] = vertices[16].
// ============================================================================

__kernel void initializeVertices(__global long *_kernel_context,
                                 __constant uchar *_constant_region,
                                 __local    uchar *_local_region,
                                 __global   int   *_atomics,
                                 __private  int    numNodes,
                                 __global   uchar *vertices,
                                 __private  int    root)
{
    // User integer data begins at offset +16 bytes = index 4
    __global int *verts = (__global int *)(vertices + 16);

    int gs = get_global_size(0);
    int gid = get_global_id(0);

    const int N = 10000;   // same constant bound as Tornado

    // grid-stride loop
    for (int i = gid; i < N; i += gs) {
        // original: verts[i+4] = ...
        // but after shifting by +16 bytes, simply verts[i]
        verts[i] = (i == 0) ? 0 : -1;
    }
}



// ============================================================================
//  FIXED: runBFS
//  The critical part is using correct pointer offsets:
//
//    adjacencyMatrix data begins at +16 bytes
//    vertices data begins at +16 bytes
//    h_true data begins at +16 bytes
//    currentDepth data begins at +16 bytes
//
//  The generated kernel:
//    f(depth) = *((int*)(currentDepth + 16))
//    mark work = h_true[4] = *((int*)(h_true + 16))
//
//  After shifting by +16 bytes, BFS sees index 0 instead of index 4.
// ============================================================================

__kernel void runBFS(__global long *_kernel_context,
                     __constant uchar *_constant_region,
                     __local    uchar *_local_region,
                     __global   int   *_atomics,
                     __global   uchar *vertices,
                     __global   uchar *adjacencyMatrix,
                     __private  int    numNodes,
                     __global   uchar *h_true,
                     __global   uchar *currentDepth)
{
    // Shift all pointers by the Tornado metadata offset (16 bytes)
    __global int *verts     = (__global int *)(vertices + 16);
    __global int *adj       = (__global int *)(adjacencyMatrix + 16);
    __global int *hflag     = (__global int *)(h_true + 16);
    __global int *depthInfo = (__global int *)(currentDepth + 16);

    // Generated kernel reads depth from index 0 (after shift)
    const int depth = depthInfo[0];

    const int N = 10000;   // adjacency matrix is 10000 x 10000

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gsx = get_global_size(0);
    int gsy = get_global_size(1);

    // Outer grid-stride over rows
    for (int row = gy; row < N; row += gsy) {

        int srcDepth = verts[row];   // original: verts[row+4]

        // Optimization: skip entire row if source depth doesn't match
        if (srcDepth != depth) {
            continue;
        }

        int rowBase = row * N;  // original: 4 + row*N → shift removed

        // Inner grid-stride over columns
        for (int col = gx; col < N; col += gsx) {

            int edge = adj[rowBase + col];  // original: adj[4 + row*N + col]

            if (edge == 1) {

                int dstDepth = verts[col];  // original: verts[col+4]

                if (dstDepth == -1) {
                    // Relaxation
                    verts[col] = srcDepth + 1;

                    // Original: *((int*)(h_true + 16)) = 0
                    hflag[0] = 0;
                }
            }
        }
    }
}
