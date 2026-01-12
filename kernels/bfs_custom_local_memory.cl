#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// ============================================================================
// BFS Kernels - Local Memory Version (Fixed for TornadoVM)
// ============================================================================
// Fixes:
//   1. Removed reqd_work_group_size attribute (let TornadoVM choose)
//   2. Kept 2D indexing to match original kernel structure
//   3. Dynamic local memory size based on get_local_size()
// ============================================================================

#define INT_BASE_INDEX 4
#define NUM_NODES 10000

// ----------------------------------------------------------------------------
// initializeVertices - Simple version (not much to optimize here)
// ----------------------------------------------------------------------------
__kernel void initializeVertices(__global long *_kernel_context,
                                  __constant uchar *_constant_region,
                                  __local uchar *_local_region,
                                  __global int *_atomics,
                                  __private int numNodes,
                                  __global uchar *vertices,
                                  __private int root)
{
    (void)_kernel_context;
    (void)_constant_region;
    (void)_local_region;
    (void)_atomics;
    (void)numNodes;

    __global int *verts = ((__global int *)vertices) + INT_BASE_INDEX;

    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);

    for (int i = gid; i < NUM_NODES; i += gsize) {
        verts[i] = (i == root) ? 0 : -1;
    }
}

// ----------------------------------------------------------------------------
// runBFS - Local Memory Version with 2D indexing (matches TornadoVM)
// ----------------------------------------------------------------------------
__kernel void runBFS(__global long *_kernel_context,
                     __constant uchar *_constant_region,
                     __local uchar *_local_region,
                     __global int *_atomics,
                     __global uchar *vertices,
                     __global uchar *adjacencyMatrix,
                     __private int numNodes,
                     __global uchar *h_true,
                     __global uchar *currentDepth)
{
    (void)_kernel_context;
    (void)_constant_region;
    (void)_atomics;
    (void)numNodes;

    __global int *verts = ((__global int *)vertices) + INT_BASE_INDEX;
    __global const int *adjMatrix = ((__global const int *)adjacencyMatrix) + INT_BASE_INDEX;
    __global int *h_flag = ((__global int *)h_true) + INT_BASE_INDEX;
    __global const int *depth = ((__global const int *)currentDepth) + INT_BASE_INDEX;

    // 2D indexing (matches original TornadoVM kernel)
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    const int gsize_x = get_global_size(0);
    const int gsize_y = get_global_size(1);
    
    // Local thread info
    const int lid_x = get_local_id(0);
    const int lsize_x = get_local_size(0);

    // Use TornadoVM-provided local memory region
    // Cast to int array for our tile
    __local int *localTile = (__local int *)_local_region;

    // Read currentDepth ONCE (hoisted)
    const int curDepth = depth[0];
    const int nextDepth = curDepth + 1;

    // Grid-stride loop over source vertices (Y dimension = rows)
    for (int src = gid_y; src < NUM_NODES; src += gsize_y) {
        
        // EARLY EXIT: Skip row if source not at current depth
        const int srcDepth = verts[src];
        if (srcDepth != curDepth) {
            continue;
        }

        const int rowOffset = src * NUM_NODES;

        // Tile over destination vertices using local memory
        for (int tileStart = 0; tileStart < NUM_NODES; tileStart += lsize_x) {
            const int dst = tileStart + lid_x;
            
            // Load destination vertex depths into local memory (cooperative)
            int dstDepth = -2;  // Invalid marker
            if (dst < NUM_NODES) {
                dstDepth = verts[dst];
            }
            localTile[lid_x] = dstDepth;
            
            barrier(CLK_LOCAL_MEM_FENCE);

            // Now process this tile - each thread handles its element
            if (dst < NUM_NODES) {
                int edge = adjMatrix[rowOffset + dst];
                if (edge == 1 && localTile[lid_x] == -1) {
                    verts[dst] = nextDepth;
                    h_flag[0] = 0;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}
