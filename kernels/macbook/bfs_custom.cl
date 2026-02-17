/*
 * Optimized BFS (Breadth-First Search) Kernel for Apple M4 Pro Max
 *
 * Algorithm: Level-synchronous BFS
 * - Each kernel invocation processes one BFS level
 * - All vertices at current depth explore their neighbors
 * - Unvisited neighbors are marked with depth + 1
 *
 * KEY ISSUES FIXED:
 * 1. Size was hardcoded to 10000 - now uses numNodes parameter
 * 2. currentDepth was loaded inside inner loop - now hoisted
 * 3. Redundant address calculations - simplified
 * 4. Apple M4 work-group limit (32 threads) - accounted for
 *
 * Optimizations applied:
 * - Loop invariant hoisting (currentDepth, base addresses)
 * - Simplified pointer arithmetic
 * - restrict keyword for compiler hints
 * - Early exit on edge check
 * - Coalesced memory access where possible
 *
 * Bottleneck: MEMORY-BOUND with BRANCH DIVERGENCE
 * - Irregular memory access (graph-dependent)
 * - Unpredictable branching (edge existence)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TornadoVM array header offsets
#define INT_BASE_INDEX 4    // 16 bytes / 4 bytes per int = 4

/*
 * Kernel 1: Initialize vertex distances
 * - Root vertex gets distance 0
 * - All other vertices get -1 (unvisited)
 */
__kernel void initializeVertices(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __private int numNodes,
    __global uchar * restrict vertices,
    __private int root)
{
    __global int * restrict verts = ((__global int *)vertices) + INT_BASE_INDEX;

    const int gid = get_global_id(0);
    const int stride = get_global_size(0);

    // Grid-stride loop for any graph size
    for (int i = gid; i < numNodes; i += stride) {
        verts[i] = (i == root) ? 0 : -1;
    }
}


/*
 * Kernel 2: Main BFS traversal (one level per invocation)
 *
 * For each edge (src -> dst):
 *   If edge exists AND src is at currentDepth AND dst is unvisited:
 *     Set dst distance = currentDepth + 1
 *     Set h_true[0] = 0 (signal more work needed)
 */
__kernel void runBFS(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar * restrict vertices,
    __global uchar * restrict adjacencyMatrix,
    __private int numNodes,
    __global uchar * restrict h_true,
    __global uchar * restrict currentDepth)
{
    // Get typed pointers with TornadoVM header offset
    __global int * restrict verts = ((__global int *)vertices) + INT_BASE_INDEX;
    __global int * restrict adjMatrix = ((__global int *)adjacencyMatrix) + INT_BASE_INDEX;
    __global int * restrict modify = ((__global int *)h_true) + INT_BASE_INDEX;
    __global int * restrict depthPtr = ((__global int *)currentDepth) + INT_BASE_INDEX;

    // OPTIMIZATION 1: Hoist loop-invariant load
    // currentDepth is constant for entire kernel invocation
    const int depth = depthPtr[0];
    const int nextDepth = depth + 1;

    // 2D grid indices
    const int gidX = get_global_id(0);   // Target vertex (column)
    const int gidY = get_global_id(1);   // Source vertex (row)
    const int strideX = get_global_size(0);
    const int strideY = get_global_size(1);

    // Grid-stride loop over source vertices (rows)
    for (int src = gidY; src < numNodes; src += strideY) {

        // OPTIMIZATION 2: Load source vertex distance once per row
        const int srcDist = verts[src];

        // OPTIMIZATION 3: Early skip if source not at current depth
        // This reduces work significantly as BFS progresses
        if (srcDist != depth) {
            continue;  // Skip entire row
        }

        // Base index for this row in adjacency matrix
        const int rowBase = src * numNodes;

        // Grid-stride loop over target vertices (columns)
        for (int dst = gidX; dst < numNodes; dst += strideX) {

            // Check if edge exists
            const int edgeExists = adjMatrix[rowBase + dst];

            if (edgeExists == 1) {
                // Load target vertex distance
                const int dstDist = verts[dst];

                // If target is unvisited, update it
                if (dstDist == -1) {
                    verts[dst] = nextDepth;
                    modify[0] = 0;  // Signal that we made progress
                }
            }
        }
    }
}


/*
 * Alternative Kernel: Row-based BFS (better for sparse graphs)
 * Each work-item processes one source vertex (row)
 * Better work distribution for sparse adjacency matrices
 */
__kernel void runBFSRowBased(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar * restrict vertices,
    __global uchar * restrict adjacencyMatrix,
    __private int numNodes,
    __global uchar * restrict h_true,
    __global uchar * restrict currentDepth)
{
    __global int * restrict verts = ((__global int *)vertices) + INT_BASE_INDEX;
    __global int * restrict adjMatrix = ((__global int *)adjacencyMatrix) + INT_BASE_INDEX;
    __global int * restrict modify = ((__global int *)h_true) + INT_BASE_INDEX;
    __global int * restrict depthPtr = ((__global int *)currentDepth) + INT_BASE_INDEX;

    const int depth = depthPtr[0];
    const int nextDepth = depth + 1;

    const int gid = get_global_id(0);
    const int stride = get_global_size(0);

    // Each work-item processes source vertices with stride
    for (int src = gid; src < numNodes; src += stride) {

        const int srcDist = verts[src];

        // Skip if not at current depth
        if (srcDist != depth) {
            continue;
        }

        const int rowBase = src * numNodes;

        // Process all neighbors of this source vertex
        for (int dst = 0; dst < numNodes; dst++) {

            if (adjMatrix[rowBase + dst] == 1) {
                if (verts[dst] == -1) {
                    verts[dst] = nextDepth;
                    modify[0] = 0;
                }
            }
        }
    }
}


/*
 * Alternative Kernel: Optimized with local memory for neighbor list
 * Caches adjacency row in local memory before processing
 * Better for dense graphs where most edges exist
 */
__kernel void runBFSLocalMem(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar * restrict vertices,
    __global uchar * restrict adjacencyMatrix,
    __private int numNodes,
    __global uchar * restrict h_true,
    __global uchar * restrict currentDepth)
{
    // Local memory for caching adjacency row (max 32 elements for Apple M4)
    __local int localRow[32];

    __global int * restrict verts = ((__global int *)vertices) + INT_BASE_INDEX;
    __global int * restrict adjMatrix = ((__global int *)adjacencyMatrix) + INT_BASE_INDEX;
    __global int * restrict modify = ((__global int *)h_true) + INT_BASE_INDEX;
    __global int * restrict depthPtr = ((__global int *)currentDepth) + INT_BASE_INDEX;

    const int depth = depthPtr[0];
    const int nextDepth = depth + 1;

    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);
    const int groupId = get_group_id(0);
    const int numGroups = get_num_groups(0);

    // Each work-group processes one source vertex
    for (int src = groupId; src < numNodes; src += numGroups) {

        const int srcDist = verts[src];

        // Skip if not at current depth
        if (srcDist != depth) {
            continue;
        }

        const int rowBase = src * numNodes;

        // Process neighbors in chunks of localSize
        for (int chunkStart = 0; chunkStart < numNodes; chunkStart += localSize) {

            int dst = chunkStart + lid;

            // Cooperative load of adjacency chunk into local memory
            if (dst < numNodes) {
                localRow[lid] = adjMatrix[rowBase + dst];
            } else {
                localRow[lid] = 0;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Process this chunk
            if (dst < numNodes && localRow[lid] == 1) {
                if (verts[dst] == -1) {
                    verts[dst] = nextDepth;
                    modify[0] = 0;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}
