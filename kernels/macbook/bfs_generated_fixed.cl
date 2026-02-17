/*
 * Fixed version of TornadoVM-generated BFS kernel
 *
 * ISSUE: Original kernel had size hardcoded to 10000
 * FIX: Now uses the 'numNodes' parameter for all loop bounds
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void initializeVertices(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __private int numNodes,
    __global uchar *vertices,
    __private int root)
{
    long l_7, l_6;
    int i_10, i_4, i_3, i_2, i_1, i_5;
    bool b_9;
    ulong ul_8, ul_0;

    // BLOCK 0
    ul_0 = (ulong) vertices;
    i_1 = get_global_size(0);
    i_2 = get_global_id(0);
    // BLOCK 1 MERGES [0 5]
    i_3 = i_2;
    for (; i_3 < numNodes;)  // FIXED: was hardcoded to 10000
    {
        // BLOCK 2
        i_4 = i_1 + i_3;
        i_5 = i_3 + 4;
        l_6 = (long) i_5;
        l_7 = l_6 << 2;
        ul_8 = ul_0 + l_7;
        b_9 = i_3 == root;  // FIXED: compare to root parameter
        if (b_9)
        {
            // BLOCK 3
            *((__global int *) ul_8) = 0;
        }
        else
        {
            // BLOCK 4
            *((__global int *) ul_8) = -1;
        }
        // BLOCK 5 MERGES [3 4]
        i_10 = i_4;
        i_3 = i_10;
    }
    // BLOCK 6
    return;
}

__kernel void runBFS(
    __global long *_kernel_context,
    __constant uchar *_constant_region,
    __local uchar *_local_region,
    __global int *_atomics,
    __global uchar *vertices,
    __global uchar *adjacencyMatrix,
    __private int numNodes,
    __global uchar *h_true,
    __global uchar *currentDepth)
{
    long l_25, l_26, l_21, l_20, l_13, l_14;
    int i_28, i_30, i_4, i_36, i_35, i_33, i_8, i_7, i_6, i_5, i_12, i_11, i_10, i_9, i_16, i_19, i_17, i_24, i_23;
    bool b_18, b_31, b_32;
    ulong ul_29, ul_27, ul_22, ul_3, ul_2, ul_34, ul_1, ul_0, ul_15;

    // BLOCK 0
    ul_0 = (ulong) vertices;
    ul_1 = (ulong) adjacencyMatrix;
    ul_2 = (ulong) h_true;
    ul_3 = (ulong) currentDepth;
    i_4 = get_global_size(0);
    i_5 = get_global_size(1);
    i_6 = get_global_id(0);
    i_7 = get_global_id(1);
    // BLOCK 1 MERGES [0 12]
    i_8 = i_7;
    for (; i_8 < numNodes;)  // FIXED: was hardcoded to 10000
    {
        // BLOCK 2
        i_9 = i_8 * numNodes;  // FIXED: was i_8 * 10000
        i_10 = i_9 + 4;
        // BLOCK 3 MERGES [2 11]
        i_11 = i_6;
        for (; i_11 < numNodes;)  // FIXED: was hardcoded to 10000
        {
            // BLOCK 4
            i_12 = i_10 + i_11;
            l_13 = (long) i_12;
            l_14 = l_13 << 2;
            ul_15 = ul_1 + l_14;
            i_16 = *((__global int *) ul_15);
            i_17 = i_4 + i_11;
            b_18 = i_16 == 1;
            if (b_18)
            {
                // BLOCK 5
                i_19 = i_8 + 4;
                l_20 = (long) i_19;
                l_21 = l_20 << 2;
                ul_22 = ul_0 + l_21;
                i_23 = *((__global int *) ul_22);
                i_24 = i_11 + 4;
                l_25 = (long) i_24;
                l_26 = l_25 << 2;
                ul_27 = ul_0 + l_26;
                i_28 = *((__global int *) ul_27);
                ul_29 = ul_3 + 16L;
                i_30 = *((__global int *) ul_29);
                b_31 = i_23 == i_30;
                if (b_31)
                {
                    // BLOCK 6
                    b_32 = i_28 == -1;
                    if (b_32)
                    {
                        // BLOCK 7
                        i_33 = i_23 + 1;
                        *((__global int *) ul_27) = i_33;
                        ul_34 = ul_2 + 16L;
                        *((__global int *) ul_34) = 0;
                    }
                }
            }
            // BLOCK 11 MERGES [10 7 8 9]
            i_35 = i_17;
            i_11 = i_35;
        }
        // BLOCK 12
        i_36 = i_5 + i_8;
        i_8 = i_36;
    }
    // BLOCK 13
    return;
}
