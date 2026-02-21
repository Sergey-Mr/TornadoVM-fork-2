package uk.ac.manchester.tornado.examples.compute.custom;

import uk.ac.manchester.tornado.api.mcp.MCPKernelOptimizer;

/**
 * Test the MCP Kernel Optimizer integration.
 *
 * Usage:
 *   export TORNADOVM_MCP_PATH=/path/to/MCP-server
 *   java -Dtornado.mcp.optimization=true MCPIntegrationTest
 */
public class MCPIntegrationTest {

    // Sample inefficient kernel for testing
    private static final String TEST_KERNEL = """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable

        __kernel void matrixMultiplication(
            __global long *_kernel_context,
            __constant uchar *_constant_region,
            __local uchar *_local_region,
            __global int *_atomics,
            __global uchar *A,
            __global uchar *B,
            __global uchar *C,
            __private int size)
        {
            int N = (int) _kernel_context[0];

            __global float *a = (__global float *)(A + 16);
            __global float *b = (__global float *)(B + 16);
            __global float *c = (__global float *)(C + 16);

            int row = get_global_id(0);
            int col = get_global_id(1);

            if (row < N && col < N) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum = sum + a[row * N + k] * b[k * N + col];
                }
                c[row * N + col] = sum;
            }
        }
        """;

    public static void main(String[] args) {
        System.out.println("=== MCP Kernel Optimizer Integration Test ===\n");

        // Check environment
        String mcpPath = System.getenv("TORNADOVM_MCP_PATH");
        if (mcpPath == null || mcpPath.isEmpty()) {
            System.err.println("ERROR: TORNADOVM_MCP_PATH environment variable not set");
            System.err.println("Set it to the MCP-server directory path");
            System.exit(1);
        }
        System.out.println("MCP Server Path: " + mcpPath);

        MCPKernelOptimizer optimizer = new MCPKernelOptimizer();

        try {
            System.out.println("\n[1] Starting MCP server...");
            optimizer.start();
            System.out.println("    Server started successfully");

            System.out.println("\n[2] Sending kernel for optimization...");
            System.out.println("    Original kernel: " + TEST_KERNEL.length() + " chars");

            long startTime = System.currentTimeMillis();

            String optimizedKernel = optimizer.optimizeKernel(
                TEST_KERNEL,
                "opencl",
                "nvidia_ada",
                15_000_000,     // kernel_time_ns (15ms)
                2_000_000,      // copy_in_time_ns (2ms)
                1_000_000,      // copy_out_time_ns (1ms)
                8_388_608,      // copy_in_bytes (8MB)
                4_194_304,      // copy_out_bytes (4MB)
                new int[]{1024, 1024},  // global_work_size
                new int[]{16, 16}       // local_work_size
            );

            long elapsed = System.currentTimeMillis() - startTime;

            System.out.println("\n[3] Optimization complete in " + elapsed + "ms");
            System.out.println("    Optimized kernel: " + optimizedKernel.length() + " chars");

            System.out.println("\n=== Optimized Kernel ===\n");
            System.out.println(optimizedKernel);

        } catch (Exception e) {
            System.err.println("ERROR: " + e.getMessage());
            e.printStackTrace();
        } finally {
            System.out.println("\n[4] Stopping MCP server...");
            optimizer.stop();
            System.out.println("    Done");
        }
    }
}
