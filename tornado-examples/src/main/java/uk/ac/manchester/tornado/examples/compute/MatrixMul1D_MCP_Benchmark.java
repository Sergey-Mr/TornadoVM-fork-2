/*
 * Matrix Multiplication 1D - MCP Optimization Benchmark
 *
 * This benchmark properly compares:
 * - Original TornadoVM kernel (GPU)
 * - MCP-optimized kernel (GPU)
 *
 * Usage:
 *   1. Start MCP server: python -m tornadovm_mcp.http_server 8090
 *   2. Run: tornado -m tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMul1D_MCP_Benchmark
 */
package uk.ac.manchester.tornado.examples.compute;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.mcp.MCPKernelOptimizer;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class MatrixMul1D_MCP_Benchmark {

    private static final int WARMUP = 15;
    private static final int ITERATIONS = 50;

    private static void matrixMultiplication(final FloatArray A, final FloatArray B,
                                              final FloatArray C, final int size) {
        for (@Parallel int i = 0; i < size; i++) {
            for (@Parallel int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    sum += A.get((i * size) + k) * B.get((k * size) + j);
                }
                C.set((i * size) + j, sum);
            }
        }
    }

    public static void main(String[] args) {
        int size = 512;
        if (args.length >= 1) {
            try { size = Integer.parseInt(args[0]); } catch (Exception e) {}
        }

        System.out.println("╔════════════════════════════════════════════════════════════════╗");
        System.out.println("║       Matrix Multiplication - MCP Optimization Benchmark       ║");
        System.out.println("╚════════════════════════════════════════════════════════════════╝");
        System.out.println("\nMatrix size: " + size + "x" + size);
        System.out.println("Warmup: " + WARMUP + " iterations, Benchmark: " + ITERATIONS + " iterations\n");

        // Initialize data
        FloatArray A = new FloatArray(size * size);
        FloatArray B = new FloatArray(size * size);
        FloatArray C = new FloatArray(size * size);

        Random r = new Random(42);
        IntStream.range(0, size * size).forEach(idx -> {
            A.set(idx, r.nextFloat());
            B.set(idx, r.nextFloat());
        });

        // Create TaskGraph
        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, A, B)
                .task("t0", MatrixMul1D_MCP_Benchmark::matrixMultiplication, A, B, C, size)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, C);

        ImmutableTaskGraph itg = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(itg);

        // ═══════════════════════════════════════════════════════════════
        // PHASE 1: Benchmark ORIGINAL TornadoVM kernel
        // ═══════════════════════════════════════════════════════════════
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("PHASE 1: Benchmarking ORIGINAL TornadoVM Kernel");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Warmup
        System.out.print("  Warming up... ");
        for (int i = 0; i < WARMUP; i++) {
            executor.execute();
        }
        System.out.println("done");

        // Benchmark original kernel
        System.out.print("  Benchmarking... ");
        List<Long> originalTimes = new ArrayList<>();
        for (int i = 0; i < ITERATIONS; i++) {
            TornadoExecutionResult result = executor.withProfiler(ProfilerMode.SILENT).execute();
            long kernelTime = result.getProfilerResult().getDeviceKernelTime();
            originalTimes.add(kernelTime);
        }
        System.out.println("done");

        double originalAvgNs = originalTimes.stream().mapToLong(Long::longValue).average().orElse(0);
        double originalAvgMs = originalAvgNs / 1_000_000.0;
        double originalGflops = (2.0 * Math.pow(size, 3)) / originalAvgNs;

        System.out.printf("  Result: %.3f ms (%.2f GFLOP/s)\n", originalAvgMs, originalGflops);

        // Save original kernel source
        String originalKernel = executor.getGeneratedKernelSource("t0");
        System.out.println("  Original kernel: " + originalKernel.length() + " chars");

        // ═══════════════════════════════════════════════════════════════
        // PHASE 2: Call MCP Server for Optimization
        // ═══════════════════════════════════════════════════════════════
        System.out.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("PHASE 2: Calling MCP Server for AI Optimization");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        MCPKernelOptimizer optimizer = new MCPKernelOptimizer();
        System.out.println("  Sending to MCP server (this may take 30-60 seconds)...");

        String optimizedKernel = optimizer.optimize(originalKernel, "opencl", (long) originalAvgNs);

        if (optimizedKernel == null || optimizedKernel.isEmpty()) {
            System.err.println("  ERROR: MCP optimization failed!");
            System.err.println("  Make sure MCP server is running: python -m tornadovm_mcp.http_server 8090");
            return;
        }

        System.out.println("  Optimized kernel received: " + optimizedKernel.length() + " chars");

        // Replace kernel
        boolean replaced = executor.replaceKernelSource("t0", optimizedKernel);
        if (!replaced) {
            System.err.println("  ERROR: Kernel replacement failed!");
            return;
        }
        System.out.println("  Kernel replaced successfully");

        // ═══════════════════════════════════════════════════════════════
        // PHASE 3: Benchmark MCP-OPTIMIZED kernel
        // ═══════════════════════════════════════════════════════════════
        System.out.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("PHASE 3: Benchmarking MCP-OPTIMIZED Kernel");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Reset result
        for (int i = 0; i < size * size; i++) {
            C.set(i, 0);
        }

        // Warmup optimized kernel
        System.out.print("  Warming up... ");
        for (int i = 0; i < WARMUP; i++) {
            executor.execute();
        }
        System.out.println("done");

        // Benchmark optimized kernel
        System.out.print("  Benchmarking... ");
        List<Long> optimizedTimes = new ArrayList<>();
        for (int i = 0; i < ITERATIONS; i++) {
            TornadoExecutionResult result = executor.withProfiler(ProfilerMode.SILENT).execute();
            long kernelTime = result.getProfilerResult().getDeviceKernelTime();
            optimizedTimes.add(kernelTime);
        }
        System.out.println("done");

        double optimizedAvgNs = optimizedTimes.stream().mapToLong(Long::longValue).average().orElse(0);
        double optimizedAvgMs = optimizedAvgNs / 1_000_000.0;
        double optimizedGflops = (2.0 * Math.pow(size, 3)) / optimizedAvgNs;

        System.out.printf("  Result: %.3f ms (%.2f GFLOP/s)\n", optimizedAvgMs, optimizedGflops);

        // Verify correctness
        FloatArray reference = new FloatArray(size * size);
        matrixMultiplication(A, B, reference, size);
        boolean correct = true;
        for (int i = 0; i < size * size; i++) {
            if (Math.abs(C.get(i) - reference.get(i)) > 0.1f) {
                correct = false;
                break;
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // RESULTS
        // ═══════════════════════════════════════════════════════════════
        System.out.println("\n╔════════════════════════════════════════════════════════════════╗");
        System.out.println("║                          RESULTS                               ║");
        System.out.println("╠════════════════════════════════════════════════════════════════╣");
        System.out.printf("║  Original TornadoVM:  %8.3f ms  (%7.2f GFLOP/s)           ║\n", originalAvgMs, originalGflops);
        System.out.printf("║  MCP-Optimized:       %8.3f ms  (%7.2f GFLOP/s)           ║\n", optimizedAvgMs, optimizedGflops);
        System.out.println("╠════════════════════════════════════════════════════════════════╣");

        double speedup = originalAvgMs / optimizedAvgMs;
        double improvement = ((originalAvgMs - optimizedAvgMs) / originalAvgMs) * 100;

        if (speedup >= 1.0) {
            System.out.printf("║  Speedup: %.2fx FASTER (%.1f%% improvement)                    ║\n", speedup, improvement);
        } else {
            System.out.printf("║  Speedup: %.2fx SLOWER (%.1f%% slower)                         ║\n", speedup, -improvement);
        }
        System.out.printf("║  Correctness: %-6s                                           ║\n", correct ? "PASSED" : "FAILED");
        System.out.println("╚════════════════════════════════════════════════════════════════╝");
    }
}
