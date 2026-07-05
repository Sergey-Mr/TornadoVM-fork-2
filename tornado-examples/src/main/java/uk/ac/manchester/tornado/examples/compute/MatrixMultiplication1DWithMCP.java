/*
 * Matrix Multiplication 1D with MCP Optimization
 *
 * Real-world example showing:
 * 1. Run original TornadoVM kernel
 * 2. Extract generated OpenCL kernel
 * 3. Call MCP server for AI optimization
 * 4. Replace kernel and re-run
 * 5. Compare performance
 *
 * Usage:
 *   Terminal 1: cd MCP-server && source .venv/bin/activate && python -m tornadovm_mcp.http_server 8090
 *   Terminal 2: tornado -m tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1DWithMCP
 */
package uk.ac.manchester.tornado.examples.compute;

import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.IntStream;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class MatrixMultiplication1DWithMCP {

    // MCP Server configuration
    private static final String MCP_SERVER_URL = "http://localhost:8090/optimize";

    // Benchmark configuration
    private static final int WARMUP_ITERATIONS = 15;
    private static final int BENCHMARK_ITERATIONS = 50;

    /**
     * Matrix multiplication kernel - same as original TornadoVM example
     */
    private static void matrixMultiplication(final FloatArray matrixA, final FloatArray matrixB,
                                              final FloatArray result, final int size) {
        for (@Parallel int i = 0; i < size; i++) {
            for (@Parallel int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    sum += matrixA.get((i * size) + k) * matrixB.get((k * size) + j);
                }
                result.set((i * size) + j, sum);
            }
        }
    }

    public static void main(String[] args) {
        int size = 512;
        if (args.length >= 1) {
            try {
                size = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                // ignore
            }
        }

        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║     Matrix Multiplication 1D with MCP Optimization           ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
        System.out.println("\nMatrix size: " + size + "x" + size);
        System.out.println("FLOPs per run: " + String.format("%.2f", 2.0 * Math.pow(size, 3) / 1e9) + " GFLOP\n");

        // Initialize matrices
        FloatArray matrixA = new FloatArray(size * size);
        FloatArray matrixB = new FloatArray(size * size);
        FloatArray matrixC = new FloatArray(size * size);

        Random r = new Random(42);  // Fixed seed for reproducibility
        IntStream.range(0, size * size).forEach(idx -> {
            matrixA.set(idx, r.nextFloat());
            matrixB.set(idx, r.nextFloat());
        });

        // Create TaskGraph
        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA, matrixB)
                .task("t0", MatrixMultiplication1DWithMCP::matrixMultiplication, matrixA, matrixB, matrixC, size)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, matrixC);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);

        // ═══════════════════════════════════════════════════════════════
        // PHASE 1: Benchmark Original Kernel
        // ═══════════════════════════════════════════════════════════════
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("PHASE 1: Running Original TornadoVM Kernel");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Warmup
        System.out.println("Warming up (" + WARMUP_ITERATIONS + " iterations)...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            executor.execute();
        }

        // Benchmark
        System.out.println("Benchmarking (" + BENCHMARK_ITERATIONS + " iterations)...");
        List<Long> originalTimes = new ArrayList<>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            TornadoExecutionResult result = executor.withProfiler(ProfilerMode.SILENT).execute();
            long kernelTime = result.getProfilerResult().getDeviceKernelTime();
            originalTimes.add(kernelTime);
        }

        double originalAvgNs = originalTimes.stream().mapToLong(Long::longValue).average().orElse(0);
        double originalAvgMs = originalAvgNs / 1_000_000.0;
        double originalGflops = (2.0 * Math.pow(size, 3)) / originalAvgNs;

        System.out.printf("\n  Original kernel time: %.3f ms (%.2f GFLOP/s)\n", originalAvgMs, originalGflops);

        // ═══════════════════════════════════════════════════════════════
        // PHASE 2: Extract Generated Kernel
        // ═══════════════════════════════════════════════════════════════
        System.out.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("PHASE 2: Extracting Generated OpenCL Kernel");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        String kernelSource = executor.getGeneratedKernelSource("t0");
        if (kernelSource == null) {
            System.err.println("ERROR: Could not extract kernel source!");
            return;
        }

        System.out.println("  Kernel extracted: " + kernelSource.length() + " characters");
        System.out.println("\n  --- First 600 chars of generated kernel ---");
        System.out.println(kernelSource.substring(0, Math.min(600, kernelSource.length())));
        System.out.println("  ...\n");

        // ═══════════════════════════════════════════════════════════════
        // PHASE 3: Call MCP Server for Optimization
        // ═══════════════════════════════════════════════════════════════
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("PHASE 3: Calling MCP Server for AI Optimization");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        String optimizedKernel = null;
        try {
            System.out.println("  Connecting to " + MCP_SERVER_URL + "...");
            System.out.println("  Sending kernel + profiling data (kernel_time=" + (long)originalAvgNs + "ns)");
            System.out.println("  Waiting for Claude AI optimization (this may take 30-60 seconds)...\n");

            optimizedKernel = callMCPServer(kernelSource, "opencl", "apple_m4", (long) originalAvgNs);

            if (optimizedKernel != null) {
                System.out.println("  MCP optimization received!");
                System.out.println("  Optimized kernel: " + optimizedKernel.length() + " characters");
            }
        } catch (IOException e) {
            System.err.println("\n  ERROR: MCP server call failed: " + e.getMessage());
            System.err.println("\n  Make sure MCP server is running:");
            System.err.println("    cd /Users/serhiitupikin/Documents/Coding/TornadoVM_MCP/MCP-server");
            System.err.println("    source .venv/bin/activate");
            System.err.println("    python -m tornadovm_mcp.http_server 8090");
            return;
        }

        if (optimizedKernel == null || optimizedKernel.isEmpty()) {
            System.err.println("  ERROR: MCP returned empty optimization!");
            return;
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 4: Replace Kernel and Re-benchmark
        // ═══════════════════════════════════════════════════════════════
        System.out.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("PHASE 4: Running MCP-Optimized Kernel");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        boolean replaced = executor.replaceKernelSource("t0", optimizedKernel);
        if (!replaced) {
            System.err.println("  ERROR: Kernel replacement failed!");
            return;
        }
        System.out.println("  Kernel replaced successfully.");

        // Clear result for fresh run
        for (int i = 0; i < size * size; i++) {
            matrixC.set(i, 0);
        }

        // Warmup optimized kernel
        System.out.println("  Warming up optimized kernel...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            executor.execute();
        }

        // Benchmark optimized kernel
        System.out.println("  Benchmarking optimized kernel...");
        List<Long> optimizedTimes = new ArrayList<>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            TornadoExecutionResult result = executor.withProfiler(ProfilerMode.SILENT).execute();
            long kernelTime = result.getProfilerResult().getDeviceKernelTime();
            optimizedTimes.add(kernelTime);
        }

        double optimizedAvgNs = optimizedTimes.stream().mapToLong(Long::longValue).average().orElse(0);
        double optimizedAvgMs = optimizedAvgNs / 1_000_000.0;
        double optimizedGflops = (2.0 * Math.pow(size, 3)) / optimizedAvgNs;

        System.out.printf("\n  Optimized kernel time: %.3f ms (%.2f GFLOP/s)\n", optimizedAvgMs, optimizedGflops);

        // Verify correctness
        FloatArray reference = new FloatArray(size * size);
        matrixMultiplication(matrixA, matrixB, reference, size);
        boolean correct = verify(matrixC, reference, size);

        // ═══════════════════════════════════════════════════════════════
        // RESULTS
        // ═══════════════════════════════════════════════════════════════
        System.out.println("\n╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║                         RESULTS                              ║");
        System.out.println("╠══════════════════════════════════════════════════════════════╣");
        System.out.printf("║  Original Kernel:  %8.3f ms  (%6.2f GFLOP/s)            ║\n", originalAvgMs, originalGflops);
        System.out.printf("║  Optimized Kernel: %8.3f ms  (%6.2f GFLOP/s)            ║\n", optimizedAvgMs, optimizedGflops);
        System.out.println("╠══════════════════════════════════════════════════════════════╣");

        double speedup = originalAvgMs / optimizedAvgMs;
        if (speedup > 1.0) {
            System.out.printf("║  Speedup: %.2fx FASTER                                       ║\n", speedup);
        } else {
            System.out.printf("║  Speedup: %.2fx (slower)                                     ║\n", speedup);
        }
        System.out.printf("║  Correctness: %s                                          ║\n", correct ? "PASSED" : "FAILED");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
    }

    /**
     * Call the MCP HTTP server for kernel optimization
     */
    private static String callMCPServer(String kernelCode, String backend, String deviceFamily, long kernelTimeNs)
            throws IOException {

        URL url = new URL(MCP_SERVER_URL);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        conn.setConnectTimeout(60000);
        conn.setReadTimeout(180000);  // 3 min for LLM

        String jsonRequest = String.format(
                "{\"kernel_code\": %s, \"backend\": \"%s\", \"device_family\": \"%s\", \"kernel_time_ns\": %d}",
                escapeJson(kernelCode), backend, deviceFamily, kernelTimeNs
        );

        try (OutputStream os = conn.getOutputStream()) {
            os.write(jsonRequest.getBytes(StandardCharsets.UTF_8));
        }

        int responseCode = conn.getResponseCode();
        if (responseCode != 200) {
            try (Scanner scanner = new Scanner(conn.getErrorStream(), StandardCharsets.UTF_8)) {
                String error = scanner.useDelimiter("\\A").next();
                throw new IOException("MCP error (" + responseCode + "): " + error);
            }
        }

        String response;
        try (Scanner scanner = new Scanner(conn.getInputStream(), StandardCharsets.UTF_8)) {
            response = scanner.useDelimiter("\\A").next();
        }

        // Parse optimized_kernel from JSON
        int start = response.indexOf("\"optimized_kernel\":");
        if (start == -1) return null;
        start = response.indexOf("\"", start + 19) + 1;
        int end = findEndOfJsonString(response, start);

        return response.substring(start, end)
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
    }

    private static String escapeJson(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 32) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
            }
        }
        return sb.append("\"").toString();
    }

    private static int findEndOfJsonString(String json, int start) {
        boolean escaped = false;
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') return i;
        }
        return json.length();
    }

    private static boolean verify(FloatArray par, FloatArray seq, int size) {
        for (int i = 0; i < size * size; i++) {
            if (Math.abs(par.get(i) - seq.get(i)) > 0.1f) {
                return false;
            }
        }
        return true;
    }
}
