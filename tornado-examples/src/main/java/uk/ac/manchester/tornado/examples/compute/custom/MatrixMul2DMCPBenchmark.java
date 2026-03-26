package uk.ac.manchester.tornado.examples.compute.custom;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LongSummaryStatistics;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;

/**
 * MCP Kernel Optimization Benchmark for Matrix Multiplication 2D.
 *
 * Uses prebuiltTask for BOTH original and optimized kernels to ensure
 * fair comparison with proper grid configuration.
 *
 * Usage: java ... MatrixMul2DMCPBenchmark <original_kernel.cl> [size] [mcp_server_url]
 */
public class MatrixMul2DMCPBenchmark {

    private static final int DEFAULT_SIZE = 1024;
    private static final int TS = 16; // Default tile size
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;
    private static final String ENTRY_POINT = "matrixMultiplication";
    private static final String DEFAULT_MCP_URL = "http://localhost:8090/optimize";
    private static final Random RANDOM = new Random(42);

    // Store MCP response for grid config extraction
    private static String lastMcpResponse = null;

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Usage: MatrixMul2DMCPBenchmark <original_kernel.cl> [size] [mcp_server_url]");
            System.exit(1);
        }

        String originalKernelPath = args[0];
        int size = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_SIZE;
        String mcpUrl = (args.length >= 3) ? args[2] : DEFAULT_MCP_URL;

        System.out.println("=== MCP Matrix Multiplication 2D Benchmark ===");
        System.out.println("Original kernel: " + originalKernelPath);
        System.out.println("Matrix size: " + size + "x" + size);
        System.out.println("MCP server: " + mcpUrl);
        System.out.println();

        // Read original kernel
        String originalKernel = readFile(originalKernelPath);

        // Setup data
        FloatArray matrixA = new FloatArray(size * size);
        FloatArray matrixB = new FloatArray(size * size);
        FloatArray matrixC = new FloatArray(size * size);
        fillRandomData(matrixA);
        fillRandomData(matrixB);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);
        System.out.println();

        // Benchmark original kernel
        System.out.println("--- Benchmarking Original Kernel ---");
        double originalTimeMs = benchmarkKernel(originalKernel, matrixA, matrixB, matrixC, size, device, null);
        System.out.printf("Original kernel: %.3f ms%n", originalTimeMs);
        System.out.println();

        // Get optimized kernel from MCP
        System.out.println("--- Getting Optimized Kernel from MCP ---");
        String optimizedKernel = callMCPServer(mcpUrl, originalKernel, originalTimeMs);

        if (optimizedKernel == null || optimizedKernel.isEmpty()) {
            System.err.println("MCP returned empty kernel, aborting.");
            return;
        }

        // Save optimized kernel for inspection
        String optimizedPath = originalKernelPath.replace(".cl", "_mcp_optimized.cl");
        writeFile(optimizedPath, optimizedKernel);
        System.out.println("Optimized kernel saved to: " + optimizedPath);
        System.out.println();

        // Parse grid config from MCP response
        Map<String, Long> paramValues = new HashMap<>();
        paramValues.put("size", (long) size);
        GridConfigParser.GridConfig gridConfig = null;
        if (lastMcpResponse != null) {
            gridConfig = GridConfigParser.parseFromResponse(lastMcpResponse, paramValues);
            if (gridConfig != null) {
                System.out.println("Grid config from MCP: " + gridConfig);
            } else {
                System.out.println("No grid config in MCP response, using defaults");
            }
        }
        System.out.println();

        // Benchmark optimized kernel
        System.out.println("--- Benchmarking Optimized Kernel ---");
        double optimizedTimeMs = benchmarkKernel(optimizedKernel, matrixA, matrixB, matrixC, size, device, gridConfig);
        System.out.printf("Optimized kernel: %.3f ms%n", optimizedTimeMs);
        System.out.println();

        // Results
        double speedup = originalTimeMs / optimizedTimeMs;
        double improvement = ((originalTimeMs - optimizedTimeMs) / originalTimeMs) * 100;

        System.out.println("╔═══════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                   MCP OPTIMIZATION RESULTS                            ║");
        System.out.println("╠═══════════════════════════════════════════════════════════════════════╣");
        System.out.printf("║  Original kernel:  %.3f ms                                            ║%n", originalTimeMs);
        System.out.printf("║  Optimized kernel: %.3f ms                                            ║%n", optimizedTimeMs);
        System.out.println("╠═══════════════════════════════════════════════════════════════════════╣");
        if (speedup > 1.02) {
            System.out.printf("║  Speedup: %.2fx FASTER (%.1f%% improvement)                           ║%n", speedup, improvement);
        } else if (speedup < 0.98) {
            System.out.printf("║  Result: %.2fx SLOWER (%.1f%% regression)                             ║%n", speedup, -improvement);
        } else {
            System.out.println("║  Result: NO SIGNIFICANT CHANGE                                        ║");
        }
        System.out.println("╚═══════════════════════════════════════════════════════════════════════╝");
    }

    private static double benchmarkKernel(String kernelSource, FloatArray matrixA, FloatArray matrixB,
                                          FloatArray matrixC, int size, TornadoDevice device,
                                          GridConfigParser.GridConfig gridConfig) throws Exception {
        // Write kernel to temp file
        File tempFile = File.createTempFile("kernel_", ".cl");
        tempFile.deleteOnExit();
        writeFile(tempFile.getAbsolutePath(), kernelSource);

        // Setup kernel parameters
        AccessorParameters accessors = new AccessorParameters(4);
        accessors.set(0, matrixA, Access.READ_ONLY);
        accessors.set(1, matrixB, Access.READ_ONLY);
        accessors.set(2, matrixC, Access.WRITE_ONLY);
        accessors.set(3, Integer.valueOf(size), Access.NONE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA, matrixB)
                .prebuiltTask("t0", ENTRY_POINT, tempFile.getAbsolutePath(), accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, matrixC);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // Configure grid based on MCP response or defaults
        WorkerGrid worker;
        if (gridConfig != null && gridConfig.globalWorkSize != null && gridConfig.localWorkSize != null) {
            // Use grid config from MCP
            if (gridConfig.dimensions == 1) {
                worker = new WorkerGrid1D(gridConfig.globalWorkSize[0]);
                worker.setLocalWork(gridConfig.localWorkSize[0], 1, 1);
                System.out.println("Using MCP grid: " + gridConfig.globalWorkSize[0] + " global, " + gridConfig.localWorkSize[0] + " local (1D)");
            } else {
                worker = new WorkerGrid2D(gridConfig.globalWorkSize[0], gridConfig.globalWorkSize[1]);
                worker.setLocalWork(gridConfig.localWorkSize[0], gridConfig.localWorkSize[1], 1);
                System.out.println("Using MCP grid: " + gridConfig.globalWorkSize[0] + "x" + gridConfig.globalWorkSize[1] + " global, " + gridConfig.localWorkSize[0] + "x" + gridConfig.localWorkSize[1] + " local (2D)");
            }
        } else if (kernelSource.contains("reqd_work_group_size")) {
            // Fallback: extract from kernel attribute
            worker = new WorkerGrid2D(size, size);
            worker.setLocalWork(TS, TS, 1);
            System.out.println("Using fallback grid: " + size + "x" + size + " global, " + TS + "x" + TS + " local");
        } else {
            // Default: let TornadoVM decide
            worker = new WorkerGrid2D(size, size);
            System.out.println("Using default grid: " + size + "x" + size + " global");
        }
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        ArrayList<Long> kernelTimes = new ArrayList<>();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler);

            // Warmup
            System.out.println("Warming up (" + WARM_UP_ITERATIONS + " iterations)...");
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                plan.execute();
            }

            // Benchmark
            System.out.println("Benchmarking (" + BENCHMARK_ITERATIONS + " iterations)...");
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                TornadoExecutionResult result = plan
                        .withProfiler(ProfilerMode.SILENT)
                        .execute();

                TornadoProfilerResult profilerResult = result.getProfilerResult();
                long kernelTime = profilerResult.getDeviceKernelTime();
                kernelTimes.add(kernelTime);
            }
        }

        LongSummaryStatistics stats = kernelTimes.stream().mapToLong(Long::longValue).summaryStatistics();
        return stats.getAverage() / 1_000_000.0;
    }

    private static String callMCPServer(String serverUrl, String kernelCode, double kernelTimeMs) throws IOException {
        System.out.println("Calling MCP server at " + serverUrl + "...");

        URL url = new URL(serverUrl);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        conn.setConnectTimeout(10000);
        conn.setReadTimeout(180000); // 3 minutes for LLM

        // Build JSON request
        String json = String.format(
            "{\"kernel_code\": %s, \"backend\": \"opencl\", \"device_family\": \"apple_m4\", \"kernel_time_ns\": %d}",
            escapeJson(kernelCode),
            (long)(kernelTimeMs * 1_000_000)
        );

        try (OutputStream os = conn.getOutputStream()) {
            os.write(json.getBytes(StandardCharsets.UTF_8));
        }

        int responseCode = conn.getResponseCode();
        if (responseCode != 200) {
            String error = "";
            if (conn.getErrorStream() != null) {
                try (Scanner scanner = new Scanner(conn.getErrorStream(), StandardCharsets.UTF_8)) {
                    error = scanner.useDelimiter("\\A").hasNext() ? scanner.next() : "";
                }
            }
            throw new IOException("HTTP " + responseCode + ": " + error);
        }

        String response;
        try (Scanner scanner = new Scanner(conn.getInputStream(), StandardCharsets.UTF_8)) {
            response = scanner.useDelimiter("\\A").hasNext() ? scanner.next() : "";
        }

        // Store full response for grid config extraction
        lastMcpResponse = response;

        // Extract optimized_kernel from JSON
        return extractOptimizedKernel(response);
    }

    private static String extractOptimizedKernel(String json) {
        int start = json.indexOf("\"optimized_kernel\":");
        if (start == -1) return null;

        start = json.indexOf("\"", start + 19) + 1;
        if (start == 0) return null;

        int end = findEndOfJsonString(json, start);
        String kernel = json.substring(start, end);

        return kernel
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
    }

    private static int findEndOfJsonString(String json, int start) {
        boolean escaped = false;
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                return i;
            }
        }
        return json.length();
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
                    if (c < 32) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        sb.append("\"");
        return sb.toString();
    }

    private static void fillRandomData(FloatArray array) {
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, RANDOM.nextFloat() * 2.0f - 1.0f);
        }
    }

    private static String readFile(String path) throws IOException {
        return new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(path)));
    }

    private static void writeFile(String path, String content) throws IOException {
        java.nio.file.Files.write(java.nio.file.Paths.get(path), content.getBytes(StandardCharsets.UTF_8));
    }
}
