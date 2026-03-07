/*
 * Full MCP HTTP Integration Test
 *
 * This demonstrates the complete workflow:
 * 1. Run original kernel with TornadoVM
 * 2. Extract kernel source AFTER execution
 * 3. Call MCP server via HTTP for optimization
 * 4. Replace kernel with optimized version
 * 5. Re-run and compare performance
 *
 * Prerequisites:
 * - MCP server running: cd MCP-server && python -m tornadovm_mcp.http_server
 */
package uk.ac.manchester.tornado.examples.compute.custom;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class MCPHttpIntegrationTest {

    private static final String MCP_SERVER_URL = "http://localhost:8090/optimize";
    private static final int WARMUP_ITERATIONS = 10;
    private static final int BENCHMARK_ITERATIONS = 20;

    // Simple matrix-vector multiplication kernel
    public static void matrixVectorMul(FloatArray matrix, FloatArray vector, FloatArray result, int size) {
        for (@Parallel int i = 0; i < size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < size; j++) {
                sum += matrix.get(i * size + j) * vector.get(j);
            }
            result.set(i, sum);
        }
    }

    public static void main(String[] args) {
        final int size = 512;

        // Initialize data
        FloatArray matrix = new FloatArray(size * size);
        FloatArray vector = new FloatArray(size);
        FloatArray result = new FloatArray(size);

        for (int i = 0; i < size * size; i++) {
            matrix.set(i, (float) (i % 100) / 100.0f);
        }
        for (int i = 0; i < size; i++) {
            vector.set(i, (float) i / size);
        }

        // Create TaskGraph
        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrix, vector)
                .task("t0", MCPHttpIntegrationTest::matrixVectorMul, matrix, vector, result, size)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, result);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);

        System.out.println("=== MCP HTTP Integration Test ===\n");
        System.out.println("Matrix size: " + size + "x" + size);

        // ============================================================
        // PHASE 1: Run original kernel and collect profiling data
        // ============================================================
        System.out.println("\n--- PHASE 1: Original Kernel ---");

        // Warmup
        System.out.println("Warming up (" + WARMUP_ITERATIONS + " iterations)...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            executor.execute();
        }

        // Benchmark original kernel
        List<Long> originalTimes = new ArrayList<>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            TornadoExecutionResult execResult = executor.withProfiler(ProfilerMode.SILENT).execute();
            long kernelTime = execResult.getProfilerResult().getDeviceKernelTime();
            originalTimes.add(kernelTime);
        }

        double originalAvg = originalTimes.stream().mapToLong(Long::longValue).average().orElse(0) / 1_000_000.0;
        System.out.printf("Original kernel avg time: %.3f ms%n", originalAvg);

        // ============================================================
        // PHASE 2: Extract kernel source
        // ============================================================
        System.out.println("\n--- PHASE 2: Extract Kernel Source ---");

        String kernelSource = executor.getGeneratedKernelSource("t0");
        if (kernelSource == null) {
            System.err.println("ERROR: Could not extract kernel source!");
            return;
        }

        System.out.println("Kernel extracted successfully (" + kernelSource.length() + " chars)");
        System.out.println("\nFirst 500 chars of kernel:");
        System.out.println(kernelSource.substring(0, Math.min(500, kernelSource.length())));
        System.out.println("...\n");

        // ============================================================
        // PHASE 3: Call MCP server for optimization
        // ============================================================
        System.out.println("--- PHASE 3: Call MCP Server ---");

        String optimizedKernel = null;
        try {
            optimizedKernel = callMCPServer(kernelSource, "opencl", "apple_m4",
                    (long) (originalAvg * 1_000_000));  // Convert ms to ns

            if (optimizedKernel != null) {
                System.out.println("MCP optimization successful!");
                System.out.println("Optimized kernel length: " + optimizedKernel.length() + " chars");
            }
        } catch (IOException e) {
            System.err.println("MCP server call failed: " + e.getMessage());
            System.err.println("\nMake sure the MCP server is running:");
            System.err.println("  cd /Users/serhiitupikin/Documents/Coding/TornadoVM_MCP/MCP-server");
            System.err.println("  source .venv/bin/activate");
            System.err.println("  python -m tornadovm_mcp.http_server");
            return;
        }

        if (optimizedKernel == null || optimizedKernel.isEmpty()) {
            System.err.println("ERROR: MCP returned empty optimization!");
            return;
        }

        // ============================================================
        // PHASE 4: Replace kernel and re-run
        // ============================================================
        System.out.println("\n--- PHASE 4: Run Optimized Kernel ---");

        boolean replaced = executor.replaceKernelSource("t0", optimizedKernel);
        if (!replaced) {
            System.err.println("ERROR: Kernel replacement failed!");
            return;
        }
        System.out.println("Kernel replaced successfully.");

        // Reset result array
        for (int i = 0; i < size; i++) {
            result.set(i, 0);
        }

        // Warmup optimized kernel
        System.out.println("Warming up optimized kernel...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            executor.execute();
        }

        // Benchmark optimized kernel
        List<Long> optimizedTimes = new ArrayList<>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            TornadoExecutionResult execResult = executor.withProfiler(ProfilerMode.SILENT).execute();
            long kernelTime = execResult.getProfilerResult().getDeviceKernelTime();
            optimizedTimes.add(kernelTime);
        }

        double optimizedAvg = optimizedTimes.stream().mapToLong(Long::longValue).average().orElse(0) / 1_000_000.0;
        System.out.printf("Optimized kernel avg time: %.3f ms%n", optimizedAvg);

        // ============================================================
        // PHASE 5: Compare results
        // ============================================================
        System.out.println("\n--- RESULTS ---");
        System.out.printf("Original:  %.3f ms%n", originalAvg);
        System.out.printf("Optimized: %.3f ms%n", optimizedAvg);

        double speedup = originalAvg / optimizedAvg;
        if (speedup > 1.0) {
            System.out.printf("Speedup:   %.2fx FASTER%n", speedup);
        } else {
            System.out.printf("Speedup:   %.2fx (slower)%n", speedup);
        }

        System.out.println("\n=== Test Complete ===");
    }

    /**
     * Call the MCP HTTP server to optimize a kernel.
     */
    private static String callMCPServer(String kernelCode, String backend, String deviceFamily, long kernelTimeNs)
            throws IOException {

        System.out.println("Calling MCP server at " + MCP_SERVER_URL + "...");

        URL url = new URL(MCP_SERVER_URL);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        conn.setConnectTimeout(60000);  // 60 second timeout
        conn.setReadTimeout(120000);    // 120 second read timeout (LLM can be slow)

        // Build JSON request (manual to avoid dependencies)
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
                throw new IOException("MCP server error (" + responseCode + "): " + error);
            }
        }

        // Read response
        String response;
        try (Scanner scanner = new Scanner(conn.getInputStream(), StandardCharsets.UTF_8)) {
            response = scanner.useDelimiter("\\A").next();
        }

        // Extract optimized_kernel from JSON (simple parsing)
        int start = response.indexOf("\"optimized_kernel\":");
        if (start == -1) {
            System.err.println("Response does not contain optimized_kernel: " + response.substring(0, Math.min(500, response.length())));
            return null;
        }

        // Find the string value after "optimized_kernel":
        start = response.indexOf("\"", start + 19) + 1;
        int end = findEndOfJsonString(response, start);

        String optimizedKernel = response.substring(start, end);
        // Unescape JSON string
        optimizedKernel = optimizedKernel
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");

        return optimizedKernel;
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
}
