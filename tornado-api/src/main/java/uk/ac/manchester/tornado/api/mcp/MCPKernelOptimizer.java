/*
 * MCP Kernel Optimizer - HTTP Client for automatic kernel optimization
 *
 * This is called automatically by TornadoVM after the first kernel execution
 * to optimize the kernel using the MCP server.
 *
 * Supports iterative feedback loop: if optimized kernel is slower,
 * retry with feedback up to MAX_ATTEMPTS times.
 */
package uk.ac.manchester.tornado.api.mcp;

import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;

public class MCPKernelOptimizer {

    private static final String DEFAULT_MCP_URL = "http://localhost:8090/optimize";
    private static final String TEST_MCP_URL = "http://localhost:8090/optimize-test";
    private static final int CONNECT_TIMEOUT = 10000;  // 10 seconds
    private static final int READ_TIMEOUT = 180000;    // 3 minutes for LLM
    private static final int MAX_ATTEMPTS = 3;         // Maximum optimization attempts

    private final String mcpServerUrl;
    private final String deviceFamily;
    private final boolean testMode;

    /**
     * Record for tracking a previous optimization attempt that failed or was slower.
     */
    public record PreviousAttempt(
            String optimizedKernel,
            double originalTimeMs,
            double optimizedTimeMs,
            int attemptNumber,
            String validationError  // null if validation passed, error message if failed
    ) {
        // Constructor without validation error for backwards compatibility
        public PreviousAttempt(String optimizedKernel, double originalTimeMs, double optimizedTimeMs, int attemptNumber) {
            this(optimizedKernel, originalTimeMs, optimizedTimeMs, attemptNumber, null);
        }
    }

    /**
     * Grid configuration for launching the optimized kernel.
     * Supports 1D, 2D, and 3D grids with optional pattern hints.
     */
    public record GridConfig(
            int dimensions,           // 1, 2, or 3
            String[] globalWorkSize,  // Parameter names or expressions like ["size", "size"] or ["numBodies"] or ["D * 32"]
            int[] localWorkSize,      // Concrete values like [16, 16] or [256] or [8, 8, 4]
            String pattern            // Optional: "default", "reduction", "stencil", "tiled" (nullable)
    ) {
        // Constructor without pattern for backwards compatibility
        public GridConfig(int dimensions, String[] globalWorkSize, int[] localWorkSize) {
            this(dimensions, globalWorkSize, localWorkSize, null);
        }
    }

    /**
     * Device capabilities from OpenCL/CUDA query.
     */
    public record DeviceCapabilities(
            int maxWorkGroupSize,
            int[] maxWorkItemSizes,
            long localMemorySize,
            int computeUnits,
            String deviceName,
            int simdWidth  // 0 if unknown
    ) {}

    /**
     * Execution context with actual parameter values.
     */
    public record ExecutionContext(
            int[] globalWorkSize,
            int[] localWorkSize,
            java.util.Map<String, Integer> parameterValues
    ) {}

    /**
     * Result of an optimization attempt.
     */
    public record OptimizationResult(
            String optimizedKernel,
            int attemptNumber,
            boolean success,
            double optimizedTimeMs,  // Store the benchmarked time to avoid re-benchmarking
            GridConfig gridConfig    // Grid configuration from LLM (nullable)
    ) {}

    /**
     * Result of benchmarking an optimized kernel.
     * Includes both timing and validation status.
     */
    public record BenchmarkResult(
            double timeMs,           // Execution time in milliseconds (Double.MAX_VALUE if failed)
            String validationError   // null if validation passed, error message if failed
    ) {
        public boolean isValid() {
            return validationError == null && timeMs < Double.MAX_VALUE;
        }
    }

    public MCPKernelOptimizer() {
        this.testMode = Boolean.getBoolean("tornado.mcp.test");
        String defaultUrl = testMode ? TEST_MCP_URL : DEFAULT_MCP_URL;
        this.mcpServerUrl = System.getProperty("tornado.mcp.server.url", defaultUrl);
        this.deviceFamily = detectDeviceFamily();
        if (testMode) {
            System.out.println("[MCP] TEST MODE: Using single-call endpoint with known-good example");
        }
    }

    public MCPKernelOptimizer(String serverUrl, String deviceFamily) {
        this.mcpServerUrl = serverUrl;
        this.deviceFamily = deviceFamily;
        this.testMode = false;
    }

    /**
     * Check if MCP optimization is enabled via system property.
     */
    public static boolean isEnabled() {
        return Boolean.getBoolean("tornado.mcp.optimization");
    }

    /**
     * Optimize a kernel by calling the MCP HTTP server (single attempt, no retry).
     *
     * @param kernelSource  The original kernel source code
     * @param backend       "opencl" or "ptx"
     * @param kernelTimeNs  Kernel execution time from profiler (nanoseconds)
     * @return Optimized kernel source, or null if optimization failed
     */
    public String optimize(String kernelSource, String backend, long kernelTimeNs) {
        return optimize(kernelSource, backend, kernelTimeNs, null, null);
    }

    /**
     * Optimize a kernel with device capabilities and execution context.
     *
     * @param kernelSource        The original kernel source code
     * @param backend             "opencl" or "ptx"
     * @param kernelTimeNs        Kernel execution time from profiler (nanoseconds)
     * @param deviceCapabilities  Device capabilities from OpenCL query (nullable)
     * @param executionContext    Execution context with work sizes and parameter values (nullable)
     * @return Optimized kernel source, or null if optimization failed
     */
    public String optimize(String kernelSource, String backend, long kernelTimeNs,
                          DeviceCapabilities deviceCapabilities, ExecutionContext executionContext) {
        if (kernelSource == null || kernelSource.isEmpty()) {
            return null;
        }

        try {
            System.out.println("[MCP] Sending kernel to " + mcpServerUrl + " for optimization...");
            System.out.println("[MCP] Backend: " + backend + ", Device: " + deviceFamily + ", Kernel time: " + kernelTimeNs + "ns");
            if (deviceCapabilities != null) {
                System.out.println("[MCP] Device: " + deviceCapabilities.deviceName() +
                        ", MaxWorkGroup: " + deviceCapabilities.maxWorkGroupSize() +
                        ", LocalMem: " + deviceCapabilities.localMemorySize() + " bytes");
            }
            if (executionContext != null && executionContext.parameterValues() != null) {
                System.out.println("[MCP] Parameter values: " + executionContext.parameterValues());
            }

            MCPResponse response = callMCPServer(kernelSource, backend, deviceFamily, kernelTimeNs,
                    null, deviceCapabilities, executionContext);

            if (response != null && response.kernel() != null && !response.kernel().isEmpty()) {
                System.out.println("[MCP] Optimization successful! Received " + response.kernel().length() + " chars");
                return response.kernel();
            } else {
                System.out.println("[MCP] Optimization returned empty result");
                return null;
            }
        } catch (IOException e) {
            System.err.println("[MCP] Optimization failed: " + e.getMessage());
            return null;
        }
    }

    /**
     * Optimize a kernel with iterative feedback loop.
     *
     * If the optimized kernel is slower than the original, retry with feedback
     * up to MAX_ATTEMPTS times. Includes validation feedback for incorrect results.
     *
     * @param kernelSource     The original kernel source code
     * @param backend          "opencl" or "ptx"
     * @param originalTimeMs   Original kernel execution time in milliseconds
     * @param benchmarkFunc    Function to benchmark a kernel with grid config, returns BenchmarkResult with timing and validation
     * @return OptimizationResult with the best kernel found
     */
    public OptimizationResult optimizeWithFeedback(
            String kernelSource,
            String backend,
            double originalTimeMs,
            java.util.function.BiFunction<String, GridConfig, BenchmarkResult> benchmarkFunc) {

        if (kernelSource == null || kernelSource.isEmpty()) {
            return new OptimizationResult(kernelSource, 0, false, originalTimeMs, null);
        }

        // Minimum improvement threshold (2%) to count as success
        // This prevents false positives from measurement noise
        final double MIN_SPEEDUP_THRESHOLD = 0.98;  // optimizedTime must be <= 98% of original

        // In test mode, only 1 attempt (testing benchmarking with known-good kernel)
        final int maxAttempts = testMode ? 1 : MAX_ATTEMPTS;

        List<PreviousAttempt> previousAttempts = new ArrayList<>();
        String bestKernel = kernelSource;
        double bestTimeMs = originalTimeMs;
        GridConfig bestGridConfig = null;

        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                System.out.println("[MCP] Optimization attempt " + attempt + "/" + MAX_ATTEMPTS);
                System.out.println("[MCP] Sending kernel to " + mcpServerUrl + " for optimization...");
                System.out.println("[MCP] Backend: " + backend + ", Device: " + deviceFamily);

                if (!previousAttempts.isEmpty()) {
                    System.out.println("[MCP] Including " + previousAttempts.size() + " previous failed attempt(s) as feedback");
                }

                // Call MCP server with previous attempts if any
                MCPResponse mcpResponse = callMCPServer(
                        kernelSource,
                        backend,
                        deviceFamily,
                        (long) (originalTimeMs * 1_000_000),
                        previousAttempts.isEmpty() ? null : previousAttempts
                );

                if (mcpResponse == null || mcpResponse.kernel() == null || mcpResponse.kernel().isEmpty()) {
                    System.out.println("[MCP] Attempt " + attempt + " returned empty result");
                    continue;
                }

                String optimized = mcpResponse.kernel();
                GridConfig gridConfig = mcpResponse.gridConfig();

                System.out.println("[MCP] Received optimized kernel (" + optimized.length() + " chars)");
                if (gridConfig != null) {
                    System.out.printf("[MCP] Grid config: %dD, global=%s, local=%s%n",
                            gridConfig.dimensions(),
                            java.util.Arrays.toString(gridConfig.globalWorkSize()),
                            java.util.Arrays.toString(gridConfig.localWorkSize()));
                }

                // Benchmark the optimized kernel
                System.out.println("[MCP] Benchmarking optimized kernel...");
                BenchmarkResult benchmarkResult = benchmarkFunc.apply(optimized, gridConfig);
                double optimizedTimeMs = benchmarkResult.timeMs();
                String validationError = benchmarkResult.validationError();

                // Check for validation failure first
                if (validationError != null) {
                    System.out.printf("[MCP] ✗ Attempt %d VALIDATION FAILED: %s%n", attempt, validationError);
                    // Track this failed attempt with validation error for feedback
                    previousAttempts.add(new PreviousAttempt(
                            optimized,
                            originalTimeMs,
                            optimizedTimeMs,
                            attempt,
                            validationError
                    ));
                    if (attempt < maxAttempts) {
                        System.out.println("[MCP] Retrying with validation error feedback...");
                    }
                    continue;  // Skip to next attempt
                }

                double speedup = originalTimeMs / optimizedTimeMs;
                // Must be at least 2% faster to count as success (handles measurement noise)
                boolean isFaster = optimizedTimeMs <= (originalTimeMs * MIN_SPEEDUP_THRESHOLD);

                // Track best result seen (only if validation passed)
                if (optimizedTimeMs < bestTimeMs) {
                    bestKernel = optimized;
                    bestTimeMs = optimizedTimeMs;
                    bestGridConfig = gridConfig;
                }

                if (isFaster) {
                    double improvement = ((originalTimeMs - optimizedTimeMs) / originalTimeMs) * 100;
                    System.out.printf("[MCP] ✓ Attempt %d SUCCESS: %.3f ms → %.3f ms (%.2fx speedup, %.1f%% faster)%n",
                            attempt, originalTimeMs, optimizedTimeMs, speedup, improvement);
                    return new OptimizationResult(optimized, attempt, true, optimizedTimeMs, gridConfig);
                } else {
                    double diff = ((optimizedTimeMs - originalTimeMs) / originalTimeMs) * 100;
                    if (diff >= 0) {
                        System.out.printf("[MCP] ✗ Attempt %d NOT FASTER: %.3f ms → %.3f ms (%.1f%% slower)%n",
                                attempt, originalTimeMs, optimizedTimeMs, diff);
                    } else {
                        System.out.printf("[MCP] ✗ Attempt %d MARGINAL: %.3f ms → %.3f ms (%.1f%% faster, below 2%% threshold)%n",
                                attempt, originalTimeMs, optimizedTimeMs, -diff);
                    }

                    // Track this failed attempt for feedback (no validation error)
                    previousAttempts.add(new PreviousAttempt(
                            optimized,
                            originalTimeMs,
                            optimizedTimeMs,
                            attempt,
                            null  // No validation error
                    ));

                    if (attempt < maxAttempts) {
                        System.out.println("[MCP] Retrying with feedback about what didn't work...");
                    }
                }

            } catch (IOException e) {
                System.err.println("[MCP] Attempt " + attempt + " failed: " + e.getMessage());
            }
        }

        System.out.println("[MCP] All " + maxAttempts + " attempts completed without significant improvement");
        return new OptimizationResult(bestKernel, maxAttempts, false, bestTimeMs, bestGridConfig);
    }

    /**
     * Make HTTP POST request to MCP server (legacy signature for backwards compatibility).
     */
    private MCPResponse callMCPServer(
            String kernelCode,
            String backend,
            String device,
            long kernelTimeNs,
            List<PreviousAttempt> previousAttempts) throws IOException {
        return callMCPServer(kernelCode, backend, device, kernelTimeNs, previousAttempts, null, null);
    }

    /**
     * Make HTTP POST request to MCP server with full context.
     */
    private MCPResponse callMCPServer(
            String kernelCode,
            String backend,
            String device,
            long kernelTimeNs,
            List<PreviousAttempt> previousAttempts,
            DeviceCapabilities deviceCapabilities,
            ExecutionContext executionContext) throws IOException {

        URL url = new URL(mcpServerUrl);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        conn.setConnectTimeout(CONNECT_TIMEOUT);
        conn.setReadTimeout(READ_TIMEOUT);

        // Build JSON request
        StringBuilder json = new StringBuilder();
        json.append("{");
        json.append("\"kernel_code\": ").append(escapeJson(kernelCode)).append(", ");
        json.append("\"backend\": \"").append(backend).append("\", ");
        json.append("\"device_family\": \"").append(device).append("\", ");
        json.append("\"kernel_time_ns\": ").append(kernelTimeNs);

        // Add device capabilities if provided
        if (deviceCapabilities != null) {
            json.append(", \"max_work_group_size\": ").append(deviceCapabilities.maxWorkGroupSize());
            if (deviceCapabilities.maxWorkItemSizes() != null && deviceCapabilities.maxWorkItemSizes().length > 0) {
                json.append(", \"max_work_item_sizes\": [");
                for (int i = 0; i < deviceCapabilities.maxWorkItemSizes().length; i++) {
                    if (i > 0) json.append(", ");
                    json.append(deviceCapabilities.maxWorkItemSizes()[i]);
                }
                json.append("]");
            }
            json.append(", \"local_memory_size\": ").append(deviceCapabilities.localMemorySize());
            json.append(", \"compute_units\": ").append(deviceCapabilities.computeUnits());
            if (deviceCapabilities.deviceName() != null) {
                json.append(", \"device_name\": \"").append(deviceCapabilities.deviceName().replace("\"", "\\\"")).append("\"");
            }
            if (deviceCapabilities.simdWidth() > 0) {
                json.append(", \"simd_width\": ").append(deviceCapabilities.simdWidth());
            }
        }

        // Add execution context if provided
        if (executionContext != null) {
            if (executionContext.globalWorkSize() != null && executionContext.globalWorkSize().length > 0) {
                json.append(", \"global_work_size\": [");
                for (int i = 0; i < executionContext.globalWorkSize().length; i++) {
                    if (i > 0) json.append(", ");
                    json.append(executionContext.globalWorkSize()[i]);
                }
                json.append("]");
            }
            if (executionContext.localWorkSize() != null && executionContext.localWorkSize().length > 0) {
                json.append(", \"local_work_size\": [");
                for (int i = 0; i < executionContext.localWorkSize().length; i++) {
                    if (i > 0) json.append(", ");
                    json.append(executionContext.localWorkSize()[i]);
                }
                json.append("]");
            }
            if (executionContext.parameterValues() != null && !executionContext.parameterValues().isEmpty()) {
                json.append(", \"parameter_values\": {");
                boolean first = true;
                for (var entry : executionContext.parameterValues().entrySet()) {
                    if (!first) json.append(", ");
                    first = false;
                    json.append("\"").append(entry.getKey()).append("\": ").append(entry.getValue());
                }
                json.append("}");
            }
        }

        // Add previous attempts if any
        if (previousAttempts != null && !previousAttempts.isEmpty()) {
            json.append(", \"previous_attempts\": [");
            for (int i = 0; i < previousAttempts.size(); i++) {
                PreviousAttempt attempt = previousAttempts.get(i);
                if (i > 0) json.append(", ");
                json.append("{");
                json.append("\"optimized_kernel\": ").append(escapeJson(attempt.optimizedKernel())).append(", ");
                json.append("\"original_time_ms\": ").append(attempt.originalTimeMs()).append(", ");
                json.append("\"optimized_time_ms\": ").append(attempt.optimizedTimeMs()).append(", ");
                json.append("\"attempt_number\": ").append(attempt.attemptNumber());
                // Include validation error if present
                if (attempt.validationError() != null) {
                    json.append(", \"validation_error\": ").append(escapeJson(attempt.validationError()));
                }
                json.append("}");
            }
            json.append("]");
        }

        json.append("}");

        try (OutputStream os = conn.getOutputStream()) {
            os.write(json.toString().getBytes(StandardCharsets.UTF_8));
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

        // Read response
        String response;
        try (Scanner scanner = new Scanner(conn.getInputStream(), StandardCharsets.UTF_8)) {
            response = scanner.useDelimiter("\\A").hasNext() ? scanner.next() : "";
        }

        // Extract optimized_kernel and grid_config from JSON response
        return extractMCPResponse(response);
    }

    /**
     * Internal result from MCP server containing kernel and grid config.
     */
    private record MCPResponse(String kernel, GridConfig gridConfig) {}

    /**
     * Extract optimized_kernel and grid_config from JSON response.
     */
    private MCPResponse extractMCPResponse(String json) {
        String kernel = extractJsonString(json, "optimized_kernel");
        GridConfig gridConfig = extractGridConfig(json);
        return new MCPResponse(kernel, gridConfig);
    }

    /**
     * Extract a string field from JSON.
     */
    private String extractJsonString(String json, String fieldName) {
        int start = json.indexOf("\"" + fieldName + "\":");
        if (start == -1) {
            return null;
        }

        // Find the opening quote of the value
        start = json.indexOf("\"", start + fieldName.length() + 3) + 1;
        if (start == 0) {
            return null;
        }

        // Find the closing quote (handling escapes)
        int end = findEndOfJsonString(json, start);

        String value = json.substring(start, end);

        // Unescape JSON string
        return value
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
    }

    /**
     * Extract grid_config from JSON response.
     */
    private GridConfig extractGridConfig(String json) {
        // Find grid_config object
        int start = json.indexOf("\"grid_config\":");
        if (start == -1) {
            return null;
        }

        // Find the opening brace
        start = json.indexOf("{", start);
        if (start == -1) {
            return null;
        }

        // Find matching closing brace
        int braceCount = 1;
        int end = start + 1;
        while (end < json.length() && braceCount > 0) {
            char c = json.charAt(end);
            if (c == '{') braceCount++;
            else if (c == '}') braceCount--;
            end++;
        }

        String configJson = json.substring(start, end);

        try {
            // Parse dimensions (1, 2, or 3)
            int dimensions = extractJsonInt(configJson, "dimensions");

            // Parse global_work_size array (strings - parameter names or expressions)
            String[] globalWorkSize = extractJsonStringArray(configJson, "global_work_size");

            // Parse local_work_size array (ints)
            int[] localWorkSize = extractJsonIntArray(configJson, "local_work_size");

            // Parse optional pattern field
            String pattern = extractJsonStringValue(configJson, "pattern");

            if (dimensions > 0 && globalWorkSize != null && localWorkSize != null) {
                return new GridConfig(dimensions, globalWorkSize, localWorkSize, pattern);
            }
        } catch (Exception e) {
            System.err.println("[MCP] Failed to parse grid_config: " + e.getMessage());
        }

        return null;
    }

    /**
     * Extract a simple string value from JSON (not an array).
     */
    private String extractJsonStringValue(String json, String fieldName) {
        String pattern = "\"" + fieldName + "\":\\s*\"([^\"]+)\"";
        java.util.regex.Matcher m = java.util.regex.Pattern.compile(pattern).matcher(json);
        if (m.find()) {
            return m.group(1);
        }
        return null;
    }

    private int extractJsonInt(String json, String fieldName) {
        String pattern = "\"" + fieldName + "\":\\s*(\\d+)";
        java.util.regex.Matcher m = java.util.regex.Pattern.compile(pattern).matcher(json);
        if (m.find()) {
            return Integer.parseInt(m.group(1));
        }
        return -1;
    }

    private String[] extractJsonStringArray(String json, String fieldName) {
        String pattern = "\"" + fieldName + "\":\\s*\\[([^\\]]+)\\]";
        java.util.regex.Matcher m = java.util.regex.Pattern.compile(pattern).matcher(json);
        if (m.find()) {
            String arrayContent = m.group(1);
            // Extract quoted strings
            java.util.List<String> values = new ArrayList<>();
            java.util.regex.Matcher valueMatcher = java.util.regex.Pattern.compile("\"([^\"]+)\"").matcher(arrayContent);
            while (valueMatcher.find()) {
                values.add(valueMatcher.group(1));
            }
            return values.toArray(new String[0]);
        }
        return null;
    }

    private int[] extractJsonIntArray(String json, String fieldName) {
        String pattern = "\"" + fieldName + "\":\\s*\\[([^\\]]+)\\]";
        java.util.regex.Matcher m = java.util.regex.Pattern.compile(pattern).matcher(json);
        if (m.find()) {
            String arrayContent = m.group(1);
            // Extract integers
            java.util.List<Integer> values = new ArrayList<>();
            java.util.regex.Matcher valueMatcher = java.util.regex.Pattern.compile("(\\d+)").matcher(arrayContent);
            while (valueMatcher.find()) {
                values.add(Integer.parseInt(valueMatcher.group(1)));
            }
            return values.stream().mapToInt(Integer::intValue).toArray();
        }
        return null;
    }

    private int findEndOfJsonString(String json, int start) {
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

    private String escapeJson(String s) {
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

    /**
     * Detect device family from system or environment.
     */
    private String detectDeviceFamily() {
        String configured = System.getProperty("tornado.mcp.device.family");
        if (configured != null && !configured.isEmpty()) {
            return configured;
        }

        // Auto-detect based on OS/architecture
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();

        if (os.contains("mac") && arch.contains("aarch64")) {
            return "apple_m4";  // Default to M4 for Apple Silicon
        } else if (os.contains("linux")) {
            return "nvidia_ada";  // Default to Ada for Linux (assume RTX 40 series)
        }

        return "generic";
    }
}
