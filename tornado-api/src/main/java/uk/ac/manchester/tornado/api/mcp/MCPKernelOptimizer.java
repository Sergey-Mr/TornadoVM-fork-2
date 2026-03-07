/*
 * MCP Kernel Optimizer - HTTP Client for automatic kernel optimization
 *
 * This is called automatically by TornadoVM after the first kernel execution
 * to optimize the kernel using the MCP server.
 */
package uk.ac.manchester.tornado.api.mcp;

import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

public class MCPKernelOptimizer {

    private static final String DEFAULT_MCP_URL = "http://localhost:8090/optimize";
    private static final int CONNECT_TIMEOUT = 10000;  // 10 seconds
    private static final int READ_TIMEOUT = 180000;    // 3 minutes for LLM

    private final String mcpServerUrl;
    private final String deviceFamily;

    public MCPKernelOptimizer() {
        this.mcpServerUrl = System.getProperty("tornado.mcp.server.url", DEFAULT_MCP_URL);
        this.deviceFamily = detectDeviceFamily();
    }

    public MCPKernelOptimizer(String serverUrl, String deviceFamily) {
        this.mcpServerUrl = serverUrl;
        this.deviceFamily = deviceFamily;
    }

    /**
     * Check if MCP optimization is enabled via system property.
     */
    public static boolean isEnabled() {
        return Boolean.getBoolean("tornado.mcp.optimization");
    }

    /**
     * Optimize a kernel by calling the MCP HTTP server.
     *
     * @param kernelSource  The original kernel source code
     * @param backend       "opencl" or "ptx"
     * @param kernelTimeNs  Kernel execution time from profiler (nanoseconds)
     * @return Optimized kernel source, or null if optimization failed
     */
    public String optimize(String kernelSource, String backend, long kernelTimeNs) {
        if (kernelSource == null || kernelSource.isEmpty()) {
            return null;
        }

        try {
            System.out.println("[MCP] Sending kernel to " + mcpServerUrl + " for optimization...");
            System.out.println("[MCP] Backend: " + backend + ", Device: " + deviceFamily + ", Kernel time: " + kernelTimeNs + "ns");

            String optimized = callMCPServer(kernelSource, backend, deviceFamily, kernelTimeNs);

            if (optimized != null && !optimized.isEmpty()) {
                System.out.println("[MCP] Optimization successful! Received " + optimized.length() + " chars");
                return optimized;
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
     * Make HTTP POST request to MCP server.
     */
    private String callMCPServer(String kernelCode, String backend, String device, long kernelTimeNs)
            throws IOException {

        URL url = new URL(mcpServerUrl);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        conn.setConnectTimeout(CONNECT_TIMEOUT);
        conn.setReadTimeout(READ_TIMEOUT);

        // Build JSON request
        String jsonRequest = String.format(
                "{\"kernel_code\": %s, \"backend\": \"%s\", \"device_family\": \"%s\", \"kernel_time_ns\": %d}",
                escapeJson(kernelCode), backend, device, kernelTimeNs
        );

        try (OutputStream os = conn.getOutputStream()) {
            os.write(jsonRequest.getBytes(StandardCharsets.UTF_8));
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

        // Extract optimized_kernel from JSON response
        return extractOptimizedKernel(response);
    }

    /**
     * Extract the optimized_kernel field from JSON response.
     */
    private String extractOptimizedKernel(String json) {
        int start = json.indexOf("\"optimized_kernel\":");
        if (start == -1) {
            return null;
        }

        // Find the opening quote of the value
        start = json.indexOf("\"", start + 19) + 1;
        if (start == 0) {
            return null;
        }

        // Find the closing quote (handling escapes)
        int end = findEndOfJsonString(json, start);

        String kernel = json.substring(start, end);

        // Unescape JSON string
        return kernel
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
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
