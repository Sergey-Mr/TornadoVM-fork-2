/*
 * TornadoVM MCP Kernel Optimizer Client
 *
 * Communicates with the MCP server over stdio using JSON-RPC protocol
 * to optimize generated OpenCL/PTX kernels.
 */
package uk.ac.manchester.tornado.api.mcp;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Map;
import java.util.HashMap;
import java.util.List;

/**
 * Client for the TornadoVM MCP Kernel Optimizer server.
 *
 * Usage:
 * <pre>
 * MCPKernelOptimizer optimizer = new MCPKernelOptimizer();
 * optimizer.start();
 *
 * String optimizedKernel = optimizer.optimizeKernel(
 *     kernelCode,
 *     "opencl",
 *     "nvidia_ada",
 *     kernelTimeNs,
 *     copyInTimeNs,
 *     copyOutTimeNs,
 *     copyInBytes,
 *     copyOutBytes,
 *     new int[]{1024, 1024},
 *     new int[]{16, 16}
 * );
 *
 * optimizer.stop();
 * </pre>
 */
public class MCPKernelOptimizer {

    private static final String MCP_SERVER_COMMAND = System.getProperty(
        "tornado.mcp.server.command",
        "python -m tornadovm_mcp.server"
    );

    private static final String MCP_SERVER_PATH = System.getProperty(
        "tornado.mcp.server.path",
        System.getenv("TORNADOVM_MCP_PATH")
    );

    private static final boolean MCP_ENABLED = Boolean.parseBoolean(
        System.getProperty("tornado.mcp.optimization", "false")
    );

    private Process serverProcess;
    private BufferedWriter serverInput;
    private BufferedReader serverOutput;
    private BufferedReader serverError;
    private final AtomicInteger requestId = new AtomicInteger(0);
    private boolean initialized = false;
    private Thread shutdownHook;

    /**
     * Check if MCP optimization is enabled via system property.
     */
    public static boolean isEnabled() {
        return MCP_ENABLED;
    }

    /**
     * Start the MCP server process.
     */
    public void start() throws IOException {
        if (serverProcess != null && serverProcess.isAlive()) {
            return; // Already running
        }

        ProcessBuilder pb;
        if (MCP_SERVER_PATH != null && !MCP_SERVER_PATH.isEmpty()) {
            // Use virtual environment
            String pythonPath = MCP_SERVER_PATH + "/.venv/bin/python";
            pb = new ProcessBuilder(pythonPath, "-m", "tornadovm_mcp.server");
            pb.directory(new File(MCP_SERVER_PATH));

            // Set PYTHONPATH
            Map<String, String> env = pb.environment();
            env.put("PYTHONPATH", MCP_SERVER_PATH + "/src");
        } else {
            // Use system Python
            String[] cmd = MCP_SERVER_COMMAND.split("\\s+");
            pb = new ProcessBuilder(cmd);
        }

        pb.redirectErrorStream(false);
        serverProcess = pb.start();

        serverInput = new BufferedWriter(
            new OutputStreamWriter(serverProcess.getOutputStream(), StandardCharsets.UTF_8)
        );
        serverOutput = new BufferedReader(
            new InputStreamReader(serverProcess.getInputStream(), StandardCharsets.UTF_8)
        );
        serverError = new BufferedReader(
            new InputStreamReader(serverProcess.getErrorStream(), StandardCharsets.UTF_8)
        );

        // Start error reader thread
        startErrorReader();

        // Initialize MCP session
        initialize();

        // Register shutdown hook for graceful cleanup
        shutdownHook = new Thread(() -> {
            System.out.println("[TornadoVM-MCP] Shutting down MCP server...");
            stop();
        }, "MCP-Shutdown-Hook");
        Runtime.getRuntime().addShutdownHook(shutdownHook);
    }

    /**
     * Initialize MCP session with handshake.
     */
    private void initialize() throws IOException {
        // Send initialize request
        String initRequest = createJsonRpcRequest("initialize", Map.of(
            "protocolVersion", "2024-11-05",
            "capabilities", Map.of(),
            "clientInfo", Map.of(
                "name", "TornadoVM",
                "version", "1.0.0"
            )
        ));

        sendRequest(initRequest);
        String response = readResponse();

        // Send initialized notification
        String initializedNotification = String.format(
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n"
        );
        serverInput.write(initializedNotification);
        serverInput.flush();

        initialized = true;
    }

    /**
     * Optimize a kernel using the MCP server.
     *
     * @param kernelCode The generated kernel source code
     * @param backend "opencl" or "ptx"
     * @param deviceFamily Target device (e.g., "nvidia_ada", "apple_m4")
     * @param kernelTimeNs Kernel execution time in nanoseconds
     * @param copyInTimeNs Copy-in time in nanoseconds
     * @param copyOutTimeNs Copy-out time in nanoseconds
     * @param copyInBytes Bytes transferred to device
     * @param copyOutBytes Bytes transferred from device
     * @param globalWorkSize Global work size [x, y, z]
     * @param localWorkSize Local work size [x, y, z]
     * @return Optimized kernel code, or original if optimization fails
     */
    public String optimizeKernel(
            String kernelCode,
            String backend,
            String deviceFamily,
            long kernelTimeNs,
            long copyInTimeNs,
            long copyOutTimeNs,
            long copyInBytes,
            long copyOutBytes,
            int[] globalWorkSize,
            int[] localWorkSize
    ) {
        if (!initialized) {
            System.err.println("[MCP] Server not initialized, returning original kernel");
            return kernelCode;
        }

        try {
            Map<String, Object> args = new HashMap<>();
            args.put("kernel_code", kernelCode);
            args.put("backend", backend);
            args.put("device_family", deviceFamily);
            args.put("kernel_time_ns", kernelTimeNs);
            args.put("copy_in_time_ns", copyInTimeNs);
            args.put("copy_out_time_ns", copyOutTimeNs);
            args.put("copy_in_bytes", copyInBytes);
            args.put("copy_out_bytes", copyOutBytes);
            args.put("global_work_size", arrayToList(globalWorkSize));
            args.put("local_work_size", arrayToList(localWorkSize));

            String request = createJsonRpcRequest("tools/call", Map.of(
                "name", "optimize_tornadovm_kernel",
                "arguments", args
            ));

            sendRequest(request);
            String response = readResponse();

            // Parse response to extract optimized kernel
            String optimizedKernel = parseOptimizedKernel(response);

            if (optimizedKernel != null && !optimizedKernel.isEmpty()) {
                System.out.println("[MCP] Kernel optimization successful");
                return optimizedKernel;
            } else {
                System.err.println("[MCP] Could not extract optimized kernel from response");
                return kernelCode;
            }

        } catch (Exception e) {
            System.err.println("[MCP] Optimization failed: " + e.getMessage());
            e.printStackTrace();
            return kernelCode;
        }
    }

    /**
     * Stop the MCP server process.
     */
    public void stop() {
        if (serverProcess != null) {
            try {
                serverInput.close();
                serverOutput.close();
                serverError.close();
            } catch (IOException e) {
                // Ignore
            }
            serverProcess.destroy();
            serverProcess = null;
            initialized = false;

            // Remove shutdown hook if stopping manually
            if (shutdownHook != null) {
                try {
                    Runtime.getRuntime().removeShutdownHook(shutdownHook);
                } catch (IllegalStateException e) {
                    // JVM is already shutting down
                }
                shutdownHook = null;
            }
        }
    }

    private void sendRequest(String request) throws IOException {
        serverInput.write(request);
        serverInput.newLine();
        serverInput.flush();
    }

    private String readResponse() throws IOException {
        // Read until we get a complete JSON response
        StringBuilder response = new StringBuilder();
        String line;

        while ((line = serverOutput.readLine()) != null) {
            response.append(line);
            // Check if we have a complete JSON object
            if (isCompleteJson(response.toString())) {
                break;
            }
        }

        return response.toString();
    }

    private boolean isCompleteJson(String s) {
        int braces = 0;
        boolean inString = false;
        boolean escape = false;

        for (char c : s.toCharArray()) {
            if (escape) {
                escape = false;
                continue;
            }
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') {
                inString = !inString;
                continue;
            }
            if (!inString) {
                if (c == '{') braces++;
                if (c == '}') braces--;
            }
        }

        return braces == 0 && s.contains("{");
    }

    private String createJsonRpcRequest(String method, Map<String, Object> params) {
        int id = requestId.incrementAndGet();
        StringBuilder json = new StringBuilder();
        json.append("{\"jsonrpc\":\"2.0\",\"id\":").append(id);
        json.append(",\"method\":\"").append(method).append("\"");
        json.append(",\"params\":").append(toJson(params));
        json.append("}");
        return json.toString();
    }

    private String toJson(Object obj) {
        if (obj == null) {
            return "null";
        } else if (obj instanceof String) {
            return "\"" + escapeJson((String) obj) + "\"";
        } else if (obj instanceof Number) {
            return obj.toString();
        } else if (obj instanceof Boolean) {
            return obj.toString();
        } else if (obj instanceof Map) {
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) obj).entrySet()) {
                if (!first) sb.append(",");
                first = false;
                sb.append("\"").append(entry.getKey()).append("\":");
                sb.append(toJson(entry.getValue()));
            }
            sb.append("}");
            return sb.toString();
        } else if (obj instanceof List) {
            StringBuilder sb = new StringBuilder("[");
            boolean first = true;
            for (Object item : (List<?>) obj) {
                if (!first) sb.append(",");
                first = false;
                sb.append(toJson(item));
            }
            sb.append("]");
            return sb.toString();
        } else if (obj instanceof int[]) {
            return toJson(arrayToList((int[]) obj));
        }
        return "\"" + obj.toString() + "\"";
    }

    private String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private List<Integer> arrayToList(int[] arr) {
        if (arr == null) return null;
        List<Integer> list = new java.util.ArrayList<>();
        for (int i : arr) {
            list.add(i);
        }
        return list;
    }

    private String parseOptimizedKernel(String response) {
        // Look for "optimized_kernel" in the response
        // The response structure is:
        // {"jsonrpc":"2.0","id":N,"result":{"content":[{"type":"text","text":"{...}"}]}}

        try {
            // Find the inner result text
            int contentStart = response.indexOf("\"text\":");
            if (contentStart == -1) return null;

            contentStart = response.indexOf("\"", contentStart + 7) + 1;
            int contentEnd = findMatchingQuote(response, contentStart);

            if (contentEnd == -1) return null;

            String innerJson = response.substring(contentStart, contentEnd);
            // Unescape the JSON string
            innerJson = innerJson.replace("\\\"", "\"")
                                  .replace("\\n", "\n")
                                  .replace("\\\\", "\\");

            // Find optimized_kernel in inner JSON
            int kernelStart = innerJson.indexOf("\"optimized_kernel\":");
            if (kernelStart == -1) return null;

            kernelStart = innerJson.indexOf("\"", kernelStart + 19) + 1;
            int kernelEnd = findMatchingQuote(innerJson, kernelStart);

            if (kernelEnd == -1) return null;

            String kernel = innerJson.substring(kernelStart, kernelEnd);
            return kernel.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\");

        } catch (Exception e) {
            System.err.println("[MCP] Failed to parse response: " + e.getMessage());
            return null;
        }
    }

    private int findMatchingQuote(String s, int start) {
        boolean escape = false;
        for (int i = start; i < s.length(); i++) {
            char c = s.charAt(i);
            if (escape) {
                escape = false;
                continue;
            }
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') {
                return i;
            }
        }
        return -1;
    }

    private void startErrorReader() {
        Thread errorThread = new Thread(() -> {
            try {
                String line;
                while ((line = serverError.readLine()) != null) {
                    System.err.println("[MCP Server] " + line);
                }
            } catch (IOException e) {
                // Server closed
            }
        }, "MCP-Error-Reader");
        errorThread.setDaemon(true);
        errorThread.start();
    }
}
