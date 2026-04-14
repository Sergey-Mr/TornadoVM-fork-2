/*
 * Copyright (c) 2013-2024, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package uk.ac.manchester.tornado.api;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.api.plan.types.OffConcurrentDevices;
import uk.ac.manchester.tornado.api.plan.types.OffMemoryLimit;
import uk.ac.manchester.tornado.api.plan.types.OffPrintKernel;
import uk.ac.manchester.tornado.api.plan.types.OffProfiler;
import uk.ac.manchester.tornado.api.plan.types.OffThreadInfo;
import uk.ac.manchester.tornado.api.plan.types.WithAllGraphs;
import uk.ac.manchester.tornado.api.plan.types.WithBatch;
import uk.ac.manchester.tornado.api.plan.types.WithClearProfiles;
import uk.ac.manchester.tornado.api.plan.types.WithCompilerFlags;
import uk.ac.manchester.tornado.api.plan.types.WithConcurrentDevices;
import uk.ac.manchester.tornado.api.plan.types.WithDefaultScheduler;
import uk.ac.manchester.tornado.api.plan.types.WithDevice;
import uk.ac.manchester.tornado.api.plan.types.WithFreeDeviceMemory;
import uk.ac.manchester.tornado.api.plan.types.WithGraph;
import uk.ac.manchester.tornado.api.plan.types.WithGridScheduler;
import uk.ac.manchester.tornado.api.plan.types.WithMemoryLimit;
import uk.ac.manchester.tornado.api.plan.types.WithPreCompilation;
import uk.ac.manchester.tornado.api.plan.types.WithPrintKernel;
import uk.ac.manchester.tornado.api.plan.types.WithProfiler;
import uk.ac.manchester.tornado.api.plan.types.WithResetDevice;
import uk.ac.manchester.tornado.api.plan.types.WithThreadInfo;
import uk.ac.manchester.tornado.api.plan.types.WithWarmUpIterations;
import uk.ac.manchester.tornado.api.plan.types.WithWarmUpTime;
import uk.ac.manchester.tornado.api.mcp.MCPKernelOptimizer;
import uk.ac.manchester.tornado.api.runtime.ExecutorFrame;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/**
 * Class to create and optimize execution plans for running a set of
 * immutable tasks-graphs on modern hardware. An executor plan contains an
 * executor object, which in turn, contains a set of immutable task-graphs.
 * All actions applied to the execution plan affect to all the immutable
 * graphs associated with it.
 *
 * @since v0.15
 */
public sealed class TornadoExecutionPlan implements AutoCloseable permits ExecutionPlanType {

    /**
     * Method to obtain the default device in TornadoVM. The default one corresponds
     * to the device assigned to the driver (backend) with index 0 and device 0.
     */
    public static TornadoDevice DEFAULT_DEVICE = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

    private static final AtomicLong globalExecutionPlanCounter = new AtomicLong(0);

    /**
     * The TornadoVM executor is a list of chain of actions to be performed.
     * Each action can enable/disable runtime features, influence the compiler,
     * influence the code optimization, adapt runtime parameters, etc.
     */
    protected TornadoExecutor tornadoExecutor;

    protected ExecutorFrame executionFrame;

    /**
     * Reference to the Root of the List.
     */
    protected TornadoExecutionPlan rootNode;

    /**
     * Reference to the next node in the list.
     */
    protected TornadoExecutionPlan childLink;

    /**
     * Reference to the previous node in the list.
     */
    protected TornadoExecutionPlan parentLink;

    protected List<TornadoExecutionResult> planResults;

    /**
     * Track whether MCP optimization has been applied for each task.
     */
    private boolean mcpOptimizationApplied = false;
    private int mcpExecutionCount = 0;

    /**
     * MCP warmup and benchmark iterations - configurable via system properties.
     * -Dtornado.mcp.warmup=N     : Number of warmup iterations before collecting timing (default: 5)
     * -Dtornado.mcp.benchmark=N  : Number of benchmark iterations to collect timing (default: 10)
     *
     * Total iterations before MCP triggers = warmup + benchmark
     * Example: -Dtornado.mcp.warmup=2 -Dtornado.mcp.benchmark=3 triggers after 5 runs
     */
    private static final int MCP_WARMUP_ITERATIONS = Integer.getInteger("tornado.mcp.warmup", 5);
    private static final int MCP_BENCHMARK_ITERATIONS = Integer.getInteger("tornado.mcp.benchmark", 10);
    private List<Long> originalKernelTimes = new ArrayList<>();
    private String originalKernelSource = null;  // Store for signature validation

    // Validation support
    private static final float MCP_VALIDATION_TOLERANCE = 1e-4f;  // Tolerance for float comparison
    private List<float[]> originalOutputValues = null;  // Store original kernel output for validation
    private String lastValidationError = null;  // Store validation error message for feedback

    /**
     * Create an Execution Plan: Object to create and optimize an execution plan for
     * running a set of immutable tasks-graphs. An executor plan contains an
     * executor object, which in turn, contains a set of immutable task-graphs. All
     * actions applied to the execution plan affect to all the immutable graphs
     * associated with it.
     *
     * @param immutableTaskGraphs
     *     {@link ImmutableTaskGraph}
     */
    public TornadoExecutionPlan(ImmutableTaskGraph... immutableTaskGraphs) {
        tornadoExecutor = new TornadoExecutor(immutableTaskGraphs);
        final long id = globalExecutionPlanCounter.incrementAndGet();
        executionFrame = new ExecutorFrame(id);
        updateAccess(immutableTaskGraphs);
        rootNode = this;
        planResults = new ArrayList<>();
    }

    /**
     * If the {@code TornadoExecutionPlan} consists of multiple task-graphs, this function
     * updates the access type of the input and output data of each task-graph, as necessary.
     *
     * @param immutableTaskGraphs
     *     The list of the immutable task-graphs in the {@code TornadoExecutionPlan}
     */
    private void updateAccess(ImmutableTaskGraph... immutableTaskGraphs) {
        if (immutableTaskGraphs.length > 1) {
            for (ImmutableTaskGraph immutableTaskGraph : immutableTaskGraphs) {
                TaskGraph taskGraph = immutableTaskGraph.getTaskGraph();
                TornadoTaskGraphInterface taskGraphImpl = taskGraph.getTaskGraphImpl();
                taskGraphImpl.updateObjectAccess();
            }
        }
    }

    /**
     * Method to obtain a specific device using the driver index (backend index) and
     * device index.
     *
     * @param driverIndex
     *     Integer value that identifies the backend to be used.
     * @param deviceIndex
     *     Integer value that identifies the device within the backend to be
     *     used.
     * @return {@link TornadoDevice}
     *
     */
    public static TornadoDevice getDevice(int driverIndex, int deviceIndex) {
        return TornadoRuntimeProvider.getTornadoRuntime().getBackend(driverIndex).getDevice(deviceIndex);
    }

    /**
     * Method to return the total number of execution plans instantiated in a single JVM instance.
     *
     * @since 1.0.2
     * 
     * @return int
     */
    public static int getTotalPlans() {
        return globalExecutionPlanCounter.intValue();
    }

    /**
     * Return a data structure that contains all drivers and devices that the TornadoVM Runtime can access.
     * 
     * @return {@link TornadoDeviceMap}
     */
    public static TornadoDeviceMap getTornadoDeviceMap() {
        return new TornadoDeviceMap();
    }

    /**
     * Execute an execution plan. It returns a {@link TornadoExecutionPlan} for
     * further build different optimization after the execution as well as obtain
     * the profiler results.
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionResult execute() {
        // If MCP optimization is enabled, always enable profiler to collect timing data
        if (MCPKernelOptimizer.isEnabled() && executionFrame.getProfilerMode() == null) {
            executionFrame.setProfilerMode(ProfilerMode.SILENT);
        }

        tornadoExecutor.execute(executionFrame);
        TornadoProfilerResult profilerResult = new TornadoProfilerResult(tornadoExecutor, this.getTraceExecutionPlan());
        TornadoExecutionResult executionResult = new TornadoExecutionResult(profilerResult);
        planResults.add(executionResult);
        tornadoExecutor.updateLastExecutedTaskGraph();

        // MCP Optimization: Collect kernel times during warmup, then optimize
        mcpExecutionCount++;
        if (MCPKernelOptimizer.isEnabled() && !mcpOptimizationApplied) {
            // Use getDeviceKernelTime() - same as manual prebuiltTask benchmarks
            TornadoProfilerResult pr = executionResult.getProfilerResult();
            long kernelTimeNs = pr.getDeviceKernelTime();

            // Skip first few runs (compilation/warmup), then collect measurements
            if (mcpExecutionCount > MCP_WARMUP_ITERATIONS) {
                originalKernelTimes.add(kernelTimeNs);
            }

            // After collecting enough samples, optimize
            if (originalKernelTimes.size() >= MCP_BENCHMARK_ITERATIONS) {
                // Store original outputs for validation before optimization
                List<Object> outputs = tornadoExecutor.getOutputs();
                storeOriginalOutputs(outputs);

                applyMCPOptimization();
            }
        }

        return executionResult;
    }

    /**
     * Apply MCP optimization to the kernel with iterative feedback loop.
     * If the optimized kernel is slower, retry with feedback up to 3 times.
     * Uses same methodology as custom benchmarks: average multiple KERNEL_TIME measurements.
     */
    private void applyMCPOptimization() {
        mcpOptimizationApplied = true;  // Prevent re-entry
        System.out.printf("[MCP] Triggered after %d warmup + %d benchmark iterations%n",
                MCP_WARMUP_ITERATIONS, MCP_BENCHMARK_ITERATIONS);

        // Calculate statistics for original kernel (collected during warmup)
        long originalSum = originalKernelTimes.stream().mapToLong(Long::longValue).sum();
        long originalAvgNs = originalSum / originalKernelTimes.size();
        long originalMin = originalKernelTimes.stream().mapToLong(Long::longValue).min().orElse(0);
        long originalMax = originalKernelTimes.stream().mapToLong(Long::longValue).max().orElse(0);

        MCPKernelOptimizer optimizer = new MCPKernelOptimizer();
        String backend = System.getProperty("tornado.mcp.backend", "opencl");
        String taskId = tornadoExecutor.getFirstTaskId();
        System.out.println("[MCP] Detected task ID: " + taskId);

        String kernelSource = getGeneratedKernelSource(taskId);
        if (kernelSource == null || kernelSource.isEmpty()) {
            System.err.println("[MCP] Could not extract kernel source");
            return;
        }

        // Store original kernel for signature validation
        this.originalKernelSource = kernelSource;

        final double originalTimeMs = originalAvgNs / 1_000_000.0;
        System.out.printf("[MCP] Original kernel time (avg of %d runs): %.3f ms%n",
                originalKernelTimes.size(), originalTimeMs);
        System.out.printf("[MCP] Original min/max: %.3f / %.3f ms%n",
                originalMin / 1_000_000.0, originalMax / 1_000_000.0);

        // Use optimizeWithFeedback for iterative optimization with retry
        MCPKernelOptimizer.OptimizationResult result = optimizer.optimizeWithFeedback(
                kernelSource,
                backend,
                originalTimeMs,
                (optimizedKernel, gridConfig) -> benchmarkOptimizedKernelWithValidation(optimizedKernel, taskId, gridConfig)
        );

        // Use the stored timing from the result (no re-benchmarking to avoid variance)
        String finalKernel = result.optimizedKernel();
        double finalTimeMs = result.optimizedTimeMs();
        double speedup = originalTimeMs / finalTimeMs;
        double improvement = ((originalTimeMs - finalTimeMs) / originalTimeMs) * 100;

        // Note: We don't apply the optimized kernel to the original execution plan.
        // The prebuiltTask benchmark proves the optimization works. If users want to use
        // the optimized kernel in production, they should use prebuiltTask directly with
        // the optimized kernel file.

        System.out.println();
        System.out.println("╔═══════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                   MCP KERNEL OPTIMIZATION RESULTS                     ║");
        System.out.println("║                   (KERNEL_TIME only, same as benchmarks)              ║");
        System.out.println("╠═══════════════════════════════════════════════════════════════════════╣");
        System.out.printf("║  Original TornadoVM kernel:                                           ║%n");
        System.out.printf("║    Avg: %8.3f ms  (Min: %.3f, Max: %.3f)                         ║%n",
                originalTimeMs, originalMin / 1_000_000.0, originalMax / 1_000_000.0);
        System.out.printf("║  MCP-Optimized kernel (attempt %d):                                    ║%n", result.attemptNumber());
        System.out.printf("║    Avg: %8.3f ms                                                   ║%n", finalTimeMs);
        System.out.println("╠═══════════════════════════════════════════════════════════════════════╣");
        // Use actual speedup value to determine faster/slower (not result.success())
        if (speedup > 1.0) {
            if (result.success()) {
                System.out.printf("║  Speedup: %.2fx FASTER (%.1f%% improvement) ✓                         ║%n", speedup, improvement);
            } else {
                System.out.printf("║  Speedup: %.2fx FASTER (%.1f%% improvement, below threshold)          ║%n", speedup, improvement);
            }
        } else if (speedup == 1.0) {
            System.out.printf("║  Result: NO CHANGE (same performance)                                 ║%n");
        } else {
            System.out.printf("║  Speedup: %.2fx SLOWER (%.1f%% slower) - all %d attempts failed       ║%n",
                    speedup, -improvement, result.attemptNumber());
        }
        System.out.println("╚═══════════════════════════════════════════════════════════════════════╝");
        System.out.println();

        // Exit after MCP optimization - the benchmark is complete
        System.out.println("[MCP] Optimization complete. Exiting.");
        System.exit(0);
    }

    // Timeout for benchmarking to prevent infinite loops in generated kernels
    private static final int MCP_BENCHMARK_TIMEOUT_SECONDS = 60;

    /**
     * Benchmark an optimized kernel and return a BenchmarkResult with timing and validation status.
     * This is the main entry point for MCP optimization benchmarking.
     */
    private MCPKernelOptimizer.BenchmarkResult benchmarkOptimizedKernelWithValidation(
            String optimizedKernel, String taskId, MCPKernelOptimizer.GridConfig gridConfig) {
        double timeMs = benchmarkOptimizedKernel(optimizedKernel, taskId, gridConfig);
        String validationError = lastValidationError;  // Captured during benchmark
        lastValidationError = null;  // Reset for next attempt
        return new MCPKernelOptimizer.BenchmarkResult(timeMs, validationError);
    }

    /**
     * Benchmark an optimized kernel using prebuiltTask approach.
     * This creates a fresh TaskGraph with proper grid configuration,
     * ensuring the optimized kernel runs with correct thread mapping.
     *
     * Includes timeout protection to prevent hanging on infinite loops.
     */
    private double benchmarkOptimizedKernel(String optimizedKernel, String taskId, MCPKernelOptimizer.GridConfig gridConfig) {
        // Validate kernel signature matches original
        if (originalKernelSource != null && !validateKernelSignature(originalKernelSource, optimizedKernel)) {
            System.err.println("[MCP] Rejecting optimized kernel due to signature mismatch");
            return Double.MAX_VALUE;
        }

        // Validate array offsets before benchmarking
        validateArrayOffsets(optimizedKernel);

        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<Double> future = executor.submit(() -> benchmarkWithPrebuiltTask(optimizedKernel, taskId, gridConfig));

        try {
            // Wait with timeout to prevent infinite loops
            double result = future.get(MCP_BENCHMARK_TIMEOUT_SECONDS, TimeUnit.SECONDS);
            return result;
        } catch (TimeoutException e) {
            System.err.println("[MCP] ⚠ Benchmark TIMEOUT after " + MCP_BENCHMARK_TIMEOUT_SECONDS +
                    "s - kernel may have infinite loop or be extremely slow");
            future.cancel(true);
            return Double.MAX_VALUE;  // Treat as failed optimization
        } catch (ExecutionException e) {
            System.err.println("[MCP] PrebuiltTask benchmark failed: " + e.getCause().getMessage());
            e.getCause().printStackTrace();
            // Fallback to replaceKernelSource approach
            return benchmarkWithReplaceKernel(optimizedKernel, taskId);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("[MCP] Benchmark interrupted");
            return Double.MAX_VALUE;
        } finally {
            executor.shutdownNow();
        }
    }

    /**
     * Validate that array offsets in the kernel use TornadoVM's +4 pattern.
     * Warns if unusual offsets are detected that might cause incorrect results.
     */
    private void validateArrayOffsets(String kernelSource) {
        // Look for pointer casting patterns: ((__global float *)A) + N
        Pattern pattern = Pattern.compile("\\(\\s*\\(\\s*__global\\s+\\w+\\s*\\*\\s*\\)\\s*\\w+\\s*\\)\\s*\\+\\s*(\\d+)");
        Matcher matcher = pattern.matcher(kernelSource);

        while (matcher.find()) {
            int offset = Integer.parseInt(matcher.group(1));
            if (offset != 4) {
                System.err.println("[MCP] ⚠ WARNING: Unusual array offset detected: +" + offset +
                        " (TornadoVM expects +4 for float arrays)");
                System.err.println("[MCP]   This may cause incorrect results. Matched: " + matcher.group());
            }
        }

        // Also check for missing offsets (direct cast without offset)
        Pattern directCastPattern = Pattern.compile("\\(\\s*\\(\\s*__global\\s+\\w+\\s*\\*\\s*\\)\\s*\\w+\\s*\\)\\s*\\[");
        Matcher directMatcher = directCastPattern.matcher(kernelSource);
        if (directMatcher.find()) {
            System.err.println("[MCP] ⚠ WARNING: Direct array access without offset detected");
            System.err.println("[MCP]   TornadoVM arrays require +4 offset for float data. Matched: " + directMatcher.group());
        }
    }

    /**
     * Validate that the optimized kernel has the same signature as the original.
     * Returns true if signatures match, false otherwise.
     */
    private boolean validateKernelSignature(String originalKernel, String optimizedKernel) {
        String originalSig = extractKernelSignature(originalKernel);
        String optimizedSig = extractKernelSignature(optimizedKernel);

        if (originalSig == null || optimizedSig == null) {
            System.err.println("[MCP] ⚠ WARNING: Could not extract kernel signature for validation");
            return true;  // Allow to proceed, but warn
        }

        // Count parameters
        int originalParamCount = countParameters(originalSig);
        int optimizedParamCount = countParameters(optimizedSig);

        if (originalParamCount != optimizedParamCount) {
            System.err.println("[MCP] ERROR: Kernel signature mismatch!");
            System.err.println("[MCP]   Original has " + originalParamCount + " parameters");
            System.err.println("[MCP]   Optimized has " + optimizedParamCount + " parameters");
            System.err.println("[MCP]   Original signature: " + originalSig.substring(0, Math.min(200, originalSig.length())) + "...");
            System.err.println("[MCP]   Optimized signature: " + optimizedSig.substring(0, Math.min(200, optimizedSig.length())) + "...");
            return false;
        }

        return true;
    }

    /**
     * Extract the kernel function signature (parameters) from kernel source.
     */
    private String extractKernelSignature(String kernelSource) {
        // Match __kernel void funcName(params)
        Pattern pattern = Pattern.compile("__kernel\\s+void\\s+\\w+\\s*\\(([^)]+)\\)");
        Matcher matcher = pattern.matcher(kernelSource);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        return null;
    }

    /**
     * Count the number of parameters in a kernel signature.
     */
    private int countParameters(String signature) {
        if (signature == null || signature.isEmpty()) {
            return 0;
        }
        // Count commas + 1 (but handle nested parentheses)
        int count = 1;
        int parenDepth = 0;
        for (char c : signature.toCharArray()) {
            if (c == '(') parenDepth++;
            else if (c == ')') parenDepth--;
            else if (c == ',' && parenDepth == 0) count++;
        }
        return count;
    }

    /**
     * Benchmark using prebuiltTask - creates fresh TaskGraph with proper grid configuration.
     * This is the preferred method as it correctly handles optimized kernels that expect
     * 1:1 thread mapping instead of grid-stride loops.
     */
    private double benchmarkWithPrebuiltTask(String optimizedKernel, String taskId, MCPKernelOptimizer.GridConfig gridConfig) throws IOException, TornadoExecutionPlanException {
        // 1. Write optimized kernel to temp file
        File tempKernelFile = File.createTempFile("mcp_optimized_", ".cl");
        tempKernelFile.deleteOnExit();
        try (FileOutputStream fos = new FileOutputStream(tempKernelFile)) {
            fos.write(optimizedKernel.getBytes(StandardCharsets.UTF_8));
        }

        // 2. Extract entry point from kernel
        String entryPoint = extractEntryPoint(optimizedKernel);
        if (entryPoint == null) {
            throw new RuntimeException("Could not extract entry point from kernel");
        }

        // 3. Get inputs and outputs from the original task graph
        List<Object> inputs = tornadoExecutor.getInputs();
        List<Object> outputs = tornadoExecutor.getOutputs();

        if (inputs.isEmpty()) {
            throw new RuntimeException("No inputs found in task graph");
        }

        // 4. Determine problem size based on gridConfig dimensions
        int[] globalSizes = resolveGlobalWorkSize(gridConfig, inputs, outputs);
        System.out.printf("[MCP] Resolved global work size: %s%n", java.util.Arrays.toString(globalSizes));

        // 5. Create AccessorParameters based on actual kernel signature
        // Count user parameters (total params - 4 TornadoVM internal params)
        String kernelSig = extractKernelSignature(optimizedKernel);
        int totalKernelParams = countParameters(kernelSig);
        int userParams = totalKernelParams - 4;  // Subtract 4 TornadoVM params

        int dataParams = inputs.size() + outputs.size();

        // Count scalar params directly from signature (don't derive from data objects).
        // This handles kernels where inputs/outputs lists contain duplicates or shared objects
        // (e.g., BFS has 'vertices' in both inputs and outputs).
        int scalarCount = 0;
        {
            String[] sigParams = kernelSig.split(",");
            for (int i = 4; i < sigParams.length; i++) {
                if (sigParams[i].contains("__private")) {
                    scalarCount++;
                }
            }
        }

        System.out.printf("[MCP] Kernel has %d total params, %d user params, %d data objects, %d scalar params%n",
                totalKernelParams, userParams, dataParams, scalarCount);

        // Size array to match the kernel signature - this is the source of truth for the kernel.
        // The fallback loop below will only fill slots if there's room (bounds-checked).
        int numParams = userParams;
        AccessorParameters accessors = new AccessorParameters(numParams);

        // CRITICAL: Parameters must be added in the EXACT order the kernel expects them.
        // For NBody, params are interleaved: numBodies(scalar), refPos(array), refVel(array), delT(scalar), espSqr(scalar)
        // We process ALL user params in signature order, checking type for each.

        List<Object> remainingInputs = new ArrayList<>(inputs);
        List<Object> remainingOutputs = new ArrayList<>(outputs);

        // Get all user parameters from signature in order
        String[] allParams = kernelSig.split(",");
        int paramIndex = 0;
        int startIdx = 4;  // Skip 4 TornadoVM internal params

        System.out.printf("[MCP] Processing %d user params from signature (after %d internal)%n",
                allParams.length - startIdx, startIdx);

        for (int i = startIdx; i < allParams.length; i++) {
            String param = allParams[i].trim();
            String[] words = param.split("\\s+");
            String paramName = words.length > 0 ? words[words.length - 1].replaceAll("[*\\[\\]]", "") : "";

            if (param.contains("__global") && param.contains("uchar")) {
                // This is a data array parameter
                Object matched = matchParameterToDataObject(paramName, remainingInputs, remainingOutputs);
                if (matched != null) {
                    Access accessType = remainingInputs.contains(matched) ? Access.READ_ONLY : Access.WRITE_ONLY;
                    if (!remainingInputs.remove(matched)) {
                        remainingOutputs.remove(matched);
                    }
                    accessors.set(paramIndex++, matched, accessType);
                    System.out.printf("[MCP] Param[%d] '%s' -> %s (%s) [ARRAY]%n", i - startIdx, paramName,
                            matched.getClass().getSimpleName(), accessType);
                } else {
                    System.err.printf("[MCP] WARNING: Could not match array param '%s' to any data object%n", paramName);
                }
            } else if (param.contains("__private")) {
                // This is a scalar parameter - determine if int or float
                boolean isFloat = param.contains("float");
                if (isFloat) {
                    float floatValue = resolveFloatParameter(paramName, inputs, outputs);
                    accessors.set(paramIndex++, Float.valueOf(floatValue), Access.NONE);
                    System.out.printf("[MCP] Param[%d] '%s' = %.6f [SCALAR FLOAT]%n", i - startIdx, paramName, floatValue);
                } else {
                    int intValue = resolveScalarParameter(paramName, inputs, outputs, globalSizes, paramIndex);
                    accessors.set(paramIndex++, Integer.valueOf(intValue), Access.NONE);
                    System.out.printf("[MCP] Param[%d] '%s' = %d [SCALAR INT]%n", i - startIdx, paramName, intValue);
                }
            } else {
                System.out.printf("[MCP] Skipping unknown param type: '%s'%n", param);
            }
        }

        // Add any remaining unmatched objects as fallback (bounds-checked).
        // The kernel only accepts `numParams` arguments; extra data objects in the task graph
        // (e.g., duplicates when same array is both input and output) are skipped.
        for (Object input : remainingInputs) {
            if (paramIndex >= numParams) {
                System.out.printf("[MCP] Skipping extra input %s (kernel only takes %d params)%n",
                        input.getClass().getSimpleName(), numParams);
                continue;
            }
            accessors.set(paramIndex++, input, Access.READ_ONLY);
            System.out.printf("[MCP] Fallback: added remaining input %s%n", input.getClass().getSimpleName());
        }
        for (Object output : remainingOutputs) {
            if (paramIndex >= numParams) {
                System.out.printf("[MCP] Skipping extra output %s (kernel only takes %d params)%n",
                        output.getClass().getSimpleName(), numParams);
                continue;
            }
            accessors.set(paramIndex++, output, Access.WRITE_ONLY);
            System.out.printf("[MCP] Fallback: added remaining output %s%n", output.getClass().getSimpleName());
        }

        // 6. Create new TaskGraph with prebuiltTask
        String graphName = "mcp_bench";
        TaskGraph benchGraph = new TaskGraph(graphName)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, inputs.toArray())
                .prebuiltTask(taskId, entryPoint, tempKernelFile.getAbsolutePath(), accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputs.toArray());

        ImmutableTaskGraph snapshot = benchGraph.snapshot();

        // 7. Configure grid scheduler based on gridConfig from LLM
        WorkerGrid workerGrid = createWorkerGrid(gridConfig, globalSizes, optimizedKernel);
        GridScheduler gridScheduler = new GridScheduler(graphName + "." + taskId, workerGrid);

        // 8. Warmup and benchmark
        // NOTE: We don't use try-with-resources here because closing the plan
        // would free device memory for the shared input/output objects, which
        // breaks the original execution plan that shares these objects.
        List<Long> benchTimes = new ArrayList<>();
        TornadoDevice device = tornadoExecutor.getDevice(0);

        TornadoExecutionPlan benchPlan = new TornadoExecutionPlan(snapshot);
        benchPlan.withDevice(device).withGridScheduler(gridScheduler);

        // Warmup
        for (int i = 0; i < MCP_WARMUP_ITERATIONS; i++) {
            benchPlan.execute();
        }

        // Benchmark
        for (int i = 0; i < MCP_BENCHMARK_ITERATIONS; i++) {
            TornadoExecutionResult result = benchPlan.withProfiler(ProfilerMode.SILENT).execute();
            long kernelTime = result.getProfilerResult().getDeviceKernelTime();
            benchTimes.add(kernelTime);
        }
        // Don't close benchPlan - let it be garbage collected without freeing device memory

        // Calculate average time in ms
        long sum = benchTimes.stream().mapToLong(Long::longValue).sum();
        long avgNs = sum / benchTimes.size();

        // Validate: if all times are 0, the kernel failed to execute
        boolean allZero = benchTimes.stream().allMatch(t -> t == 0);
        if (allZero || avgNs == 0) {
            System.err.println("[MCP] ERROR: Optimized kernel failed to execute (0ms timing indicates compilation/execution failure)");
            lastValidationError = "Kernel execution failed (0ms timing)";
            return Double.MAX_VALUE;  // Treat as failed optimization
        }

        // Validate output correctness
        ValidationResult validation = validateOutputs(outputs);
        if (!validation.passed()) {
            System.err.printf("[MCP] ✗ Validation %s%n", validation);
            lastValidationError = validation.message();
            return Double.MAX_VALUE;  // Treat as failed optimization - wrong results
        } else {
            System.out.printf("[MCP] ✓ Validation %s%n", validation);
            lastValidationError = null;
        }

        return avgNs / 1_000_000.0;
    }

    /**
     * Fallback benchmark using replaceKernelSource approach.
     * Less accurate for optimized kernels but works when prebuiltTask fails.
     */
    private double benchmarkWithReplaceKernel(String optimizedKernel, String taskId) {
        boolean replaced = replaceKernelSource(taskId, optimizedKernel);
        if (!replaced) {
            System.err.println("[MCP] Kernel replacement failed!");
            return Double.MAX_VALUE;
        }

        configureMCPGridScheduler(optimizedKernel, taskId);

        // Warmup
        for (int i = 0; i < MCP_WARMUP_ITERATIONS; i++) {
            tornadoExecutor.execute(executionFrame);
        }

        // Benchmark
        List<Long> optimizedTimes = new ArrayList<>();
        for (int i = 0; i < MCP_BENCHMARK_ITERATIONS; i++) {
            tornadoExecutor.execute(executionFrame);
            TornadoProfilerResult pr = new TornadoProfilerResult(tornadoExecutor, this.getTraceExecutionPlan());
            long kernelTime = pr.getDeviceKernelTime();
            optimizedTimes.add(kernelTime);
        }

        long optimizedSum = optimizedTimes.stream().mapToLong(Long::longValue).sum();
        long optimizedAvgNs = optimizedSum / optimizedTimes.size();
        return optimizedAvgNs / 1_000_000.0;
    }

    /**
     * Extract kernel entry point name from OpenCL or PTX source.
     * OpenCL: __kernel void functionName(...)
     * PTX: .visible .entry functionName(...)
     */
    private String extractEntryPoint(String kernelSource) {
        // Try OpenCL pattern first: __kernel void functionName(
        Pattern openclPattern = Pattern.compile("__kernel\\s+void\\s+(\\w+)\\s*\\(");
        Matcher openclMatcher = openclPattern.matcher(kernelSource);
        if (openclMatcher.find()) {
            return openclMatcher.group(1);
        }

        // Try PTX pattern: .visible .entry functionName(
        Pattern ptxPattern = Pattern.compile("\\.visible\\s+\\.entry\\s+(\\w+)\\s*\\(");
        Matcher ptxMatcher = ptxPattern.matcher(kernelSource);
        if (ptxMatcher.find()) {
            return ptxMatcher.group(1);
        }

        // Try alternative PTX pattern without .visible: .entry functionName(
        Pattern ptxAltPattern = Pattern.compile("\\.entry\\s+(\\w+)\\s*\\(");
        Matcher ptxAltMatcher = ptxAltPattern.matcher(kernelSource);
        if (ptxAltMatcher.find()) {
            return ptxAltMatcher.group(1);
        }

        System.err.println("[MCP] Could not extract entry point from kernel");
        return null;
    }

    // =========================================================================
    // MCP Output Validation
    // =========================================================================

    /**
     * Store the current output values from the original kernel execution.
     * Called before running the optimized kernel.
     */
    private void storeOriginalOutputs(List<Object> outputs) {
        originalOutputValues = new ArrayList<>();
        for (Object output : outputs) {
            float[] copy = copyOutputToFloatArray(output);
            if (copy != null) {
                originalOutputValues.add(copy);
            }
        }
        if (!originalOutputValues.isEmpty()) {
            System.out.printf("[MCP] Stored %d output arrays for validation%n", originalOutputValues.size());
        }
    }

    /**
     * Copy output object to a float array for comparison.
     * Supports FloatArray, DoubleArray, and common array types.
     */
    private float[] copyOutputToFloatArray(Object output) {
        try {
            String className = output.getClass().getName();

            // Handle FloatArray
            if (className.contains("FloatArray")) {
                java.lang.reflect.Method getSize = output.getClass().getMethod("getSize");
                int size = (int) getSize.invoke(output);
                java.lang.reflect.Method get = output.getClass().getMethod("get", int.class);

                float[] copy = new float[size];
                for (int i = 0; i < size; i++) {
                    copy[i] = (float) get.invoke(output, i);
                }
                return copy;
            }

            // Handle DoubleArray (convert to float for comparison)
            if (className.contains("DoubleArray")) {
                java.lang.reflect.Method getSize = output.getClass().getMethod("getSize");
                int size = (int) getSize.invoke(output);
                java.lang.reflect.Method get = output.getClass().getMethod("get", int.class);

                float[] copy = new float[size];
                for (int i = 0; i < size; i++) {
                    copy[i] = (float) (double) get.invoke(output, i);
                }
                return copy;
            }

            // Handle IntArray / LongArray / ShortArray (convert to float for comparison).
            // Integer outputs (e.g. BFS vertex distances) compare exactly — tolerance doesn't apply
            // in practice because int values don't drift, but the float representation is fine
            // for values within 2^24 (which covers typical graph/problem sizes).
            if (className.contains("IntArray") || className.contains("LongArray") || className.contains("ShortArray")) {
                java.lang.reflect.Method getSize = output.getClass().getMethod("getSize");
                int size = (int) getSize.invoke(output);
                java.lang.reflect.Method get = output.getClass().getMethod("get", int.class);

                float[] copy = new float[size];
                for (int i = 0; i < size; i++) {
                    Object val = get.invoke(output, i);
                    copy[i] = ((Number) val).floatValue();
                }
                return copy;
            }

            // Handle primitive int[]
            if (output instanceof int[]) {
                int[] src = (int[]) output;
                float[] copy = new float[src.length];
                for (int i = 0; i < src.length; i++) {
                    copy[i] = (float) src[i];
                }
                return copy;
            }

            // Handle primitive long[]
            if (output instanceof long[]) {
                long[] src = (long[]) output;
                float[] copy = new float[src.length];
                for (int i = 0; i < src.length; i++) {
                    copy[i] = (float) src[i];
                }
                return copy;
            }

            // Handle primitive float[]
            if (output instanceof float[]) {
                return ((float[]) output).clone();
            }

            // Handle primitive double[]
            if (output instanceof double[]) {
                double[] d = (double[]) output;
                float[] copy = new float[d.length];
                for (int i = 0; i < d.length; i++) {
                    copy[i] = (float) d[i];
                }
                return copy;
            }

            System.err.printf("[MCP] Warning: Unknown output type %s, skipping validation%n", className);
            return null;
        } catch (Exception e) {
            System.err.println("[MCP] Error copying output: " + e.getMessage());
            return null;
        }
    }

    /**
     * Validate the current outputs against stored original outputs.
     * Returns a ValidationResult with pass/fail status and error details.
     *
     * Detects two types of failures:
     * 1. Compilation failure: all outputs are 0.0 (kernel didn't execute)
     * 2. Validation failure: outputs differ from expected values
     */
    private ValidationResult validateOutputs(List<Object> outputs) {
        if (originalOutputValues == null || originalOutputValues.isEmpty()) {
            return new ValidationResult(true, "No original outputs to compare", 0, 0, -1);
        }

        float maxDiff = 0;
        int maxDiffIndex = -1;
        int outputIdx = 0;
        boolean allZeros = true;  // Track if all current outputs are zero (likely compilation failure)
        int totalElements = 0;

        for (int i = 0; i < outputs.size() && i < originalOutputValues.size(); i++) {
            float[] original = originalOutputValues.get(i);
            float[] current = copyOutputToFloatArray(outputs.get(i));

            if (current == null || original.length != current.length) {
                String error = String.format("Output %d size mismatch: expected %d, got %d",
                        i, original.length, current != null ? current.length : 0);
                return new ValidationResult(false, error, 0, 0, -1);
            }

            for (int j = 0; j < original.length; j++) {
                totalElements++;
                if (current[j] != 0.0f) {
                    allZeros = false;
                }
                float diff = Math.abs(original[j] - current[j]);
                if (diff > maxDiff) {
                    maxDiff = diff;
                    maxDiffIndex = j;
                    outputIdx = i;
                }
                if (diff > MCP_VALIDATION_TOLERANCE) {
                    // Check if this looks like a compilation failure (all zeros so far)
                    if (allZeros && original[j] != 0.0f) {
                        // Current is 0, original is non-zero - likely compilation failure
                        String error = String.format(
                                "LIKELY COMPILATION FAILURE: Output[%d][%d] is 0.0 but expected %.6f. " +
                                "All %d output values are 0.0 - kernel probably failed to compile or execute. " +
                                "Check for syntax errors, undefined macros, or wrong entry point name.",
                                i, j, original[j], totalElements);
                        return new ValidationResult(false, error, maxDiff, MCP_VALIDATION_TOLERANCE, j);
                    }
                    String error = String.format("Output[%d][%d] mismatch: expected %.6f, got %.6f (diff=%.6f)",
                            i, j, original[j], current[j], diff);
                    return new ValidationResult(false, error, maxDiff, MCP_VALIDATION_TOLERANCE, j);
                }
            }
        }

        return new ValidationResult(true,
                String.format("Max diff: %.2e at output[%d][%d]", maxDiff, outputIdx, maxDiffIndex),
                maxDiff, MCP_VALIDATION_TOLERANCE, maxDiffIndex);
    }

    /**
     * Result of output validation.
     */
    private record ValidationResult(boolean passed, String message, float maxDiff, float tolerance, int maxDiffIndex) {
        @Override
        public String toString() {
            if (passed) {
                return String.format("PASSED (%s, tolerance=%.0e)", message, tolerance);
            } else {
                return String.format("FAILED: %s", message);
            }
        }
    }

    /**
     * Parse local work size from reqd_work_group_size attribute.
     * Returns [localX, localY] or null if not found.
     * Handles both numeric values and macro names like TS.
     */
    private int[] parseLocalWorkSize(String kernelSource) {
        // Try to match numeric values directly: reqd_work_group_size(16, 16, 1)
        Pattern numericPattern = Pattern.compile("reqd_work_group_size\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)");
        Matcher numericMatcher = numericPattern.matcher(kernelSource);
        if (numericMatcher.find()) {
            return new int[]{
                    Integer.parseInt(numericMatcher.group(1)),
                    Integer.parseInt(numericMatcher.group(2))
            };
        }

        // Try to match macro names: reqd_work_group_size(TS, TS, 1) or reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)
        Pattern macroPattern = Pattern.compile("reqd_work_group_size\\s*\\(\\s*(\\w+)\\s*,\\s*(\\w+)\\s*,\\s*\\d+\\s*\\)");
        Matcher macroMatcher = macroPattern.matcher(kernelSource);
        if (macroMatcher.find()) {
            String macroName = macroMatcher.group(1);
            // Look for #define MACRO_NAME value
            Pattern definePattern = Pattern.compile("#define\\s+" + macroName + "\\s+(\\d+)");
            Matcher defineMatcher = definePattern.matcher(kernelSource);
            if (defineMatcher.find()) {
                int size = Integer.parseInt(defineMatcher.group(1));
                return new int[]{size, size};
            }
        }

        return null;
    }

    /**
     * Determine problem size from an input object.
     * Supports Matrix2DFloat and TornadoVM array types.
     */
    private int determineProblemSize(Object input) {
        // Try Matrix2DFloat
        if (input.getClass().getName().contains("Matrix2DFloat")) {
            try {
                java.lang.reflect.Method getRows = input.getClass().getMethod("getNumRows");
                return (int) getRows.invoke(input);
            } catch (Exception e) {
                // Fall through to next method
            }
        }

        // Try FloatArray or similar TornadoVM array types
        try {
            java.lang.reflect.Method getSize = input.getClass().getMethod("getSize");
            int totalSize = (int) getSize.invoke(input);
            // Assume square matrix
            return (int) Math.sqrt(totalSize);
        } catch (Exception e) {
            // Fall through
        }

        // Default fallback
        System.err.println("[MCP] Could not determine problem size, using default 1024");
        return 1024;
    }

    /**
     * Get the raw array size from an input object (without sqrt for 1D kernels).
     */
    private int getArraySize(Object input) {
        // Try FloatArray or similar TornadoVM array types
        try {
            java.lang.reflect.Method getSize = input.getClass().getMethod("getSize");
            return (int) getSize.invoke(input);
        } catch (Exception e) {
            // Fall through
        }

        // Try Matrix2DFloat - return total elements
        if (input.getClass().getName().contains("Matrix2DFloat")) {
            try {
                java.lang.reflect.Method getRows = input.getClass().getMethod("getNumRows");
                java.lang.reflect.Method getCols = input.getClass().getMethod("getNumColumns");
                int rows = (int) getRows.invoke(input);
                int cols = (int) getCols.invoke(input);
                return rows * cols;
            } catch (Exception e) {
                // Fall through
            }
        }

        return 1024;  // Default fallback
    }

    /**
     * Resolve global work size from gridConfig and inputs/outputs.
     * Supports 1D, 2D, and 3D grids, including non-square configurations.
     *
     * For expressions like "D * 32", evaluates using detected dimensions.
     */
    private int[] resolveGlobalWorkSize(MCPKernelOptimizer.GridConfig gridConfig, List<Object> inputs, List<Object> outputs) {
        if (gridConfig == null) {
            // Fallback to 2D square matrix assumption
            int size = determineProblemSize(inputs.get(0));
            return new int[] { size, size };
        }

        int dims = gridConfig.dimensions();
        String[] globalParams = gridConfig.globalWorkSize();

        // Get base sizes from inputs
        int inputArraySize = getArraySize(inputs.get(0));  // Total elements in first input
        int matrixDim = determineProblemSize(inputs.get(0));  // sqrt for matrices

        // Get output dimension (important for matrix-vector: output size = d)
        int outputSize = outputs.isEmpty() ? matrixDim : getArraySize(outputs.get(0));

        // For non-square operations, detect dimensions from input/output relationship
        // Matrix-vector: W[d,n] * x[n] = y[d]
        // So d = outputSize, n = inputArraySize (for vector input)
        int inputDim = inputArraySize;  // n (columns)
        int outputDim = outputSize;     // d (rows)

        System.out.printf("[MCP] Detected dimensions: inputDim=%d, outputDim=%d, matrixDim=%d%n",
                inputDim, outputDim, matrixDim);

        // Resolve each dimension
        int[] globalSizes = new int[dims];
        for (int i = 0; i < dims; i++) {
            String param = (i < globalParams.length) ? globalParams[i] : globalParams[0];
            globalSizes[i] = resolveGlobalSizeExpression(param, inputArraySize, matrixDim, inputDim, outputDim, inputs);
        }

        System.out.printf("[MCP] %dD kernel: resolved global work size %s%n",
                dims, java.util.Arrays.toString(globalSizes));
        return globalSizes;
    }

    /**
     * Resolve a global size expression to a concrete value.
     * Handles parameter names and simple expressions like "D * 32".
     */
    private int resolveGlobalSizeExpression(String expr, int arraySize, int matrixDim, int inputDim, int outputDim, List<Object> inputs) {
        expr = expr.trim();

        // Handle expressions with multiplication (e.g., "D * 32", "outputDim * 32")
        if (expr.contains("*")) {
            String[] parts = expr.split("\\*");
            int result = 1;
            for (String part : parts) {
                part = part.trim();
                if (part.matches("\\d+")) {
                    result *= Integer.parseInt(part);
                } else {
                    // Resolve the variable part
                    result *= resolveParameterName(part, arraySize, matrixDim, inputDim, outputDim, inputs);
                }
            }
            return result;
        }

        // Handle pure numbers
        if (expr.matches("\\d+")) {
            return Integer.parseInt(expr);
        }

        // Handle parameter names
        return resolveParameterName(expr, arraySize, matrixDim, inputDim, outputDim, inputs);
    }

    /**
     * Resolve a parameter name to its value.
     * Uses inputDim (n) for input-related params, outputDim (d) for output-related params.
     */
    private int resolveParameterName(String paramName, int arraySize, int matrixDim, int inputDim, int outputDim, List<Object> inputs) {
        paramName = paramName.toLowerCase();

        // 'd' is typically the OUTPUT dimension (rows, number of outputs)
        if (paramName.equals("d") || paramName.contains("output") || paramName.equals("rows") ||
            paramName.equals("numrows") || paramName.equals("m")) {
            System.out.printf("[MCP] Resolving '%s' to outputDim=%d%n", paramName, outputDim);
            return outputDim;
        }

        // 'n' is typically the INPUT dimension (columns, input length)
        if (paramName.equals("n") || paramName.equals("length") || paramName.contains("arraylength") ||
            paramName.equals("cols") || paramName.equals("numcols") || paramName.contains("input")) {
            System.out.printf("[MCP] Resolving '%s' to inputDim=%d%n", paramName, inputDim);
            return inputDim;
        }

        // numBodies and similar - use full array size
        if (paramName.contains("numbodies") || paramName.contains("num_bodies")) {
            return arraySize;
        }

        // 'size' — same disambiguation as resolveScalarParameter: matrixDim for
        // 2D square matrices (input is a perfect square), arraySize for 1D.
        if (paramName.equals("size")) {
            boolean isPerfectSquare = (matrixDim * matrixDim == arraySize);
            int resolved = isPerfectSquare ? matrixDim : arraySize;
            System.out.printf("[MCP] Resolving 'size' to %d (%s)%n",
                    resolved, isPerfectSquare ? "matrixDim, 2D" : "arraySize, 1D");
            return resolved;
        }

        // Try to get dimensions from Matrix2DFloat if available
        if ((paramName.contains("row") || paramName.contains("height")) && !inputs.isEmpty()) {
            Object input = inputs.get(0);
            if (input.getClass().getName().contains("Matrix2DFloat")) {
                try {
                    java.lang.reflect.Method getRows = input.getClass().getMethod("getNumRows");
                    return (int) getRows.invoke(input);
                } catch (Exception e) { /* fall through */ }
            }
        }
        if ((paramName.contains("col") || paramName.contains("width")) && !inputs.isEmpty()) {
            Object input = inputs.get(0);
            if (input.getClass().getName().contains("Matrix2DFloat")) {
                try {
                    java.lang.reflect.Method getCols = input.getClass().getMethod("getNumColumns");
                    return (int) getCols.invoke(input);
                } catch (Exception e) { /* fall through */ }
            }
        }

        // Default: use output dimension (safer for 1D kernels)
        System.out.printf("[MCP] Unknown param '%s', defaulting to outputDim=%d%n", paramName, outputDim);
        return outputDim;
    }

    /**
     * Extract the names of scalar parameters from the kernel signature.
     * Scalar parameters are __private int/float parameters (NOT __global arrays).
     * This method looks at the TYPE of each parameter, not just position.
     * E.g., for "..., __private int numBodies, __global uchar *refPos, __private float delT"
     * returns ["numBodies", "delT"] - only the __private parameters
     *
     * Note: kernelSig is already the parameter list without parentheses
     * (as returned by extractKernelSignature).
     */
    private List<String> extractScalarParameterNames(String kernelSig, int scalarCount) {
        List<String> names = new ArrayList<>();
        if (scalarCount <= 0 || kernelSig == null || kernelSig.isEmpty()) return names;

        // kernelSig is already the parameter list (without parentheses)
        // Split by comma to get individual parameters
        String[] params = kernelSig.split(",");

        // Skip first 4 TornadoVM internal params, then look for __private parameters
        int startIdx = 4;  // After _kernel_context, _constant_region, _local_region, _atomics
        for (int i = startIdx; i < params.length && names.size() < scalarCount; i++) {
            String param = params[i].trim();
            // Only include __private parameters (scalars), skip __global (arrays)
            if (param.contains("__private")) {
                // Extract just the parameter name (last word)
                String[] words = param.split("\\s+");
                if (words.length > 0) {
                    String name = words[words.length - 1];
                    // Remove any pointer/array symbols
                    name = name.replaceAll("[*\\[\\]]", "");
                    names.add(name);
                    System.out.printf("[MCP] DEBUG: Extracted scalar param '%s' from '%s'%n", name, param);
                }
            }
        }
        return names;
    }

    /**
     * Extract the names of data (array) parameters from the kernel signature.
     * Data parameters are __global uchar* parameters (NOT __private scalars).
     * This method looks at the TYPE of each parameter, not just position.
     * E.g., for "..., __private int numBodies, __global uchar *refPos, __global uchar *refVel, __private float delT"
     * returns ["refPos", "refVel"] - only the __global parameters
     */
    private List<String> extractDataParameterNames(String kernelSig, int dataCount) {
        List<String> names = new ArrayList<>();
        if (dataCount <= 0 || kernelSig == null || kernelSig.isEmpty()) return names;

        // kernelSig is the parameter list (without parentheses)
        String[] params = kernelSig.split(",");

        // Skip first 4 TornadoVM internal params, then look for __global parameters
        int startIdx = 4;  // After _kernel_context, _constant_region, _local_region, _atomics
        for (int i = startIdx; i < params.length && names.size() < dataCount; i++) {
            String param = params[i].trim();
            // Only include __global parameters (arrays), skip __private (scalars)
            if (param.contains("__global") && param.contains("uchar")) {
                // Extract just the parameter name (last word)
                String[] words = param.split("\\s+");
                if (words.length > 0) {
                    String name = words[words.length - 1];
                    // Remove any pointer/array symbols
                    name = name.replaceAll("[*\\[\\]]", "");
                    names.add(name);
                }
            }
        }
        return names;
    }

    /**
     * Match a kernel parameter name to the corresponding data object from inputs/outputs.
     * Uses naming heuristics and size matching.
     */
    private Object matchParameterToDataObject(String paramName, List<Object> inputs, List<Object> outputs) {
        String name = paramName.toLowerCase();

        // Heuristic 1: Output indicators (hb, out, result, dst, output, y, vel for NBody velocity)
        if (name.equals("hb") || name.contains("out") || name.contains("result") ||
            name.contains("dst") || name.equals("y") || name.equals("c") ||
            name.contains("vel") || name.equals("refvel")) {
            // Return first available output
            if (!outputs.isEmpty()) {
                return outputs.get(0);
            }
        }

        // Heuristic 2: Input indicators (x, input, src, vec, in, pos for NBody position)
        if (name.equals("x") || name.contains("input") || name.contains("src") ||
            name.equals("vec") || (name.contains("in") && !name.contains("out")) ||
            name.contains("pos") || name.equals("refpos")) {
            // Return first input
            if (!inputs.isEmpty()) {
                return inputs.get(0);
            }
        }

        // Heuristic 3: Weight/matrix indicators (w, weight, mat, matrix, a, b)
        if (name.equals("w") || name.contains("weight") || name.contains("mat") ||
            name.equals("a") || name.equals("b")) {
            // Return largest input (likely the matrix)
            return findLargestArray(inputs);
        }

        // Fallback: Return first available from either list
        if (!inputs.isEmpty()) return inputs.get(0);
        if (!outputs.isEmpty()) return outputs.get(0);
        return null;
    }

    /**
     * Find the array with smallest size from a list.
     */
    private Object findSmallestArray(List<Object> arrays) {
        if (arrays.isEmpty()) return null;
        Object smallest = arrays.get(0);
        int smallestSize = getArraySize(smallest);
        for (Object arr : arrays) {
            int size = getArraySize(arr);
            if (size < smallestSize) {
                smallestSize = size;
                smallest = arr;
            }
        }
        return smallest;
    }

    /**
     * Find the array with largest size from a list.
     */
    private Object findLargestArray(List<Object> arrays) {
        if (arrays.isEmpty()) return null;
        Object largest = arrays.get(0);
        int largestSize = getArraySize(largest);
        for (Object arr : arrays) {
            int size = getArraySize(arr);
            if (size > largestSize) {
                largestSize = size;
                largest = arr;
            }
        }
        return largest;
    }

    /**
     * Resolve a scalar parameter to its appropriate value based on its name.
     * Maps parameter names like 'n' to input dimension and 'd' to output dimension.
     */
    private int resolveScalarParameter(String paramName, List<Object> inputs, List<Object> outputs,
                                       int[] globalSizes, int paramIndex) {
        // Get actual sizes from arrays
        int inputArraySize = inputs.isEmpty() ? 1024 : getArraySize(inputs.get(0));
        int outputArraySize = outputs.isEmpty() ? 1024 : getArraySize(outputs.get(0));
        int matrixDim = inputs.isEmpty() ? 1024 : determineProblemSize(inputs.get(0));

        // Use parameter name to determine the right value
        String name = paramName.toLowerCase();

        // NBody: 'numBodies' - array size / 4 (each body has x,y,z,w)
        if (name.equals("numbodies") || name.contains("bodies") || name.contains("particles")) {
            int numBodies = inputArraySize / 4;
            System.out.printf("[MCP] resolveScalarParameter: '%s' -> %d (inputArraySize=%d / 4)%n",
                    paramName, numBodies, inputArraySize);
            return numBodies;
        }

        // 'n' typically means input dimension (columns, vector length)
        if (name.equals("n") || name.contains("col") || name.equals("length") ||
            name.contains("input") || name.equals("width")) {
            return inputArraySize;
        }

        // 'd' typically means output dimension (rows, number of outputs)
        if (name.equals("d") || name.contains("row") || name.equals("m") ||
            name.contains("output") || name.equals("height")) {
            return outputArraySize;
        }

        // 'size' is ambiguous — it means matrix dimension for 2D matrix kernels
        // (input is NxN so matrixDim = sqrt(inputArraySize)), but total length for
        // 1D kernels like reduction (input is a flat N-element array).
        // Disambiguate by checking whether the input array is a perfect square:
        //   - Perfect square → 2D matrix kernel → use matrixDim
        //   - Not a perfect square → 1D kernel → use inputArraySize
        if (name.equals("size")) {
            boolean isPerfectSquare = (matrixDim * matrixDim == inputArraySize);
            int resolved = isPerfectSquare ? matrixDim : inputArraySize;
            System.out.printf("[MCP] resolveScalarParameter: 'size' -> %d (%s, inputArraySize=%d, matrixDim=%d)%n",
                    resolved, isPerfectSquare ? "2D square matrix" : "1D array", inputArraySize, matrixDim);
            return resolved;
        }

        // 'localWorkGroupSize' - return the local work size
        if (name.contains("local") || name.contains("workgroup")) {
            return globalSizes.length > 0 ? 32 : 32;  // Default to 32 for Apple M4
        }

        // Fallback: use globalSizes based on parameter index, but be smart about it
        // For matrix-vector: first scalar is usually 'n' (input), second is 'd' (output)
        if (paramIndex == 0) {
            // First scalar param - likely 'n' (input dimension)
            return inputArraySize;
        } else if (paramIndex == 1) {
            // Second scalar param - likely 'd' (output dimension)
            return outputArraySize;
        }

        // Final fallback
        return paramIndex < globalSizes.length ? globalSizes[paramIndex] : globalSizes[0];
    }

    /**
     * Resolve a float scalar parameter to its appropriate value based on its name.
     * Used for physics simulations like NBody (delT, espSqr).
     */
    private float resolveFloatParameter(String paramName, List<Object> inputs, List<Object> outputs) {
        String name = paramName.toLowerCase();

        // NBody: 'delT' - time delta (default to 0.005f as in TornadoVM examples)
        if (name.equals("delt") || name.contains("delta") || name.contains("time") || name.contains("dt")) {
            System.out.printf("[MCP] resolveFloatParameter: '%s' -> 0.005f (time delta)%n", paramName);
            return 0.005f;
        }

        // NBody: 'espSqr' - epsilon squared for softening (default to 500.0f)
        if (name.equals("espsqr") || name.contains("epsilon") || name.contains("soft")) {
            System.out.printf("[MCP] resolveFloatParameter: '%s' -> 500.0f (epsilon squared)%n", paramName);
            return 500.0f;
        }

        // Default fallback
        System.out.printf("[MCP] resolveFloatParameter: '%s' -> 1.0f (default)%n", paramName);
        return 1.0f;
    }

    /**
     * Create appropriate WorkerGrid based on gridConfig dimensions.
     * Supports 1D, 2D, and 3D grids with flexible local work sizes.
     */
    private WorkerGrid createWorkerGrid(MCPKernelOptimizer.GridConfig gridConfig, int[] globalSizes, String optimizedKernel) {
        int[] localSize = null;

        // Use local work size from gridConfig if available
        if (gridConfig != null && gridConfig.localWorkSize() != null && gridConfig.localWorkSize().length > 0) {
            localSize = gridConfig.localWorkSize();
            System.out.printf("[MCP] Using local work size from LLM: %s%n", java.util.Arrays.toString(localSize));
        } else {
            // Fallback to parsing from kernel
            localSize = parseLocalWorkSize(optimizedKernel);
            if (localSize != null) {
                System.out.printf("[MCP] Parsed local work size from kernel: %s%n", java.util.Arrays.toString(localSize));
            }
        }

        // Determine dimensions
        int dims = (gridConfig != null) ? gridConfig.dimensions() : 2;

        // Log pattern if available
        if (gridConfig != null && gridConfig.pattern() != null) {
            System.out.printf("[MCP] Grid pattern: %s%n", gridConfig.pattern());
        }

        // Round up global sizes to be divisible by local sizes (OpenCL requirement)
        int[] roundedGlobalSizes = roundUpGlobalSizes(globalSizes, localSize);
        if (!java.util.Arrays.equals(globalSizes, roundedGlobalSizes)) {
            System.out.printf("[MCP] Rounded global work size: %s -> %s (to be divisible by local)%n",
                    java.util.Arrays.toString(globalSizes), java.util.Arrays.toString(roundedGlobalSizes));
        }

        if (dims == 1) {
            // 1D grid
            WorkerGrid1D workerGrid = new WorkerGrid1D(roundedGlobalSizes[0]);
            if (localSize != null && localSize.length >= 1) {
                workerGrid.setLocalWork(localSize[0], 1, 1);
                System.out.printf("[MCP] PrebuiltTask 1D grid: global=[%d], local=[%d]%n",
                        roundedGlobalSizes[0], localSize[0]);
            } else {
                System.out.printf("[MCP] PrebuiltTask 1D grid: global=[%d], local=default%n", roundedGlobalSizes[0]);
            }
            return workerGrid;
        } else if (dims == 2) {
            // 2D grid (supports non-square)
            int globalX = roundedGlobalSizes[0];
            int globalY = roundedGlobalSizes.length > 1 ? roundedGlobalSizes[1] : roundedGlobalSizes[0];
            WorkerGrid2D workerGrid = new WorkerGrid2D(globalX, globalY);

            if (localSize != null && localSize.length >= 2) {
                workerGrid.setLocalWork(localSize[0], localSize[1], 1);
                System.out.printf("[MCP] PrebuiltTask 2D grid: global=[%d,%d], local=[%d,%d]%n",
                        globalX, globalY, localSize[0], localSize[1]);
            } else if (localSize != null && localSize.length == 1) {
                workerGrid.setLocalWork(localSize[0], localSize[0], 1);
                System.out.printf("[MCP] PrebuiltTask 2D grid: global=[%d,%d], local=[%d,%d]%n",
                        globalX, globalY, localSize[0], localSize[0]);
            } else {
                System.out.printf("[MCP] PrebuiltTask 2D grid: global=[%d,%d], local=default%n", globalX, globalY);
            }
            return workerGrid;
        } else {
            // 3D grid
            int globalX = roundedGlobalSizes[0];
            int globalY = roundedGlobalSizes.length > 1 ? roundedGlobalSizes[1] : roundedGlobalSizes[0];
            int globalZ = roundedGlobalSizes.length > 2 ? roundedGlobalSizes[2] : 1;
            WorkerGrid3D workerGrid = new WorkerGrid3D(globalX, globalY, globalZ);

            if (localSize != null && localSize.length >= 3) {
                workerGrid.setLocalWork(localSize[0], localSize[1], localSize[2]);
                System.out.printf("[MCP] PrebuiltTask 3D grid: global=[%d,%d,%d], local=[%d,%d,%d]%n",
                        globalX, globalY, globalZ, localSize[0], localSize[1], localSize[2]);
            } else if (localSize != null && localSize.length >= 2) {
                workerGrid.setLocalWork(localSize[0], localSize[1], 1);
                System.out.printf("[MCP] PrebuiltTask 3D grid: global=[%d,%d,%d], local=[%d,%d,1]%n",
                        globalX, globalY, globalZ, localSize[0], localSize[1]);
            } else {
                System.out.printf("[MCP] PrebuiltTask 3D grid: global=[%d,%d,%d], local=default%n",
                        globalX, globalY, globalZ);
            }
            return workerGrid;
        }
    }

    /**
     * Round up global work sizes to be evenly divisible by local work sizes.
     * OpenCL requires global_size[i] % local_size[i] == 0.
     */
    private int[] roundUpGlobalSizes(int[] globalSizes, int[] localSizes) {
        if (localSizes == null || localSizes.length == 0) {
            return globalSizes;
        }

        int[] rounded = new int[globalSizes.length];
        for (int i = 0; i < globalSizes.length; i++) {
            int local = (i < localSizes.length) ? localSizes[i] : 1;
            if (local <= 0) local = 1;
            // Round up: ((global + local - 1) / local) * local
            rounded[i] = ((globalSizes[i] + local - 1) / local) * local;
        }
        return rounded;
    }

    /**
     * Configure grid scheduler for MCP-optimized kernels.
     * Parses reqd_work_group_size attribute and sets appropriate global/local dimensions.
     */
    private void configureMCPGridScheduler(String optimizedKernel, String taskId) {
        // Parse reqd_work_group_size(X, Y, Z) from kernel
        Pattern pattern = Pattern.compile("reqd_work_group_size\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)");
        Matcher matcher = pattern.matcher(optimizedKernel);

        if (!matcher.find()) {
            System.out.println("[MCP] No reqd_work_group_size found, using default grid");
            return;
        }

        int localX = Integer.parseInt(matcher.group(1));
        int localY = Integer.parseInt(matcher.group(2));
        int localZ = Integer.parseInt(matcher.group(3));

        // For matrix operations, assume problem size is power-of-2 and matches original launch
        // Try to get problem size from kernel (look for hardcoded 1024 or similar)
        int problemSize = 1024;  // Default for MatrixMultiplication2D 1024x1024
        Pattern sizePattern = Pattern.compile("for\\s*\\([^;]*<\\s*(\\d+)");
        Matcher sizeMatcher = sizePattern.matcher(optimizedKernel);
        if (sizeMatcher.find()) {
            problemSize = Integer.parseInt(sizeMatcher.group(1));
        }

        // Global size = problem size (one thread per output element for tiled kernels)
        long globalX = problemSize;
        long globalY = problemSize;

        System.out.printf("[MCP] Configuring grid: global=[%d,%d], local=[%d,%d]%n",
                globalX, globalY, localX, localY);

        // Create WorkerGrid2D with specified dimensions
        WorkerGrid2D workerGrid = new WorkerGrid2D((int) globalX, (int) globalY);
        workerGrid.setLocalWork(localX, localY, 1);

        // Get task graph name and create grid scheduler
        String taskGraphName = tornadoExecutor.getTaskGraphName();
        String fullTaskName = taskGraphName + "." + taskId;

        GridScheduler gridScheduler = new GridScheduler(fullTaskName, workerGrid);
        executionFrame.setGridScheduler(gridScheduler);
        tornadoExecutor.withGridScheduler(gridScheduler);
    }

    /**
     * Select a graph from the {@link TornadoExecutionPlan} to execute.
     * This method allows developers to select a specific graph from the
     * execution plan to launch. Developers can choose which graph from
     * the input list to use (passed in the constructor).
     *
     * 
     * @since 1.0.9
     * @param graphIndex
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withGraph(int graphIndex) {
        tornadoExecutor.selectGraph(graphIndex);
        if (executionFrame.getGridScheduler() != null) {
            tornadoExecutor.withGridScheduler(executionFrame.getGridScheduler());
        }
        return new WithGraph(this, graphIndex);
    }

    /**
     * Select all graphs from the {@link TornadoExecutionPlan}. This method
     * has an effect if the {@link #withGraph(int)} method was invoked.
     *
     * @since 1.0.9
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withAllGraphs() {
        tornadoExecutor.selectAll();
        return new WithAllGraphs(this);
    }

    /**
     * It invokes the JIT compiler for all immutable tasks-graphs associated to an
     * executor.
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withPreCompilation() {
        tornadoExecutor.withPreCompilation(executionFrame);
        return new WithPreCompilation(this);
    }

    /**
     * It selects a specific device for all immutable tasks graphs associated to an
     * executor.
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withDevice(TornadoDevice device) {
        tornadoExecutor.setDevice(device);
        return new WithDevice(this, device);
    }

    /**
     * Print all operations enabled/disabled from the Execution Plan.
     * 
     * @since 1.0.8
     */
    public void printTraceExecutionPlan() {
        System.out.println(Objects.requireNonNullElse(childLink, this));
    }

    /**
     * Returns a string with all the operations enabled/disabled from the
     * Execution Plan.
     *
     * @since 1.0.8
     */
    public String getTraceExecutionPlan() {
        if (childLink != null) {
            return childLink.toString();
        }
        return toString();
    }

    @Override
    public String toString() {
        return "Root";
    }

    /**
     * It selects a specific device for one particular task of the task-graph.
     *
     * @param taskName
     *     The task-name is identified by the task-graph name followed by a dot (".") and
     *     the task name. For example: "graph.task1".
     * @param device
     *     The device is an instance of a {@link TornadoDevice}
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withDevice(String taskName, TornadoDevice device) {
        tornadoExecutor.setDevice(taskName, device);
        return new WithDevice(this, device);
    }

    /**
     * It enables multiple tasks in a task graph to run concurrently on the same
     * or different devices. Note that the TornadoVM runtime does not check for
     * data dependencies across tasks when using this API call. Thus, it is
     * the responsibility of the programmer to provide tasks with no data dependencies
     * when invoking the method {@link TornadoExecutionPlan#withConcurrentDevices}.
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withConcurrentDevices() {
        tornadoExecutor.withConcurrentDevices();
        return new WithConcurrentDevices(this);
    }

    /**
     * It disables multiple tasks in a task graph to run concurrently on the same
     * or different devices.
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withoutConcurrentDevices() {
        tornadoExecutor.withoutConcurrentDevices();
        return new OffConcurrentDevices(this);
    }

    /**
     * It obtains the device for a specific immutable task-graph. Note that,
     * ideally, different task immutable task-graph could be executed on different
     * devices.
     *
     * @param immutableTaskGraphIndex
     *     Index of a specific immutable task-graph
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoDevice getDevice(int immutableTaskGraphIndex) {
        return tornadoExecutor.getDevice(immutableTaskGraphIndex);
    }

    /**
     * Mark all device buffers that correspond to the current execution plan as free
     * in order for the TornadoVM runtime system to reuse those buffers and avoid
     * continuous device memory deallocation and allocation.
     *
     * <p>
     * Note that, in this context, "free device memory" means the TornadoVM runtime
     * system marks device buffers to be reusable, thus, for the runtime system,
     * device buffers are no longer linked to the current execution plan.
     * </p>
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan freeDeviceMemory() {
        tornadoExecutor.freeDeviceMemory();
        return new WithFreeDeviceMemory(this);
    }

    /**
     * Use a {@link GridScheduler} for thread dispatch. The same GridScheduler will
     * be applied to all tasks within the executor. Note that the grid-scheduler API
     * can specify all workers for each task-graph.
     *
     * @param gridScheduler
     *     {@link GridScheduler}
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withGridScheduler(GridScheduler gridScheduler) {
        boolean isGridRegistered = tornadoExecutor.withGridScheduler(gridScheduler);
        if (!isGridRegistered) {
            // check for the whole set of task-graphs
            isGridRegistered = tornadoExecutor.checkAllTaskGraphsForGridScheduler();
            if (!isGridRegistered) {
                throw new TornadoRuntimeException("[ERROR] GridScheduler Name not registered in any task-graph");
            }
        }
        executionFrame.setGridScheduler(gridScheduler);
        return new WithGridScheduler(this, gridScheduler);
    }

    /**
     * Notify the TornadoVM runtime system to utilize the default thread scheduler.
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withDefaultScheduler() {
        tornadoExecutor.withDefaultScheduler();
        return new WithDefaultScheduler(this);
    }


    /**
     * Enable batch processing. TornadoVM will split the iteration space in smaller
     * batches (with batch size specified by the user). This is used mainly when
     * users want to execute big data applications that do not fit on the device's
     * global memory.
     *
     * @param batchSize
     *     String in the format a number + "MB" Example "512MB".
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withBatch(String batchSize) {
        tornadoExecutor.withBatch(batchSize);
        return new WithBatch(this, batchSize);
    }

    /**
     * Enables the profiler. The profiler includes options to query device kernel
     * time, data transfers and compilation at different stages (JIT, driver
     * compilation, Graal, etc.).
     *
     * @param profilerMode
     *     {@link ProfilerMode}
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withProfiler(ProfilerMode profilerMode) {
        executionFrame.setProfilerMode(profilerMode);
        return new WithProfiler(this, profilerMode);
    }

    /**
     * Disables the profiler if previous execution plan had the profiler enabled.
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withoutProfiler() {
        executionFrame.setProfilerOff();
        return new OffProfiler(this);
    }

    /**
     * This method sets a limit to the amount of memory used on the target
     * hardware accelerator. The TornadoVM runtime will check that the
     * current instance of the {@link TornadoExecutionPlan} does not exceed
     * the limit that was specified.
     *
     * @param memoryLimit
     *     Specify the limit in a string format. E.g., "1GB", "512MB".
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withMemoryLimit(String memoryLimit) {
        tornadoExecutor.withMemoryLimit(memoryLimit);
        return new WithMemoryLimit(this, memoryLimit);
    }

    /**
     * It disables the memory limit for the current instance of an
     * {@link TornadoExecutionPlan}. This is the default action.
     * If the memory limit is not set, then the maximum memory to use
     * is set to the maximum buffer allocation (e.g., 1/4 of the total
     * capacity using the OpenCL backend), or the maximum memory available
     * on the target device.
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withoutMemoryLimit() {
        tornadoExecutor.withoutMemoryLimit();
        return new OffMemoryLimit(this);
    }

    /**
     * Reset the execution context for the current execution plan. The TornadoVM
     * runtime system will clean the code cache and all events associated with the
     * current execution. It resets the internal GPU/FPGA/CPU execution context to
     * its default values.
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan resetDevice() {
        tornadoExecutor.resetDevice();
        return new WithResetDevice(this);
    }

    /**
     * Obtains the ID that was assigned to the execution plan.
     */
    public long getId() {
        return executionFrame.getExecutionPlanId();
    }

    /**
     * Obtains the total number of execution plans instantiated in a TornadoVM application.
     */
    public long getGlobalExecutionPlansCounter() {
        return globalExecutionPlanCounter.get();
    }

    /**
     * Clean all events associated with previous executions.
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan clearProfiles() {
        tornadoExecutor.clearProfiles();
        return new WithClearProfiles(this);
    }

    /**
     * Enable printing of the Thread-Block Deployment for the generated kernels.
     *
     * @since 1.0.2
     * 
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withThreadInfo() {
        tornadoExecutor.withThreadInfo();
        return new WithThreadInfo(this);
    }

    /**
     * Disable printing of the Thread-Block Deployment for the generated kernels.
     *
     * @since 1.0.2
     * 
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withoutThreadInfo() {
        tornadoExecutor.withoutThreadInfo();
        return new OffThreadInfo(this);
    }

    /**
     * Enable printing of the generated kernels for each task in a task-graph.
     *
     * @since 1.0.2
     * 
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withPrintKernel() {
        tornadoExecutor.withPrintKernel();
        return new WithPrintKernel(this);
    }

    /**
     * Disable printing of the generated kernels for each task in a task-graph.
     * 
     * @since 1.0.2
     *
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withoutPrintKernel() {
        tornadoExecutor.withoutPrintKernel();
        return new OffPrintKernel(this);
    }

    /**
     * Set compiler flags for each backend.
     * 
     * @param backend
     *     {@link TornadoVMBackendType}
     * @param compilerFlags
     *     {@link String}
     * @since 1.0.7
     * @return {@link TornadoExecutionPlan}
     */
    public TornadoExecutionPlan withCompilerFlags(TornadoVMBackendType backend, String compilerFlags) {
        tornadoExecutor.withCompilerFlags(backend, compilerFlags);
        return new WithCompilerFlags(this, compilerFlags);
    }

    /**
     * @since 1.0.4
     * 
     * @throws {@link
     *     TornadoExecutionPlanException}
     */
    @Override
    public void close() throws TornadoExecutionPlanException {
        tornadoExecutor.freeDeviceMemory();
    }

    /**
     * It returns the current memory usage on the device in bytes.
     * 
     * @return long
     *     Number of bytes used.
     */
    public long getCurrentDeviceMemoryUsage() {
        return tornadoExecutor.getCurrentDeviceMemoryUsage();
    }

    public TornadoExecutionResult getPlanResult(int index) {
        if (index >= planResults.size()) {
            throw new TornadoRuntimeException("[ERROR] Execution result not found");
        }
        return planResults.get(index);
    }

    /**
     * This function maps the device memory region that corresponds to a TornadoVM object to another on-device memory region.
     * This call instructs the TornadoVM runtime to avoid transferring data between `device` -> `host` -> `device`. Instead,
     * it can update the corresponding device pointers.
     *
     * <p>
     * The semantics are as follows: there is the source object, and the destination object. This call maps the dest object
     * to the source object from a given offset. The source object is passed from the task-graph `fromGraphIndex`, and
     * the destination object is taken from the `toGraphIndex`. This method can be invoked in a multi-task-graph execution
     * plan. It will not work if there is only one task-graph in the execution plan.
     * </p>
     * 
     * @param destTornadoArray
     * @param srcTornadoArray
     * @param offset
     * @param fromGraphIndex
     * @param toGraphIndex
     *
     * @since v1.1.0
     */
    public void mapOnDeviceMemoryRegion(Object destTornadoArray, Object srcTornadoArray, long offset, int fromGraphIndex, int toGraphIndex) {
        tornadoExecutor.mapOnDeviceMemoryRegion(destTornadoArray, srcTornadoArray, offset, fromGraphIndex, toGraphIndex);
    }

    /**
     * This function allows developers to warm up the whole execution plan before running it. This covers
     * copy in and out data, compiling all tasks and executing all tasks once for the specified amount
     * of time.
     * 
     * @param milliseconds
     *     Amount of time to warm up the execution plan. This amount means that the execution plan will run,
     *     at least for the specified amount of time. if the tasks within the task-graphs
     *     takes longer to execute, in a second run, the code will not be dispatched.
     * @return {@link TornadoExecutionPlan}
     * 
     * @throws {@link
     *     InterruptedException}
     */
    public TornadoExecutionPlan withWarmUpTime(long milliseconds) throws InterruptedException {
        if (milliseconds < 0) {
            throw new TornadoRuntimeException("[ERROR] Warm-up time cannot be negative");
        }
        tornadoExecutor.withWarmUpTime(milliseconds, executionFrame);
        return new WithWarmUpTime(this, milliseconds);
    }

    /**
     * This function allows developers to warm up the whole execution plan before running it. This covers
     * copy in and out data, compiling all tasks and executing all tasks once for the specified amount
     * of time.
     *
     * @param iterations
     *     Number of iterations to run the whole execution plan as warm-up.
     * @return {@link TornadoExecutionPlan}
     *
     */
    public TornadoExecutionPlan withWarmUpIterations(int iterations) {
        if (iterations < 0) {
            throw new TornadoRuntimeException("[ERROR] Warm-up time cannot be negative");
        }
        tornadoExecutor.withWarmUpIterations(iterations, executionFrame);
        return new WithWarmUpIterations(this, iterations);
    }

    // =========================================================================
    // MCP Kernel Comparison API
    // =========================================================================

    /**
     * Get the generated kernel source code for a specific task after execution.
     * This method is used for MCP kernel comparison - to extract the kernel
     * that TornadoVM generated so it can be sent to the MCP server for optimization.
     *
     * <p>
     * This method should be called AFTER at least one execution of the plan,
     * otherwise the kernel may not have been generated yet.
     * </p>
     *
     * @param taskGraphIndex Index of the task graph (0 for first/only graph)
     * @param taskId The task ID within the graph (e.g., "t0")
     * @return The kernel source code as a string, or null if not found
     *
     * @since 1.1.3
     */
    public String getGeneratedKernelSource(int taskGraphIndex, String taskId) {
        return tornadoExecutor.getGeneratedKernelSource(taskGraphIndex, taskId, executionFrame.getExecutionPlanId());
    }

    /**
     * Get the generated kernel source code for a task in the first task graph.
     * Convenience method for single-graph execution plans.
     *
     * @param taskId The task ID within the graph (e.g., "t0")
     * @return The kernel source code as a string, or null if not found
     *
     * @since 1.1.3
     */
    public String getGeneratedKernelSource(String taskId) {
        return getGeneratedKernelSource(0, taskId);
    }

    /**
     * Replace the kernel for a specific task with new source code.
     * This method is used for MCP kernel comparison - to run an optimized kernel
     * under the exact same conditions as the original.
     *
     * <p>
     * After calling this method, the next execution will use the new kernel.
     * The data buffers and work dimensions remain the same, ensuring a fair comparison.
     * </p>
     *
     * @param taskGraphIndex Index of the task graph (0 for first/only graph)
     * @param taskId The task ID within the graph (e.g., "t0")
     * @param newKernelSource The optimized kernel source code
     * @return true if replacement was successful, false otherwise
     *
     * @since 1.1.3
     */
    public boolean replaceKernelSource(int taskGraphIndex, String taskId, String newKernelSource) {
        return tornadoExecutor.replaceKernelSource(taskGraphIndex, taskId, newKernelSource, executionFrame.getExecutionPlanId());
    }

    /**
     * Replace the kernel for a task in the first task graph.
     * Convenience method for single-graph execution plans.
     *
     * @param taskId The task ID within the graph (e.g., "t0")
     * @param newKernelSource The optimized kernel source code
     * @return true if replacement was successful, false otherwise
     *
     * @since 1.1.3
     */
    public boolean replaceKernelSource(String taskId, String newKernelSource) {
        return replaceKernelSource(0, taskId, newKernelSource);
    }
}
