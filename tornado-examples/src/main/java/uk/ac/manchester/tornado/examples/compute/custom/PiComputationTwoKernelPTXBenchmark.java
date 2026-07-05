package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * PTX Benchmark for PiComputation with two-kernel reduction.
 *
 * This benchmark loads a PTX file containing both:
 * 1. computePi kernel (main computation with local memory reduction)
 * 2. rAdd kernel (final sequential reduction of partial sums)
 *
 * Usage: PiComputationTwoKernelPTXBenchmark <kernel.ptx> [size]
 *
 * The PTX file must contain both kernels with these entry points:
 * - s0_t0_computepi_arrays_floatarray_arrays_floatarray
 * - s0_t1_radd_arrays_floatarray_<partialSumCount>
 *
 * IMPORTANT: The reduce kernel function name includes the partial sum count.
 * Default size: 8192 (matches original PiComputation, produces 9 partial sums)
 */
public class PiComputationTwoKernelPTXBenchmark {

    private static final int DEFAULT_SIZE = 8192;  // Must match PTX generation size
    private static final int LOCAL_WORK_SIZE = 1024;  // Must match shared memory size in PTX
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;

    // Entry points must be lowercase to match TornadoVM naming
    private static final String COMPUTE_ENTRY = "computepi";
    private static final String REDUCE_ENTRY = "radd";

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: PiComputationTwoKernelPTXBenchmark <kernel.ptx> [size]");
            System.out.println("  Default size: " + DEFAULT_SIZE);
            System.out.println();
            System.out.println("The PTX file must contain both compute and reduce kernels.");
            System.exit(1);
        }

        String kernelPath = args[0];
        int size = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_SIZE;

        // Calculate number of workgroups (partial sums)
        int numWorkGroups = (size + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;

        System.out.println("=== Pi Computation Two-Kernel PTX Benchmark ===");
        System.out.println("Kernel: " + kernelPath);
        System.out.println("Number of terms: " + size);
        System.out.println("Local work size: " + LOCAL_WORK_SIZE);
        System.out.println("Number of workgroups: " + numWorkGroups);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        // Allocate arrays
        FloatArray input = new FloatArray(size);
        // Result array needs space for partial sums: result[0] + result[1..numWorkGroups]
        FloatArray result = new FloatArray(numWorkGroups + 1);

        input.init(0.0f);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // === Task 1: Compute kernel ===
        // PTX signature: (kernel_context, input, result) - kernel_context added by TornadoVM
        AccessorParameters computeAccessors = new AccessorParameters(2);
        computeAccessors.set(0, input, Access.READ_ONLY);
        computeAccessors.set(1, result, Access.READ_WRITE);

        TaskGraph computeGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, result)
                .prebuiltTask("t0", COMPUTE_ENTRY, kernelPath, computeAccessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, result);

        ImmutableTaskGraph computeSnapshot = computeGraph.snapshot();

        WorkerGrid1D computeWorker = new WorkerGrid1D(size);
        computeWorker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
        GridScheduler computeScheduler = new GridScheduler("s0.t0", computeWorker);

        // === Task 2: Reduce kernel ===
        // PTX function: s0_t1_radd_arrays_floatarray_<size>
        // Uses same task graph name "s0" but different task "t1" to match PTX naming
        AccessorParameters reduceAccessors = new AccessorParameters(2);
        reduceAccessors.set(0, result, Access.READ_WRITE);
        reduceAccessors.set(1, Long.valueOf(numWorkGroups + 1), Access.NONE);

        TaskGraph reduceGraph = new TaskGraph("s0")  // Same as compute graph to match PTX naming
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, result)
                .prebuiltTask("t1", REDUCE_ENTRY, kernelPath, reduceAccessors)  // t1 to match s0_t1_radd
                .transferToHost(DataTransferMode.EVERY_EXECUTION, result);

        ImmutableTaskGraph reduceSnapshot = reduceGraph.snapshot();

        WorkerGrid1D reduceWorker = new WorkerGrid1D(1);
        GridScheduler reduceScheduler = new GridScheduler("s0.t1", reduceWorker);  // s0.t1 to match

        ArrayList<Long> kernelTimes = new ArrayList<>();

        try (TornadoExecutionPlan computePlan = new TornadoExecutionPlan(computeSnapshot);
             TornadoExecutionPlan reducePlan = new TornadoExecutionPlan(reduceSnapshot)) {

            computePlan.withDevice(device).withGridScheduler(computeScheduler);
            reducePlan.withDevice(device).withGridScheduler(reduceScheduler);

            // Warmup
            System.out.println("Warming up...");
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                result.init(0.0f);
                computePlan.execute();
                reducePlan.execute();
            }

            // Benchmark
            System.out.println("Measuring kernel time...");
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                result.init(0.0f);

                TornadoExecutionResult computeResult = computePlan
                        .withProfiler(ProfilerMode.SILENT)
                        .execute();

                TornadoExecutionResult reduceResult = reducePlan
                        .withProfiler(ProfilerMode.SILENT)
                        .execute();

                long computeTime = computeResult.getProfilerResult().getDeviceKernelTime();
                long reduceTime = reduceResult.getProfilerResult().getDeviceKernelTime();
                kernelTimes.add(computeTime + reduceTime);
            }
        }

        LongSummaryStatistics stats = kernelTimes.stream()
                .mapToLong(Long::longValue)
                .summaryStatistics();

        float piValue = result.get(0) * 4.0f;
        double mtermsPerSec = (size * 1e-6) / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY - both compute + reduce)");
        System.out.println("===================================================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Terms: %d%n", size);
        System.out.printf("Pi value: %.10f%n", piValue);
        System.out.printf("Pi error: %.2e%n", Math.abs(Math.PI - piValue));
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("Million Terms/s: %.2f%n", mtermsPerSec);
    }
}
