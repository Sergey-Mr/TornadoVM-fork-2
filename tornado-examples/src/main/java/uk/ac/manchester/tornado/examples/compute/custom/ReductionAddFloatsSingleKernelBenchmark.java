package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;
import java.util.Random;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

/**
 * Single-kernel benchmark for Reduction Add Floats.
 * Measures ONLY kernel execution time - run once per kernel for fair comparison.
 *
 * Note: Reduction kernels are more complex as they involve partial reductions
 * and a final combination step. This benchmark measures the partial reduction kernel.
 *
 * Usage: java ... ReductionAddFloatsSingleKernelBenchmark <kernel.cl> [size]
 *
 * Default size: 16777216 (16M elements)
 */
public class ReductionAddFloatsSingleKernelBenchmark {

    private static final int DEFAULT_SIZE = 16777216; // 16M elements
    private static final int LOCAL_WORK_SIZE = 256;
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;
    private static final String ENTRY_POINT = "reductionAddFloats";
    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array) {
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, RANDOM.nextFloat());
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: ReductionAddFloatsSingleKernelBenchmark <kernel.cl> [size]");
            System.exit(1);
        }

        String kernelPath = args[0];
        int size = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_SIZE;

        // Calculate number of work groups for partial reduction
        int numWorkGroups = (size + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;

        System.out.println("Kernel: " + kernelPath);
        System.out.println("Array size: " + size);
        System.out.println("Work groups: " + numWorkGroups);
        System.out.println("Local work size: " + LOCAL_WORK_SIZE);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        // Input array and partial results array (one per work group)
        FloatArray input = new FloatArray(size);
        FloatArray partialSums = new FloatArray(numWorkGroups);

        fillRandomData(input);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // Kernel signature for reduction: reductionAddFloats(input, partialSums)
        // The generated kernel does partial reductions per work group
        AccessorParameters accessors = new AccessorParameters(2);
        accessors.set(0, input, Access.READ_ONLY);
        accessors.set(1, partialSums, Access.READ_WRITE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, partialSums);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // 1D grid
        WorkerGrid1D worker = new WorkerGrid1D(size);
        worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        ArrayList<Long> kernelTimes = new ArrayList<>();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler);

            // Warmup
            System.out.println("Warming up...");
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                // Reset partial sums
                for (int j = 0; j < numWorkGroups; j++) {
                    partialSums.set(j, 0.0f);
                }
                plan.execute();
            }

            // Measure kernel time only
            System.out.println("Measuring kernel time...");
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                // Reset partial sums
                for (int j = 0; j < numWorkGroups; j++) {
                    partialSums.set(j, 0.0f);
                }

                TornadoExecutionResult result = plan
                        .withProfiler(ProfilerMode.SILENT)
                        .execute();

                TornadoProfilerResult profilerResult = result.getProfilerResult();
                long kernelTime = profilerResult.getDeviceKernelTime();
                kernelTimes.add(kernelTime);
            }
        }

        // Final reduction on host (sum partial sums)
        float finalSum = 0.0f;
        for (int i = 0; i < numWorkGroups; i++) {
            finalSum += partialSums.get(i);
        }

        LongSummaryStatistics stats = kernelTimes.stream().mapToLong(Long::longValue).summaryStatistics();

        // GB/s bandwidth metric (read size floats)
        long bytesProcessed = (long) size * Float.BYTES;
        double gbPerSec = (bytesProcessed * 1e-9) / (stats.getAverage() * 1e-9);

        // Million elements per second
        double mElementsPerSec = (size * 1e-6) / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY)");
        System.out.println("==========================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Array size: %d elements%n", size);
        System.out.printf("Final sum: %.2f%n", finalSum);
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("Bandwidth: %.2f GB/s%n", gbPerSec);
        System.out.printf("MElements/s: %.2f%n", mElementsPerSec);
    }
}
