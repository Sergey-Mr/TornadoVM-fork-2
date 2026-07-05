package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Benchmark for TornadoVM-generated Pi Computation kernel.
 * This uses the actual TornadoVM task graph (not prebuilt) to measure
 * the generated reduction kernel performance.
 *
 * Usage: PiComputationGeneratedBenchmark [size]
 * Default size: 16777216 (16M terms)
 */
public class PiComputationGeneratedBenchmark {

    private static final int DEFAULT_SIZE = 16777216;
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;

    public static void computePi(FloatArray input, @Reduce FloatArray result) {
        for (@Parallel int i = 1; i < input.getSize(); i++) {
            float value = TornadoMath.pow(-1, i + 1) / (2 * i - 1);
            result.set(0, result.get(0) + value);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int size = (args.length > 0) ? Integer.parseInt(args[0]) : DEFAULT_SIZE;

        System.out.println("=== Pi Computation Generated Kernel Benchmark ===");
        System.out.println("Number of terms: " + size);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        FloatArray input = new FloatArray(size);
        FloatArray result = new FloatArray(1);

        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input)
                .task("t0", PiComputationGeneratedBenchmark::computePi, input, result)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, result);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutableTaskGraph)) {
            System.out.println("Device: " + plan.getDevice(0));

            // Warmup
            System.out.println("Warming up...");
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                result.set(0, 0.0f);
                plan.execute();
            }

            // Benchmark with profiler
            System.out.println("Measuring kernel time...");
            ArrayList<Long> kernelTimes = new ArrayList<>();

            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                result.set(0, 0.0f);
                TornadoExecutionResult execResult = plan
                        .withProfiler(ProfilerMode.SILENT)
                        .execute();
                long kernelTime = execResult.getProfilerResult().getDeviceKernelTime();
                kernelTimes.add(kernelTime);
            }

            LongSummaryStatistics stats = kernelTimes.stream()
                    .mapToLong(Long::longValue)
                    .summaryStatistics();

            float piValue = result.get(0) * 4.0f;
            double mtermsPerSec = (size * 1e-6) / (stats.getAverage() * 1e-9);

            System.out.println();
            System.out.println("Results (KERNEL TIME ONLY)");
            System.out.println("==========================");
            System.out.printf("Terms: %d%n", size);
            System.out.printf("Pi value: %.10f%n", piValue);
            System.out.printf("Pi error: %.2e%n", Math.abs(Math.PI - piValue));
            System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
            System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
            System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
            System.out.printf("Million Terms/s: %.2f%n", mtermsPerSec);
        }
    }
}
