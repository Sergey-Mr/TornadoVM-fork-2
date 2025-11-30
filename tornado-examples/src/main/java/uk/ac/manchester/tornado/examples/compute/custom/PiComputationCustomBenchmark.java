package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Benchmarks TornadoVM's generated Pi computation (reduction) kernel against
 * a user-provided OpenCL implementation using {@code prebuiltTask}.
 *
 * <p>
 * How to run:
 * </p>
 * <code>
 * java --enable-preview @tornado-argfile -cp /tmp/tornado-custom-classes:dist/tornado-sdk/tornado-sdk-1.1.2-dev-1992554/share/java/tornado/* \
 *   uk.ac.manchester.tornado.examples.compute.custom.PiComputationCustomBenchmark \
 *   kernels/picomputation_generated.cl kernels/picomputation_custom.cl
 * </code>
 */
public final class PiComputationCustomBenchmark {

    private static final String DEFAULT_GENERATED_KERNEL = "kernels/picomputation_generated.cl";
    private static final String DEFAULT_CUSTOM_KERNEL = "kernels/picomputation_custom.cl";
    private static final String ENTRY_POINT = "computePi";

    private static final int DEFAULT_SIZE = 8192;
    private static final int WARM_UP_ITERATIONS = 20;
    private static final int BENCHMARK_ITERATIONS = 50;
    private static final float EPSILON = 0.001f;

    private PiComputationCustomBenchmark() {
        // utility class
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int size = (args.length >= 1) ? Integer.parseInt(args[0]) : DEFAULT_SIZE;
        String generatedKernelPath = (args.length >= 2) ? args[1] : DEFAULT_GENERATED_KERNEL;
        String customKernelPath = (args.length >= 3) ? args[2] : DEFAULT_CUSTOM_KERNEL;

        System.out.println("Pi Computation Benchmark Configuration");
        System.out.println("======================================");
        System.out.println("Array size       : " + size);
        System.out.println("Generated kernel : " + generatedKernelPath);
        System.out.println("Custom kernel    : " + customKernelPath);
        System.out.println();

        FloatArray input = new FloatArray(size);
        FloatArray resultSeq = new FloatArray(1);
        FloatArray resultGenerated = new FloatArray(1);
        FloatArray resultCustom = new FloatArray(1);

        // Initialize input (zeros for this computation)
        input.init(0.0f);

        // Run sequential baseline
        warmUpSequential(input, resultSeq, size);
        ArrayList<Long> sequentialSamples = benchmarkSequential(input, resultSeq, size);
        float piSeq = resultSeq.get(0) * 4.0f;

        // Run generated kernel
        BenchmarkResult generatedResult = benchmarkKernel("piGenerated", generatedKernelPath,
                input, resultGenerated, size);
        float piGenerated = resultGenerated.get(0) * 4.0f;

        // Run custom kernel
        BenchmarkResult customResult = benchmarkKernel("piCustom", customKernelPath,
                input, resultCustom, size);
        float piCustom = resultCustom.get(0) * 4.0f;

        // Validate results
        boolean generatedValid = Math.abs(piSeq - piGenerated) < EPSILON;
        boolean customValid = Math.abs(piSeq - piCustom) < EPSILON;

        System.out.println("Pi Values:");
        System.out.printf("  Sequential: %.6f%n", piSeq);
        System.out.printf("  Generated : %.6f (error: %.6f)%n", piGenerated, Math.abs(piSeq - piGenerated));
        System.out.printf("  Custom    : %.6f (error: %.6f)%n", piCustom, Math.abs(piSeq - piCustom));
        System.out.println();

        if (!generatedValid) {
            System.out.println("[WARN] Generated kernel result does not match the sequential baseline.");
        }
        if (!customValid) {
            System.out.println("[WARN] Custom kernel result does not match the sequential baseline.");
        }

        // Print results
        LongSummaryStatistics sequentialStats = sequentialSamples.stream().mapToLong(Long::longValue).summaryStatistics();
        LongSummaryStatistics generatedStats = generatedResult.timingStats();
        LongSummaryStatistics customStats = customResult.timingStats();

        double sequentialMs = sequentialStats.getAverage() / 1_000_000.0;
        double generatedMs = generatedStats.getAverage() / 1_000_000.0;
        double customMs = customStats.getAverage() / 1_000_000.0;

        System.out.println("Benchmark Results");
        System.out.println("=================");
        System.out.printf("Sequential: avg=%.3f ms min=%.3f ms max=%.3f ms%n",
                sequentialMs, sequentialStats.getMin() / 1_000_000.0, sequentialStats.getMax() / 1_000_000.0);
        System.out.printf("Generated : avg=%.3f ms min=%.3f ms max=%.3f ms%n",
                generatedMs, generatedStats.getMin() / 1_000_000.0, generatedStats.getMax() / 1_000_000.0);
        System.out.printf("Custom    : avg=%.3f ms min=%.3f ms max=%.3f ms%n",
                customMs, customStats.getMin() / 1_000_000.0, customStats.getMax() / 1_000_000.0);

        System.out.println();
        System.out.printf("Speedup (Generated vs Sequential): %.2fx%n", sequentialStats.getAverage() / (double) generatedStats.getAverage());
        System.out.printf("Speedup (Custom vs Sequential)   : %.2fx%n", sequentialStats.getAverage() / (double) customStats.getAverage());
        System.out.printf("Speedup (Custom vs Generated)    : %.2fx%n", generatedStats.getAverage() / (double) customStats.getAverage());

        if (customStats.getAverage() < generatedStats.getAverage()) {
            System.out.printf("Custom kernel is faster by %.2fx%n", generatedStats.getAverage() / (double) customStats.getAverage());
        } else {
            System.out.printf("Generated kernel is faster by %.2fx%n", customStats.getAverage() / (double) generatedStats.getAverage());
        }
    }

    private static void computePiSequential(FloatArray input, FloatArray result, int size) {
        result.set(0, 0.0f);
        for (int i = 1; i < size; i++) {
            float value = input.get(i) + (float) (Math.pow(-1, i + 1) / (2 * i - 1));
            result.set(0, result.get(0) + value);
        }
    }

    private static void warmUpSequential(FloatArray input, FloatArray result, int size) {
        for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
            computePiSequential(input, result, size);
        }
    }

    private static ArrayList<Long> benchmarkSequential(FloatArray input, FloatArray result, int size) {
        ArrayList<Long> times = new ArrayList<>(BENCHMARK_ITERATIONS);
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            computePiSequential(input, result, size);
            long end = System.nanoTime();
            times.add(end - start);
        }
        return times;
    }

    private static BenchmarkResult benchmarkKernel(String graphName, String kernelPath,
            FloatArray input, FloatArray result, int size) throws TornadoExecutionPlanException {

        AccessorParameters accessors = new AccessorParameters(2);
        accessors.set(0, input, Access.READ_ONLY);
        accessors.set(1, result, Access.READ_WRITE);

        TaskGraph graph = new TaskGraph(graphName)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, result)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, result);

        ImmutableTaskGraph snapshot = graph.snapshot();

        ArrayList<Long> times = new ArrayList<>(BENCHMARK_ITERATIONS);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            // Warm up
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                result.init(0.0f);
                plan.execute();
            }

            // Benchmark
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                result.init(0.0f);
                long start = System.nanoTime();
                plan.execute();
                long end = System.nanoTime();
                times.add(end - start);
            }
        }

        return new BenchmarkResult(times.stream().mapToLong(Long::longValue).summaryStatistics());
    }

    private static class BenchmarkResult {
        private final LongSummaryStatistics timingStats;

        public BenchmarkResult(LongSummaryStatistics timingStats) {
            this.timingStats = timingStats;
        }

        public LongSummaryStatistics timingStats() {
            return timingStats;
        }
    }
}
