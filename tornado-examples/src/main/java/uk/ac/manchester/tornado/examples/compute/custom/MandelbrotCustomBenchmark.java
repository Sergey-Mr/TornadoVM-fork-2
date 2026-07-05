package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;

/**
 * Benchmarks TornadoVM's generated Mandelbrot kernel against a user-provided
 * OpenCL implementation using {@code prebuiltTask}.
 *
 * <p>
 * How to run:
 * </p>
 * <code>
 * # Using the comparison script:
 * ./scripts/compare_kernels.sh MandelbrotCustomBenchmark mandelbrot
 *
 * # Or directly:
 * java --enable-preview @tornado-argfile -cp /tmp/tornado-custom-classes:dist/tornado-sdk/tornado-sdk-1.1.2-dev-1992554/share/java/tornado/* \
 *   uk.ac.manchester.tornado.examples.compute.custom.MandelbrotCustomBenchmark \
 *   kernels/mandelbrot_generated.cl kernels/mandelbrot_custom.cl
 * </code>
 */
public final class MandelbrotCustomBenchmark {

    private static final String DEFAULT_GENERATED_KERNEL = "kernels/mandelbrot_generated.cl";
    private static final String DEFAULT_CUSTOM_KERNEL = "kernels/mandelbrot_custom.cl";
    private static final String ENTRY_POINT = "mandelbrotTornado";

    private static final int DEFAULT_SIZE = 1024;
    private static final int ITERATIONS = 10000;
    private static final int WARM_UP_ITERATIONS = 10;
    private static final int BENCHMARK_ITERATIONS = 30;

    private MandelbrotCustomBenchmark() {
        // utility class
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int size = (args.length >= 1) ? Integer.parseInt(args[0]) : DEFAULT_SIZE;
        String generatedKernelPath = (args.length >= 2) ? args[1] : DEFAULT_GENERATED_KERNEL;
        String customKernelPath = (args.length >= 3) ? args[2] : DEFAULT_CUSTOM_KERNEL;

        System.out.println("Mandelbrot Benchmark Configuration");
        System.out.println("===================================");
        System.out.println("Image size       : " + size + " x " + size);
        System.out.println("Max iterations   : " + ITERATIONS);
        System.out.println("Generated kernel : " + generatedKernelPath);
        System.out.println("Custom kernel    : " + customKernelPath);
        System.out.println();

        ShortArray sequentialOutput = new ShortArray(size * size);
        ShortArray generatedOutput = new ShortArray(size * size);
        ShortArray customOutput = new ShortArray(size * size);

        // Run sequential baseline
        warmUpSequential(sequentialOutput, size);
        ArrayList<Long> sequentialSamples = benchmarkSequential(sequentialOutput, size);

        // Run generated kernel
        BenchmarkResult generatedResult = benchmarkKernel("mandelbrotGenerated", generatedKernelPath,
                generatedOutput, size);

        // Run custom kernel
        BenchmarkResult customResult = benchmarkKernel("mandelbrotCustom", customKernelPath,
                customOutput, size);

        // Validate results
        boolean generatedValid = validate(sequentialOutput, generatedOutput, size);
        boolean customValid = validate(sequentialOutput, customOutput, size);

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

        long totalPixels = (long) size * size;
        double sequentialMPixelsPerSec = (totalPixels / 1_000_000.0) / (sequentialStats.getAverage() / 1_000_000_000.0);
        double generatedMPixelsPerSec = (totalPixels / 1_000_000.0) / (generatedStats.getAverage() / 1_000_000_000.0);
        double customMPixelsPerSec = (totalPixels / 1_000_000.0) / (customStats.getAverage() / 1_000_000_000.0);

        System.out.println();
        System.out.println("Benchmark Results");
        System.out.println("=================");
        System.out.printf("Sequential: avg=%.3f ms min=%.3f ms max=%.3f ms (%.2f MPixels/s)%n",
                sequentialMs, sequentialStats.getMin() / 1_000_000.0, sequentialStats.getMax() / 1_000_000.0, sequentialMPixelsPerSec);
        System.out.printf("Generated : avg=%.3f ms min=%.3f ms max=%.3f ms (%.2f MPixels/s)%n",
                generatedMs, generatedStats.getMin() / 1_000_000.0, generatedStats.getMax() / 1_000_000.0, generatedMPixelsPerSec);
        System.out.printf("Custom    : avg=%.3f ms min=%.3f ms max=%.3f ms (%.2f MPixels/s)%n",
                customMs, customStats.getMin() / 1_000_000.0, customStats.getMax() / 1_000_000.0, customMPixelsPerSec);

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

    private static void mandelbrotSequential(ShortArray output, int size) {
        float space = 2.0f / size;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float Zr = 0.0f;
                float Zi = 0.0f;
                float Cr = (j * space - 1.5f);
                float Ci = (i * space - 1.0f);
                float ZrN = 0;
                float ZiN = 0;
                int y = 0;

                for (int ii = 0; ii < ITERATIONS; ii++) {
                    if (ZiN + ZrN <= 4.0f) {
                        Zi = 2.0f * Zr * Zi + Ci;
                        Zr = ZrN - ZiN + Cr;
                        ZiN = Zi * Zi;
                        ZrN = Zr * Zr;
                        y++;
                    } else {
                        break;
                    }
                }
                short r = (short) ((y * 255) / ITERATIONS);
                output.set(i * size + j, r);
            }
        }
    }

    private static void warmUpSequential(ShortArray output, int size) {
        for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
            mandelbrotSequential(output, size);
        }
    }

    private static ArrayList<Long> benchmarkSequential(ShortArray output, int size) {
        ArrayList<Long> times = new ArrayList<>(BENCHMARK_ITERATIONS);
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            mandelbrotSequential(output, size);
            long end = System.nanoTime();
            times.add(end - start);
        }
        return times;
    }

    private static BenchmarkResult benchmarkKernel(String graphName, String kernelPath,
            ShortArray output, int size) throws TornadoExecutionPlanException {

        AccessorParameters accessors = new AccessorParameters(2);
        accessors.set(0, Integer.valueOf(size), Access.NONE);
        accessors.set(1, output, Access.WRITE_ONLY);

        TaskGraph graph = new TaskGraph(graphName)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, output)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();

        WorkerGrid2D workerGrid = new WorkerGrid2D(size, size);
        workerGrid.setLocalWork(16, 16, 1);
        GridScheduler scheduler = new GridScheduler(graphName + ".t0", workerGrid);

        ArrayList<Long> times = new ArrayList<>(BENCHMARK_ITERATIONS);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withGridScheduler(scheduler);

            // Warm up
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                plan.execute();
            }

            // Benchmark
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                long start = System.nanoTime();
                plan.execute();
                long end = System.nanoTime();
                times.add(end - start);
            }
        }

        return new BenchmarkResult(times.stream().mapToLong(Long::longValue).summaryStatistics());
    }

    private static boolean validate(ShortArray reference, ShortArray candidate, int size) {
        int errors = 0;
        for (int i = 0; i < size * size; i++) {
            short refVal = reference.get(i);
            short candVal = candidate.get(i);
            // Allow small difference due to floating point variations
            if (Math.abs(refVal - candVal) > 1) {
                if (errors < 10) {
                    int row = i / size;
                    int col = i % size;
                    System.out.printf("Mismatch at (%d,%d) [idx=%d]: ref=%d cand=%d diff=%d%n",
                            row, col, i, refVal, candVal, Math.abs(refVal - candVal));
                }
                errors++;
            }
        }
        if (errors > 0) {
            System.out.printf("Total mismatches: %d / %d pixels (%.2f%%)%n", errors, size * size, (100.0 * errors) / (size * size));
        }
        return errors == 0;
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
