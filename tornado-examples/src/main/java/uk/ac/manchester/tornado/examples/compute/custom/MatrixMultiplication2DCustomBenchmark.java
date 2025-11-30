package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;
import java.util.Random;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DFloat;

/**
 * Benchmarks TornadoVM's generated MatrixMultiplication2D kernel against a
 * user-provided OpenCL implementation using {@code prebuiltTask}.
 */
public final class MatrixMultiplication2DCustomBenchmark {

    private static final String DEFAULT_GENERATED_KERNEL = "kernels/matrixMultiplication2d_generated.cl";
    private static final String DEFAULT_CUSTOM_KERNEL = "kernels/matrixMultiplication2d_custom.cl";
    private static final String ENTRY_POINT = "matrixMultiplication";

    private static final int DEFAULT_SIZE = 512;
    private static final int WARM_UP_ITERATIONS = 30;
    private static final int BENCHMARK_ITERATIONS = 60;
    private static final float VALIDATION_EPSILON = 1e-3f;

    private static final Random RANDOM = new Random(42);

    private MatrixMultiplication2DCustomBenchmark() {
        // utility class
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int size = (args.length >= 1) ? Integer.parseInt(args[0]) : DEFAULT_SIZE;
        String generatedKernelPath = (args.length >= 2) ? args[1] : DEFAULT_GENERATED_KERNEL;
        String customKernelPath = (args.length >= 3) ? args[2] : DEFAULT_CUSTOM_KERNEL;

        System.out.println("Matrix size      : " + size + " x " + size);
        System.out.println("Generated kernel : " + generatedKernelPath);
        System.out.println("Custom kernel    : " + customKernelPath);

        Matrix2DFloat matrixA = new Matrix2DFloat(size, size);
        Matrix2DFloat matrixB = new Matrix2DFloat(size, size);
        Matrix2DFloat sequentialOutput = new Matrix2DFloat(size, size);
        Matrix2DFloat generatedOutput = new Matrix2DFloat(size, size);
        Matrix2DFloat customOutput = new Matrix2DFloat(size, size);

        fillMatrix(matrixA, -1.0f, 1.0f);
        fillMatrix(matrixB, -1.0f, 1.0f);

        warmUpSequential(matrixA, matrixB, sequentialOutput, size);
        ArrayList<Long> sequentialSamples = benchmarkSequential(matrixA, matrixB, sequentialOutput, size);

        BenchmarkResult generatedResult = benchmarkKernel("mxmGenerated", generatedKernelPath, matrixA, matrixB, generatedOutput, size);
        BenchmarkResult customResult = benchmarkKernel("mxmCustom", customKernelPath, matrixA, matrixB, customOutput, size);

        boolean generatedValid = validate(sequentialOutput, generatedOutput);
        boolean customValid = validate(sequentialOutput, customOutput);

        if (!generatedValid) {
            System.out.println("[WARN] Generated kernel result does not match the sequential baseline.");
        }
        if (!customValid) {
            System.out.println("[WARN] Custom kernel result does not match the sequential baseline.");
        }

        LongSummaryStatistics sequentialStats = sequentialSamples.stream().mapToLong(Long::longValue).summaryStatistics();
        LongSummaryStatistics generatedStats = generatedResult.timingStats();
        LongSummaryStatistics customStats = customResult.timingStats();

        long totalFlops = 2L * size * size * size; // multiply + add per inner iteration

        double sequentialMs = sequentialStats.getAverage() / 1_000_000.0;
        double generatedMs = generatedStats.getAverage() / 1_000_000.0;
        double customMs = customStats.getAverage() / 1_000_000.0;

        double sequentialGFlops = (totalFlops * 1e-9) / (sequentialStats.getAverage() * 1e-9);
        double generatedGFlops = (totalFlops * 1e-9) / (generatedStats.getAverage() * 1e-9);
        double customGFlops = (totalFlops * 1e-9) / (customStats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Benchmark Results");
        System.out.println("=================");
        System.out.printf("Sequential: avg=%.3f ms min=%.3f ms max=%.3f ms GFLOP/s=%.2f%n",
                sequentialMs, sequentialStats.getMin() / 1_000_000.0, sequentialStats.getMax() / 1_000_000.0, sequentialGFlops);
        System.out.printf("Generated : avg=%.3f ms min=%.3f ms max=%.3f ms GFLOP/s=%.2f%n",
                generatedMs, generatedStats.getMin() / 1_000_000.0, generatedStats.getMax() / 1_000_000.0, generatedGFlops);
        System.out.printf("Custom    : avg=%.3f ms min=%.3f ms max=%.3f ms GFLOP/s=%.2f%n",
                customMs, customStats.getMin() / 1_000_000.0, customStats.getMax() / 1_000_000.0, customGFlops);

        System.out.println();
        System.out.printf("Speedup (Generated vs Sequential): %.2fx%n", sequentialStats.getAverage() / generatedStats.getAverage());
        System.out.printf("Speedup (Custom vs Sequential)   : %.2fx%n", sequentialStats.getAverage() / customStats.getAverage());
        System.out.printf("Speedup (Custom vs Generated)    : %.2fx%n", generatedStats.getAverage() / customStats.getAverage());

        if (customStats.getAverage() < generatedStats.getAverage()) {
            System.out.printf("Custom kernel is faster by %.2fx%n", generatedStats.getAverage() / customStats.getAverage());
        } else {
            System.out.printf("Generated kernel is faster by %.2fx%n", customStats.getAverage() / generatedStats.getAverage());
        }
    }

    private static void fillMatrix(Matrix2DFloat matrix, float min, float max) {
        float range = max - min;
        for (int row = 0; row < matrix.getNumRows(); row++) {
            for (int col = 0; col < matrix.getNumColumns(); col++) {
                matrix.set(row, col, min + RANDOM.nextFloat() * range);
            }
        }
    }

    private static void warmUpSequential(Matrix2DFloat a, Matrix2DFloat b, Matrix2DFloat out, int size) {
        for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
            computeSequential(a, b, out, size);
        }
    }

    private static ArrayList<Long> benchmarkSequential(Matrix2DFloat a, Matrix2DFloat b, Matrix2DFloat out, int size) {
        ArrayList<Long> times = new ArrayList<>(BENCHMARK_ITERATIONS);
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            computeSequential(a, b, out, size);
            long end = System.nanoTime();
            times.add(end - start);
        }
        return times;
    }

    private static BenchmarkResult benchmarkKernel(String graphName, String kernelPath, Matrix2DFloat a, Matrix2DFloat b, Matrix2DFloat out, int size)
            throws TornadoExecutionPlanException {
        AccessorParameters accessors = new AccessorParameters(4);
        accessors.set(0, a, Access.READ_ONLY);
        accessors.set(1, b, Access.READ_ONLY);
        accessors.set(2, out, Access.WRITE_ONLY);
        accessors.set(3, Integer.valueOf(size), Access.NONE);

        TaskGraph graph = new TaskGraph(graphName)
                // Transfer output once so the device sees a fully initialised descriptor
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, out)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        ImmutableTaskGraph snapshot = graph.snapshot();

        WorkerGrid2D workerGrid = new WorkerGrid2D(size, size);
        workerGrid.setLocalWork(16, 16, 1);
        GridScheduler scheduler = new GridScheduler(graphName + ".t0", workerGrid);

        ArrayList<Long> times = new ArrayList<>(BENCHMARK_ITERATIONS);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withGridScheduler(scheduler);
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                plan.execute();
            }
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                long start = System.nanoTime();
                plan.execute();
                long end = System.nanoTime();
                times.add(end - start);
            }
        }

        return new BenchmarkResult(times.stream().mapToLong(Long::longValue).summaryStatistics());
    }

    private static void computeSequential(Matrix2DFloat a, Matrix2DFloat b, Matrix2DFloat out, int size) {
        for (int row = 0; row < size; row++) {
            for (int col = 0; col < size; col++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    sum += a.get(row, k) * b.get(k, col);
                }
                out.set(row, col, sum);
            }
        }
    }

    private static boolean validate(Matrix2DFloat reference, Matrix2DFloat candidate) {
        for (int row = 0; row < reference.getNumRows(); row++) {
            for (int col = 0; col < reference.getNumColumns(); col++) {
                float diff = Math.abs(reference.get(row, col) - candidate.get(row, col));
                if (diff > VALIDATION_EPSILON) {
                    System.out.printf("Mismatch at (%d,%d): ref=%.6f cand=%.6f diff=%.6f%n", row, col, reference.get(row, col), candidate.get(row, col), diff);
                    return false;
                }
            }
        }
        return true;
    }

    private record BenchmarkResult(LongSummaryStatistics timingStats) {
    }
}
