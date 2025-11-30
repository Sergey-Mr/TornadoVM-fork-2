package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;
import java.util.Random;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DFloat;

/**
 * Compares TornadoVM's generated matrix-vector kernel against a custom variant.
 * The benchmark reuses the data layout from {@code MatrixVector}: a row-major
 * {@link Matrix2DFloat} multiplied by a {@link VectorFloat}.
 */
public final class MatrixVectorExampleCustomBenchmark {

    private static final String DEFAULT_GENERATED_KERNEL = "kernels/matrixvector_generated.cl";
    private static final String DEFAULT_CUSTOM_KERNEL = "kernels/matrixvector_custom.cl";
    private static final String ENTRY_POINT = "computeMatrixVector";

    private static final int DEFAULT_SIZE = 2048;
    private static final int WARM_UP_ITERATIONS = 30;
    private static final int BENCHMARK_ITERATIONS = 60;
    private static final float VALIDATION_EPSILON = 1e-4f;

    private static final Random RANDOM = new Random(42);

    private MatrixVectorExampleCustomBenchmark() {
        // no instances
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int size = (args.length >= 1) ? Integer.parseInt(args[0]) : DEFAULT_SIZE;
        String generatedKernelPath = (args.length >= 2) ? args[1] : DEFAULT_GENERATED_KERNEL;
        String customKernelPath = (args.length >= 3) ? args[2] : DEFAULT_CUSTOM_KERNEL;

        System.out.println("Matrix size      : " + size + " x " + size);
        System.out.println("Generated kernel : " + generatedKernelPath);
        System.out.println("Custom kernel    : " + customKernelPath);

        Matrix2DFloat matrix = new Matrix2DFloat(size, size);
        VectorFloat vector = new VectorFloat(size);
        VectorFloat sequentialOutput = new VectorFloat(size);
        VectorFloat generatedOutput = new VectorFloat(size);
        VectorFloat customOutput = new VectorFloat(size);

        fillMatrix(matrix, -1.0f, 1.0f);
        fillVector(vector, -0.1f, 0.1f);

        warmUpSequential(matrix, vector, sequentialOutput);
        ArrayList<Long> sequentialSamples = benchmarkSequential(matrix, vector, sequentialOutput);

        BenchmarkResult generatedResult = benchmarkKernel("matrixVectorGenerated", generatedKernelPath, matrix, vector, generatedOutput);
        BenchmarkResult customResult = benchmarkKernel("matrixVectorCustom", customKernelPath, matrix, vector, customOutput);

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

        long totalFlops = 2L * size * size; // multiply + add per matrix element

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

    private static void fillVector(VectorFloat vector, float min, float max) {
        float range = max - min;
        for (int i = 0; i < vector.getLength(); i++) {
            vector.set(i, min + RANDOM.nextFloat() * range);
        }
    }

    private static void warmUpSequential(Matrix2DFloat matrix, VectorFloat vector, VectorFloat output) {
        for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
            computeSequential(matrix, vector, output);
        }
    }

    private static ArrayList<Long> benchmarkSequential(Matrix2DFloat matrix, VectorFloat vector, VectorFloat output) {
        ArrayList<Long> times = new ArrayList<>(BENCHMARK_ITERATIONS);
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            computeSequential(matrix, vector, output);
            long end = System.nanoTime();
            times.add(end - start);
        }
        return times;
    }

    private static BenchmarkResult benchmarkKernel(String graphName, String kernelPath, Matrix2DFloat matrix, VectorFloat vector, VectorFloat output)
            throws TornadoExecutionPlanException {

        AccessorParameters accessors = new AccessorParameters(3);
        accessors.set(0, matrix, Access.READ_ONLY);
        accessors.set(1, vector, Access.READ_ONLY);
        accessors.set(2, output, Access.WRITE_ONLY);

        TaskGraph graph = new TaskGraph(graphName)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrix, vector)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();

        ArrayList<Long> times = new ArrayList<>(BENCHMARK_ITERATIONS);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
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

    private static void computeSequential(Matrix2DFloat matrix, VectorFloat vector, VectorFloat result) {
        for (int row = 0; row < matrix.getNumRows(); row++) {
            float sum = 0.0f;
            for (int col = 0; col < matrix.getNumColumns(); col++) {
                sum += matrix.get(row, col) * vector.get(col);
            }
            result.set(row, sum);
        }
    }

    private static boolean validate(VectorFloat reference, VectorFloat candidate) {
        for (int i = 0; i < reference.getLength(); i++) {
            float diff = Math.abs(reference.get(i) - candidate.get(i));
            if (diff > VALIDATION_EPSILON) {
                System.out.printf("Mismatch at %d: ref=%.6f cand=%.6f diff=%.6f%n", i, reference.get(i), candidate.get(i), diff);
                return false;
            }
        }
        return true;
    }

    private record BenchmarkResult(LongSummaryStatistics timingStats) {
    }
}
