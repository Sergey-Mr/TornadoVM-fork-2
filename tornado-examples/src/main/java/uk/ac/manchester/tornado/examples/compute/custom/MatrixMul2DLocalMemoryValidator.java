package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.Random;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/**
 * Validates matrix multiplication kernel output against sequential baseline.
 * Separate from benchmark to avoid affecting GPU warmup state.
 *
 * Usage: java ... MatrixMul2DLocalMemoryValidator <kernel1.cl> [kernel2.cl] ... [--size=N]
 */
public class MatrixMul2DLocalMemoryValidator {

    private static final int DEFAULT_SIZE = 512; // Smaller for faster validation
    private static final int TS = 16; // Tile size - must match kernel (custom uses 16)
    private static final String ENTRY_POINT = "matrixMultiplication";
    private static final float TOLERANCE = 0.01f; // Larger tolerance for accumulated FP errors in matmul
    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array, float min, float max) {
        float range = max - min;
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, min + RANDOM.nextFloat() * range);
        }
    }

    // Sequential matrix multiplication for reference
    private static void matrixMultiplicationSequential(FloatArray A, FloatArray B, FloatArray C, int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    sum += A.get(i * size + k) * B.get(k * size + j);
                }
                C.set(i * size + j, sum);
            }
        }
    }

    private static ValidationResult validate(FloatArray reference, FloatArray candidate, int size) {
        int mismatches = 0;
        float maxDiff = 0;
        int maxDiffIndex = -1;

        for (int i = 0; i < reference.getSize(); i++) {
            float diff = Math.abs(reference.get(i) - candidate.get(i));
            if (diff > maxDiff) {
                maxDiff = diff;
                maxDiffIndex = i;
            }
            if (diff > TOLERANCE) {
                mismatches++;
            }
        }

        // Convert flat index to row, col
        int maxDiffRow = maxDiffIndex / size;
        int maxDiffCol = maxDiffIndex % size;

        return new ValidationResult(mismatches == 0, mismatches, maxDiff, maxDiffIndex, maxDiffRow, maxDiffCol);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: MatrixMul2DLocalMemoryValidator <kernel1.cl> [kernel2.cl] ... [--size=N]");
            System.exit(1);
        }

        // Parse size argument if present
        int size = DEFAULT_SIZE;
        java.util.List<String> kernelPaths = new java.util.ArrayList<>();

        for (String arg : args) {
            if (arg.startsWith("--size=")) {
                size = Integer.parseInt(arg.substring(7));
            } else {
                kernelPaths.add(arg);
            }
        }

        if (size % TS != 0) {
            System.err.println("Error: size (" + size + ") must be divisible by tile size (" + TS + ")");
            System.exit(1);
        }

        if (kernelPaths.isEmpty()) {
            System.out.println("Error: No kernel files specified");
            System.exit(1);
        }

        FloatArray matrixA = new FloatArray(size * size);
        FloatArray matrixB = new FloatArray(size * size);
        FloatArray referenceC = new FloatArray(size * size);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

        // Use same random seed as benchmark
        fillRandomData(matrixA, -1.0f, 1.0f);
        fillRandomData(matrixB, -1.0f, 1.0f);

        // Compute sequential reference
        System.out.println("Computing sequential reference (" + size + "x" + size + ")...");
        long start = System.currentTimeMillis();
        matrixMultiplicationSequential(matrixA, matrixB, referenceC, size);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("Sequential computation took: " + elapsed + " ms");

        System.out.println();
        System.out.println("Validation Results (tolerance=" + TOLERANCE + ")");
        System.out.println("=".repeat(50));

        // Validate each kernel
        for (String kernelPath : kernelPaths) {
            FloatArray kernelC = new FloatArray(size * size);

            // Set up and run kernel once
            AccessorParameters accessors = new AccessorParameters(4);
            accessors.set(0, matrixA, Access.READ_ONLY);
            accessors.set(1, matrixB, Access.READ_ONLY);
            accessors.set(2, kernelC, Access.WRITE_ONLY);
            accessors.set(3, Integer.valueOf(size), Access.NONE);

            TaskGraph graph = new TaskGraph("validate")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA, matrixB)
                    .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, kernelC);

            ImmutableTaskGraph snapshot = graph.snapshot();

            WorkerGrid2D worker = new WorkerGrid2D(size, size);
            worker.setLocalWork(TS, TS, 1);
            GridScheduler scheduler = new GridScheduler("validate.t0", worker);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.withDevice(device).withGridScheduler(scheduler);
                plan.execute();
            }

            // Validate
            ValidationResult result = validate(referenceC, kernelC, size);

            System.out.printf("%s%n", kernelPath);
            if (result.valid) {
                System.out.printf("  PASSED (max diff: %.2e at [%d,%d])%n",
                        result.maxDiff, result.maxDiffRow, result.maxDiffCol);
            } else {
                System.out.printf("  FAILED: %d mismatches (max diff: %.2e at [%d,%d])%n",
                        result.mismatches, result.maxDiff, result.maxDiffRow, result.maxDiffCol);
                // Show expected vs actual for max diff location
                System.out.printf("    Expected: %.6f, Got: %.6f%n",
                        referenceC.get(result.maxDiffIndex), kernelC.get(result.maxDiffIndex));
            }
        }
    }

    private record ValidationResult(boolean valid, int mismatches, float maxDiff, int maxDiffIndex, int maxDiffRow, int maxDiffCol) {}
}
