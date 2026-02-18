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
 * PTX Validator for Matrix Multiplication 1D kernels.
 * Compares outputs from two PTX kernels to verify they produce identical results.
 *
 * Usage: MatrixMultiplication1DPTXValidator <kernel1.ptx> <kernel2.ptx> [size]
 *
 * IMPORTANT: Size must match the PTX function name suffix
 */
public class MatrixMultiplication1DPTXValidator {

    private static final float TOLERANCE = 1e-4f;
    private static final int LOCAL_WORK_SIZE_X = 16;
    private static final int LOCAL_WORK_SIZE_Y = 16;
    private static final String ENTRY_POINT = "matrixMultiplication";

    private static final Random RANDOM = new Random();

    private static void fillRandomData(FloatArray array, float min, float max, long seed) {
        RANDOM.setSeed(seed);
        float range = max - min;
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, min + RANDOM.nextFloat() * range);
        }
    }

    private static void runKernel(String kernelPath, FloatArray matrixA, FloatArray matrixB,
                                   FloatArray matrixC, int size, TornadoDevice device) throws TornadoExecutionPlanException {
        AccessorParameters accessors = new AccessorParameters(4);
        accessors.set(0, matrixA, Access.READ_ONLY);
        accessors.set(1, matrixB, Access.READ_ONLY);
        accessors.set(2, matrixC, Access.WRITE_ONLY);
        accessors.set(3, Integer.valueOf(size), Access.NONE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA, matrixB)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, matrixC);

        ImmutableTaskGraph snapshot = graph.snapshot();

        WorkerGrid2D worker = new WorkerGrid2D(size, size);
        worker.setLocalWork(LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler).execute();
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 2) {
            System.out.println("Usage: MatrixMultiplication1DPTXValidator <kernel1.ptx> <kernel2.ptx> [size]");
            System.out.println("  Default size: 512");
            System.out.println();
            System.out.println("Example:");
            System.out.println("  MatrixMultiplication1DPTXValidator kernels/ptx/matrix1d_generated.ptx kernels/ptx/matrix1d_custom.ptx 512");
            System.exit(1);
        }

        String kernel1Path = args[0];
        String kernel2Path = args[1];
        int size = (args.length >= 3) ? Integer.parseInt(args[2]) : 512;
        long seed = 42;

        System.out.println("=== Matrix Multiplication 1D PTX Validator ===");
        System.out.println("Kernel 1: " + kernel1Path);
        System.out.println("Kernel 2: " + kernel2Path);
        System.out.println("Matrix size: " + size + "x" + size);
        System.out.println("Tolerance: " + TOLERANCE);

        // Allocate arrays
        FloatArray matrixA = new FloatArray(size * size);
        FloatArray matrixB = new FloatArray(size * size);
        FloatArray matrixC1 = new FloatArray(size * size);
        FloatArray matrixC2 = new FloatArray(size * size);

        // Initialize with identical data
        fillRandomData(matrixA, -1.0f, 1.0f, seed);
        fillRandomData(matrixB, -1.0f, 1.0f, seed + 1);

        // Initialize output to zeros
        for (int i = 0; i < size * size; i++) {
            matrixC1.set(i, 0.0f);
            matrixC2.set(i, 0.0f);
        }

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // Run kernel 1
        System.out.println("\nRunning kernel 1...");
        try {
            runKernel(kernel1Path, matrixA, matrixB, matrixC1, size, device);
            System.out.println("Kernel 1 completed.");
        } catch (Exception e) {
            System.out.println("Kernel 1 FAILED: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }

        // Run kernel 2
        System.out.println("Running kernel 2...");
        try {
            runKernel(kernel2Path, matrixA, matrixB, matrixC2, size, device);
            System.out.println("Kernel 2 completed.");
        } catch (Exception e) {
            System.out.println("Kernel 2 FAILED: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }

        // Compare results
        System.out.println("\nComparing outputs...");
        int errors = 0;
        float maxDiff = 0;
        int maxDiffRow = 0, maxDiffCol = 0;

        for (int i = 0; i < size * size; i++) {
            float val1 = matrixC1.get(i);
            float val2 = matrixC2.get(i);
            float diff = Math.abs(val1 - val2);

            if (diff > maxDiff) {
                maxDiff = diff;
                maxDiffRow = i / size;
                maxDiffCol = i % size;
            }

            if (diff > TOLERANCE) {
                errors++;
                if (errors <= 5) {
                    int row = i / size;
                    int col = i % size;
                    System.out.printf("  Mismatch at [%d,%d]: %.6f vs %.6f (diff: %.6e)%n",
                            row, col, val1, val2, diff);
                }
            }
        }

        // Print summary
        System.out.println();
        System.out.println("=== Validation Results ===");
        System.out.printf("Max difference: %.6e at [%d,%d]%n", maxDiff, maxDiffRow, maxDiffCol);
        System.out.printf("Errors (diff > %.0e): %d / %d%n", TOLERANCE, errors, size * size);
        System.out.println();

        if (errors == 0) {
            System.out.println("VALIDATION PASSED - Kernels produce identical results");
        } else {
            System.out.println("VALIDATION FAILED - Kernels produce different results");
            if (errors > 5) {
                System.out.println("(Only first 5 errors shown)");
            }
        }

        // Print sample values
        System.out.println();
        System.out.println("Sample output values (corners of result matrix):");
        System.out.printf("  C[0,0]:       K1=%.6f  K2=%.6f%n", matrixC1.get(0), matrixC2.get(0));
        System.out.printf("  C[0,%d]:     K1=%.6f  K2=%.6f%n", size-1, matrixC1.get(size-1), matrixC2.get(size-1));
        System.out.printf("  C[%d,0]:     K1=%.6f  K2=%.6f%n", size-1, matrixC1.get((size-1)*size), matrixC2.get((size-1)*size));
        System.out.printf("  C[%d,%d]: K1=%.6f  K2=%.6f%n", size-1, size-1, matrixC1.get(size*size-1), matrixC2.get(size*size-1));
    }
}
