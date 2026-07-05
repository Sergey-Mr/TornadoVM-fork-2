package uk.ac.manchester.tornado.examples.compute.custom;

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

/**
 * PTX Validator for Matrix-Vector Row Major kernels.
 * Compares outputs from two PTX kernels to verify they produce identical results.
 *
 * Usage: MatrixVectorRowMajorPTXValidator <kernel1.ptx> <kernel2.ptx> [inputDim] [outputDim]
 *
 * Default: inputDim=8192, outputDim=2048
 */
public class MatrixVectorRowMajorPTXValidator {

    private static final float TOLERANCE = 1e-4f;
    private static final int LOCAL_WORK_GROUP_SIZE = 256;
    private static final String ENTRY_POINT = "matrixVectorGeneric";

    private static final Random RANDOM = new Random();

    private static void fillRandomData(FloatArray array, float min, float max, long seed) {
        RANDOM.setSeed(seed);
        float range = max - min;
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, min + RANDOM.nextFloat() * range);
        }
    }

    private static void runKernel(String kernelPath, FloatArray input, FloatArray output,
                                   FloatArray weights, int inputDim, int outputDim,
                                   TornadoDevice device) throws TornadoExecutionPlanException {
        AccessorParameters accessors = new AccessorParameters(6);
        accessors.set(0, input, Access.READ_ONLY);
        accessors.set(1, output, Access.WRITE_ONLY);
        accessors.set(2, weights, Access.READ_ONLY);
        accessors.set(3, Integer.valueOf(inputDim), Access.NONE);
        accessors.set(4, Integer.valueOf(outputDim), Access.NONE);
        accessors.set(5, Integer.valueOf(LOCAL_WORK_GROUP_SIZE), Access.NONE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input, weights)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();

        WorkerGrid1D worker = new WorkerGrid1D(outputDim * LOCAL_WORK_GROUP_SIZE);
        worker.setLocalWork(LOCAL_WORK_GROUP_SIZE, 1, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler).execute();
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 2) {
            System.out.println("Usage: MatrixVectorRowMajorPTXValidator <kernel1.ptx> <kernel2.ptx> [inputDim] [outputDim]");
            System.out.println("  Default inputDim: 8192");
            System.out.println("  Default outputDim: 2048");
            System.out.println();
            System.out.println("Example:");
            System.out.println("  MatrixVectorRowMajorPTXValidator kernels/ptx/matvec_generated.ptx kernels/ptx/matvec_custom.ptx 8192 2048");
            System.exit(1);
        }

        String kernel1Path = args[0];
        String kernel2Path = args[1];
        int inputDim = (args.length >= 3) ? Integer.parseInt(args[2]) : 8192;
        int outputDim = (args.length >= 4) ? Integer.parseInt(args[3]) : 2048;
        long seed = 42;

        System.out.println("=== Matrix-Vector Row Major PTX Validator ===");
        System.out.println("Kernel 1: " + kernel1Path);
        System.out.println("Kernel 2: " + kernel2Path);
        System.out.println("Input dimension: " + inputDim);
        System.out.println("Output dimension: " + outputDim);
        System.out.println("Tolerance: " + TOLERANCE);

        // Allocate arrays
        FloatArray input = new FloatArray(inputDim);
        FloatArray weights = new FloatArray(inputDim * outputDim);
        FloatArray output1 = new FloatArray(outputDim);
        FloatArray output2 = new FloatArray(outputDim);

        // Initialize with identical data
        fillRandomData(input, -1.0f, 1.0f, seed);
        fillRandomData(weights, -0.1f, 0.1f, seed + 1);

        // Initialize output to zeros
        for (int i = 0; i < outputDim; i++) {
            output1.set(i, 0.0f);
            output2.set(i, 0.0f);
        }

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // Run kernel 1
        System.out.println("\nRunning kernel 1...");
        try {
            runKernel(kernel1Path, input, output1, weights, inputDim, outputDim, device);
            System.out.println("Kernel 1 completed.");
        } catch (Exception e) {
            System.out.println("Kernel 1 FAILED: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }

        // Run kernel 2
        System.out.println("Running kernel 2...");
        try {
            runKernel(kernel2Path, input, output2, weights, inputDim, outputDim, device);
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
        int maxDiffIndex = 0;

        for (int i = 0; i < outputDim; i++) {
            float val1 = output1.get(i);
            float val2 = output2.get(i);
            float diff = Math.abs(val1 - val2);

            if (diff > maxDiff) {
                maxDiff = diff;
                maxDiffIndex = i;
            }

            if (diff > TOLERANCE) {
                errors++;
                if (errors <= 5) {
                    System.out.printf("  Mismatch at [%d]: %.6f vs %.6f (diff: %.6e)%n",
                            i, val1, val2, diff);
                }
            }
        }

        // Print summary
        System.out.println();
        System.out.println("=== Validation Results ===");
        System.out.printf("Max difference: %.6e at index %d%n", maxDiff, maxDiffIndex);
        System.out.printf("Errors (diff > %.0e): %d / %d%n", TOLERANCE, errors, outputDim);
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
        System.out.println("Sample output values:");
        System.out.printf("  output[0]:    K1=%.6f  K2=%.6f%n", output1.get(0), output2.get(0));
        System.out.printf("  output[%d]: K1=%.6f  K2=%.6f%n", outputDim/2, output1.get(outputDim/2), output2.get(outputDim/2));
        System.out.printf("  output[%d]: K1=%.6f  K2=%.6f%n", outputDim-1, output1.get(outputDim-1), output2.get(outputDim-1));
    }
}
