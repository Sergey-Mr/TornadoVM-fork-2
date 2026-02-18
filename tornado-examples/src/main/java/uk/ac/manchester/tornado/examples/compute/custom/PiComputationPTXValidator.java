package uk.ac.manchester.tornado.examples.compute.custom;

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
 * PTX Validator for Pi Computation kernels.
 * Compares outputs from two PTX kernels to verify they produce identical results.
 *
 * Usage: PiComputationPTXValidator <kernel1.ptx> <kernel2.ptx> [size]
 *
 * Default size: 1048576 (1M terms for faster validation)
 */
public class PiComputationPTXValidator {

    private static final float TOLERANCE = 1e-5f;
    private static final int DEFAULT_SIZE = 1048576;  // 1M terms
    private static final int LOCAL_WORK_SIZE = 256;
    private static final String ENTRY_POINT = "computePi";

    private static float runKernel(String kernelPath, FloatArray input, int size,
                                    TornadoDevice device) throws TornadoExecutionPlanException {
        FloatArray result = new FloatArray(1);
        result.init(0.0f);

        AccessorParameters accessors = new AccessorParameters(2);
        accessors.set(0, input, Access.READ_ONLY);
        accessors.set(1, result, Access.READ_WRITE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, result)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, result);

        ImmutableTaskGraph snapshot = graph.snapshot();

        WorkerGrid1D worker = new WorkerGrid1D(size);
        worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler).execute();
        }

        return result.get(0) * 4.0f;
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 2) {
            System.out.println("Usage: PiComputationPTXValidator <kernel1.ptx> <kernel2.ptx> [size]");
            System.out.println("  Default size: " + DEFAULT_SIZE);
            System.out.println();
            System.out.println("Example:");
            System.out.println("  PiComputationPTXValidator kernels/ptx/pi_generated.ptx kernels/ptx/pi_custom.ptx 1048576");
            System.exit(1);
        }

        String kernel1Path = args[0];
        String kernel2Path = args[1];
        int size = (args.length >= 3) ? Integer.parseInt(args[2]) : DEFAULT_SIZE;

        System.out.println("=== Pi Computation PTX Validator ===");
        System.out.println("Kernel 1: " + kernel1Path);
        System.out.println("Kernel 2: " + kernel2Path);
        System.out.println("Number of terms: " + size);
        System.out.println("Tolerance: " + TOLERANCE);

        // Allocate input array (zeros - values computed in kernel)
        FloatArray input = new FloatArray(size);
        input.init(0.0f);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        float pi1 = 0, pi2 = 0;

        // Run kernel 1
        System.out.println("\nRunning kernel 1...");
        try {
            pi1 = runKernel(kernel1Path, input, size, device);
            System.out.println("Kernel 1 completed.");
            System.out.printf("  Pi value: %.10f%n", pi1);
        } catch (Exception e) {
            System.out.println("Kernel 1 FAILED: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }

        // Run kernel 2
        System.out.println("Running kernel 2...");
        try {
            pi2 = runKernel(kernel2Path, input, size, device);
            System.out.println("Kernel 2 completed.");
            System.out.printf("  Pi value: %.10f%n", pi2);
        } catch (Exception e) {
            System.out.println("Kernel 2 FAILED: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }

        // Compare results
        float diff = Math.abs(pi1 - pi2);
        float error1 = (float) Math.abs(Math.PI - pi1);
        float error2 = (float) Math.abs(Math.PI - pi2);

        // Print summary
        System.out.println();
        System.out.println("=== Validation Results ===");
        System.out.printf("Kernel 1 Pi: %.10f (error from true Pi: %.6e)%n", pi1, error1);
        System.out.printf("Kernel 2 Pi: %.10f (error from true Pi: %.6e)%n", pi2, error2);
        System.out.printf("Difference between kernels: %.6e%n", diff);
        System.out.printf("True Pi:     %.10f%n", Math.PI);
        System.out.println();

        if (diff <= TOLERANCE) {
            System.out.println("VALIDATION PASSED - Kernels produce identical Pi values");
        } else {
            System.out.println("VALIDATION FAILED - Kernels produce different Pi values");
        }
    }
}
