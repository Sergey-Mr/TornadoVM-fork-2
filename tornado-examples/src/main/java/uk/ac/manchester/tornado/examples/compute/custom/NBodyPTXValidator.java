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
 * PTX Validator for NBody kernels.
 * Compares outputs from two PTX kernels to verify they produce identical results.
 *
 * Usage: NBodyPTXValidator <kernel1.ptx> <kernel2.ptx> [numBodies]
 *
 * IMPORTANT - PTX File Preparation:
 * PTX files must have headers stripped. See NBodyPTXBenchmark.java for details.
 */
public class NBodyPTXValidator {

    private static final float DEL_T = 0.005f;
    private static final float ESP_SQR = 500.0f;
    private static final float TOLERANCE = 1e-2f;  // Allow small floating point differences
    private static final int LOCAL_WORK_SIZE = 256;
    private static final String ENTRY_POINT = "nBody";

    private static final Random RANDOM = new Random();

    private static void initializeBodies(FloatArray pos, FloatArray vel, int numBodies, long seed) {
        RANDOM.setSeed(seed);
        // Position: x, y, z, mass (4 floats per body)
        for (int i = 0; i < numBodies; i++) {
            pos.set(4 * i + 0, RANDOM.nextFloat() * 2.0f - 1.0f);     // x
            pos.set(4 * i + 1, RANDOM.nextFloat() * 2.0f - 1.0f);     // y
            pos.set(4 * i + 2, RANDOM.nextFloat() * 2.0f - 1.0f);     // z
            pos.set(4 * i + 3, RANDOM.nextFloat() * 0.5f + 0.5f);     // mass
        }
        // Velocity: vx, vy, vz, padding (4 floats per body)
        for (int i = 0; i < numBodies * 4; i++) {
            vel.set(i, 0.0f);
        }
    }

    private static void runKernel(String kernelPath, FloatArray pos, FloatArray vel,
                                   int numBodies, TornadoDevice device) throws TornadoExecutionPlanException {
        AccessorParameters accessors = new AccessorParameters(5);
        accessors.set(0, Integer.valueOf(numBodies), Access.NONE);
        accessors.set(1, pos, Access.READ_WRITE);
        accessors.set(2, vel, Access.READ_WRITE);
        accessors.set(3, Float.valueOf(DEL_T), Access.NONE);
        accessors.set(4, Float.valueOf(ESP_SQR), Access.NONE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, pos, vel)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, pos, vel);

        ImmutableTaskGraph snapshot = graph.snapshot();

        WorkerGrid1D worker = new WorkerGrid1D(numBodies);
        worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler).execute();
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 2) {
            System.out.println("Usage: NBodyPTXValidator <kernel1.ptx> <kernel2.ptx> [numBodies]");
            System.out.println("  Default numBodies: 2048");
            System.out.println();
            System.out.println("Example:");
            System.out.println("  NBodyPTXValidator kernels/ptx/nbody_generated.ptx kernels/ptx/nbody_custom.ptx 2048");
            System.exit(1);
        }

        String kernel1Path = args[0];
        String kernel2Path = args[1];
        int numBodies = (args.length >= 3) ? Integer.parseInt(args[2]) : 2048;
        long seed = 42;  // Fixed seed for reproducibility

        System.out.println("=== NBody PTX Validator ===");
        System.out.println("Kernel 1: " + kernel1Path);
        System.out.println("Kernel 2: " + kernel2Path);
        System.out.println("Number of bodies: " + numBodies);
        System.out.println("Tolerance: " + TOLERANCE);

        // Allocate arrays for both kernels
        FloatArray pos1 = new FloatArray(numBodies * 4);
        FloatArray vel1 = new FloatArray(numBodies * 4);
        FloatArray pos2 = new FloatArray(numBodies * 4);
        FloatArray vel2 = new FloatArray(numBodies * 4);

        // Initialize with identical data
        initializeBodies(pos1, vel1, numBodies, seed);
        initializeBodies(pos2, vel2, numBodies, seed);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // Run kernel 1
        System.out.println("\nRunning kernel 1...");
        try {
            runKernel(kernel1Path, pos1, vel1, numBodies, device);
            System.out.println("Kernel 1 completed.");
        } catch (Exception e) {
            System.out.println("Kernel 1 FAILED: " + e.getMessage());
            System.exit(1);
        }

        // Run kernel 2
        System.out.println("Running kernel 2...");
        try {
            runKernel(kernel2Path, pos2, vel2, numBodies, device);
            System.out.println("Kernel 2 completed.");
        } catch (Exception e) {
            System.out.println("Kernel 2 FAILED: " + e.getMessage());
            System.exit(1);
        }

        // Compare position results
        System.out.println("\nComparing position outputs...");
        int posErrors = 0;
        float maxPosDiff = 0;
        int maxPosDiffIndex = 0;

        for (int i = 0; i < numBodies * 4; i++) {
            float val1 = pos1.get(i);
            float val2 = pos2.get(i);
            float diff = Math.abs(val1 - val2);

            if (diff > maxPosDiff) {
                maxPosDiff = diff;
                maxPosDiffIndex = i;
            }

            if (diff > TOLERANCE) {
                posErrors++;
                if (posErrors <= 5) {
                    int bodyIdx = i / 4;
                    int component = i % 4;
                    String componentName = new String[]{"x", "y", "z", "mass"}[component];
                    System.out.printf("  Position mismatch at body %d (%s): %.6f vs %.6f (diff: %.6e)%n",
                            bodyIdx, componentName, val1, val2, diff);
                }
            }
        }

        // Compare velocity results
        System.out.println("Comparing velocity outputs...");
        int velErrors = 0;
        float maxVelDiff = 0;

        for (int i = 0; i < numBodies * 4; i++) {
            float val1 = vel1.get(i);
            float val2 = vel2.get(i);
            float diff = Math.abs(val1 - val2);

            if (diff > maxVelDiff) {
                maxVelDiff = diff;
            }

            if (diff > TOLERANCE) {
                velErrors++;
                if (velErrors <= 5) {
                    int bodyIdx = i / 4;
                    int component = i % 4;
                    String componentName = new String[]{"vx", "vy", "vz", "pad"}[component];
                    System.out.printf("  Velocity mismatch at body %d (%s): %.6f vs %.6f (diff: %.6e)%n",
                            bodyIdx, componentName, val1, val2, diff);
                }
            }
        }

        // Print summary
        System.out.println();
        System.out.println("=== Validation Results ===");
        System.out.printf("Position - Max difference: %.6e at index %d%n", maxPosDiff, maxPosDiffIndex);
        System.out.printf("Position - Errors (diff > %.0e): %d / %d%n", TOLERANCE, posErrors, numBodies * 4);
        System.out.printf("Velocity - Max difference: %.6e%n", maxVelDiff);
        System.out.printf("Velocity - Errors (diff > %.0e): %d / %d%n", TOLERANCE, velErrors, numBodies * 4);
        System.out.println();

        if (posErrors == 0 && velErrors == 0) {
            System.out.println("VALIDATION PASSED - Kernels produce identical results");
        } else {
            System.out.println("VALIDATION FAILED - Kernels produce different results");
            if (posErrors > 5 || velErrors > 5) {
                System.out.println("(Only first 5 errors shown for each category)");
            }
        }

        // Print sample values for verification
        System.out.println();
        System.out.println("Sample output values (first 3 bodies):");
        for (int i = 0; i < Math.min(3, numBodies); i++) {
            System.out.printf("  Body %d - K1 pos: (%.4f, %.4f, %.4f) K2 pos: (%.4f, %.4f, %.4f)%n",
                    i,
                    pos1.get(4*i), pos1.get(4*i+1), pos1.get(4*i+2),
                    pos2.get(4*i), pos2.get(4*i+1), pos2.get(4*i+2));
        }
    }
}
