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
 * Validates NBody kernel output against sequential baseline.
 * Separate from benchmark to avoid affecting GPU warmup state.
 *
 * Usage: java ... NBodyValidator <kernel1.cl> [kernel2.cl] ... [--bodies=N]
 */
public class NBodyValidator {

    private static final int DEFAULT_NUM_BODIES = 1024; // Smaller for faster validation
    private static final int LOCAL_WORK_SIZE = 256;
    private static final String ENTRY_POINT = "nBody";
    private static final float TOLERANCE = 0.1f; // NBody accumulates FP errors
    private static final Random RANDOM = new Random(42);

    // NBody simulation parameters
    private static final float DEL_T = 0.005f;
    private static final float ESP_SQR = 500.0f;

    private static void initializeBodies(FloatArray pos, FloatArray vel, int numBodies) {
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

    private static void copyArray(FloatArray src, FloatArray dst) {
        for (int i = 0; i < src.getSize(); i++) {
            dst.set(i, src.get(i));
        }
    }

    // Sequential NBody for reference
    private static void nBodySequential(int numBodies, FloatArray pos, FloatArray vel, float delT, float espSqr) {
        for (int i = 0; i < numBodies; i++) {
            int body = 4 * i;

            float accX = 0.0f, accY = 0.0f, accZ = 0.0f;

            for (int j = 0; j < numBodies; j++) {
                int index = 4 * j;

                float rx = pos.get(index + 0) - pos.get(body + 0);
                float ry = pos.get(index + 1) - pos.get(body + 1);
                float rz = pos.get(index + 2) - pos.get(body + 2);

                float distSqr = rx * rx + ry * ry + rz * rz;
                float invDist = (float) (1.0f / Math.sqrt(distSqr + espSqr));
                float invDistCube = invDist * invDist * invDist;
                float s = pos.get(index + 3) * invDistCube;

                accX += s * rx;
                accY += s * ry;
                accZ += s * rz;
            }

            // Update position
            pos.set(body + 0, pos.get(body + 0) + vel.get(body + 0) * delT + 0.5f * accX * delT * delT);
            pos.set(body + 1, pos.get(body + 1) + vel.get(body + 1) * delT + 0.5f * accY * delT * delT);
            pos.set(body + 2, pos.get(body + 2) + vel.get(body + 2) * delT + 0.5f * accZ * delT * delT);

            // Update velocity
            vel.set(body + 0, vel.get(body + 0) + accX * delT);
            vel.set(body + 1, vel.get(body + 1) + accY * delT);
            vel.set(body + 2, vel.get(body + 2) + accZ * delT);
        }
    }

    private static ValidationResult validate(FloatArray refPos, FloatArray refVel,
                                             FloatArray candPos, FloatArray candVel, int numBodies) {
        int mismatches = 0;
        float maxDiff = 0;
        int maxDiffBody = -1;
        String maxDiffField = "";

        for (int i = 0; i < numBodies; i++) {
            for (int k = 0; k < 3; k++) { // x, y, z
                int idx = 4 * i + k;
                String field = (k == 0) ? "x" : (k == 1) ? "y" : "z";

                float posDiff = Math.abs(refPos.get(idx) - candPos.get(idx));
                float velDiff = Math.abs(refVel.get(idx) - candVel.get(idx));

                if (posDiff > maxDiff) {
                    maxDiff = posDiff;
                    maxDiffBody = i;
                    maxDiffField = "pos." + field;
                }
                if (velDiff > maxDiff) {
                    maxDiff = velDiff;
                    maxDiffBody = i;
                    maxDiffField = "vel." + field;
                }

                if (posDiff > TOLERANCE) mismatches++;
                if (velDiff > TOLERANCE) mismatches++;
            }
        }

        return new ValidationResult(mismatches == 0, mismatches, maxDiff, maxDiffBody, maxDiffField);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: NBodyValidator <kernel1.cl> [kernel2.cl] ... [--bodies=N]");
            System.exit(1);
        }

        // Parse arguments
        int numBodies = DEFAULT_NUM_BODIES;
        java.util.List<String> kernelPaths = new java.util.ArrayList<>();

        for (String arg : args) {
            if (arg.startsWith("--bodies=")) {
                numBodies = Integer.parseInt(arg.substring(9));
            } else {
                kernelPaths.add(arg);
            }
        }

        if (kernelPaths.isEmpty()) {
            System.out.println("Error: No kernel files specified");
            System.exit(1);
        }

        // Initialize reference arrays
        FloatArray initPos = new FloatArray(numBodies * 4);
        FloatArray initVel = new FloatArray(numBodies * 4);
        FloatArray refPos = new FloatArray(numBodies * 4);
        FloatArray refVel = new FloatArray(numBodies * 4);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

        // Initialize bodies
        initializeBodies(initPos, initVel, numBodies);

        // Compute sequential reference
        System.out.println("Computing sequential reference (" + numBodies + " bodies)...");
        copyArray(initPos, refPos);
        copyArray(initVel, refVel);
        long start = System.currentTimeMillis();
        nBodySequential(numBodies, refPos, refVel, DEL_T, ESP_SQR);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("Sequential computation took: " + elapsed + " ms");

        System.out.println();
        System.out.println("Validation Results (tolerance=" + TOLERANCE + ")");
        System.out.println("=".repeat(50));

        // Validate each kernel
        for (String kernelPath : kernelPaths) {
            FloatArray kernelPos = new FloatArray(numBodies * 4);
            FloatArray kernelVel = new FloatArray(numBodies * 4);

            // Reset to initial state
            copyArray(initPos, kernelPos);
            copyArray(initVel, kernelVel);

            // Set up and run kernel once
            AccessorParameters accessors = new AccessorParameters(5);
            accessors.set(0, Integer.valueOf(numBodies), Access.NONE);
            accessors.set(1, kernelPos, Access.READ_WRITE);
            accessors.set(2, kernelVel, Access.READ_WRITE);
            accessors.set(3, Float.valueOf(DEL_T), Access.NONE);
            accessors.set(4, Float.valueOf(ESP_SQR), Access.NONE);

            TaskGraph graph = new TaskGraph("validate")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, kernelPos, kernelVel)
                    .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, kernelPos, kernelVel);

            ImmutableTaskGraph snapshot = graph.snapshot();

            WorkerGrid1D worker = new WorkerGrid1D(numBodies);
            worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
            GridScheduler scheduler = new GridScheduler("validate.t0", worker);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.withDevice(device).withGridScheduler(scheduler);
                plan.execute();
            }

            // Validate
            ValidationResult result = validate(refPos, refVel, kernelPos, kernelVel, numBodies);

            System.out.printf("%s%n", kernelPath);
            if (result.valid) {
                System.out.printf("  PASSED (max diff: %.2e at body %d, field %s)%n",
                        result.maxDiff, result.maxDiffBody, result.maxDiffField);
            } else {
                System.out.printf("  FAILED: %d mismatches (max diff: %.2e at body %d, field %s)%n",
                        result.mismatches, result.maxDiff, result.maxDiffBody, result.maxDiffField);
            }
        }
    }

    private record ValidationResult(boolean valid, int mismatches, float maxDiff, int maxDiffBody, String maxDiffField) {}
}
