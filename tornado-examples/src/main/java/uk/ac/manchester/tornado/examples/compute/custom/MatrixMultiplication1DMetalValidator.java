package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.Random;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoRuntime;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Validates Matrix Multiplication 1D MSL kernel output against a sequential CPU
 * reference on the METAL backend. Separate from the benchmark to avoid affecting
 * GPU warmup state, matching the OpenCL/PTX validator methodology.
 *
 * Usage: java ... MatrixMultiplication1DMetalValidator <kernel1.metal> [kernel2.metal] ... [--size=N] [--entry=NAME]
 */
public class MatrixMultiplication1DMetalValidator {

    private static final int DEFAULT_SIZE = 512;
    private static final int LOCAL_WORK_SIZE = 16;
    private static final String DEFAULT_ENTRY = "uk_ac_manchester_tornado_examples_compute_MatrixMultiplication1D_matrixMultiplication";
    private static final float TOLERANCE = 1e-3f;
    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array, float min, float max) {
        float range = max - min;
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, min + RANDOM.nextFloat() * range);
        }
    }

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

    private static TornadoDevice getMetalDevice() {
        TornadoRuntime runtime = TornadoRuntimeProvider.getTornadoRuntime();
        for (int i = 0; i < runtime.getNumBackends(); i++) {
            if (runtime.getBackendType(i) == TornadoVMBackendType.METAL) {
                return runtime.getBackend(i).getDevice(0);
            }
        }
        throw new RuntimeException("No Metal backend available. Rebuild TornadoVM with --backend metal.");
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
        int maxDiffRow = maxDiffIndex / size;
        int maxDiffCol = maxDiffIndex % size;
        return new ValidationResult(mismatches == 0, mismatches, maxDiff, maxDiffIndex, maxDiffRow, maxDiffCol);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: MatrixMultiplication1DMetalValidator <kernel1.metal> ... [--size=N] [--entry=NAME]");
            System.exit(1);
        }

        int size = DEFAULT_SIZE;
        String entryPoint = DEFAULT_ENTRY;
        java.util.List<String> kernelPaths = new java.util.ArrayList<>();
        for (String arg : args) {
            if (arg.startsWith("--size=")) {
                size = Integer.parseInt(arg.substring(7));
            } else if (arg.startsWith("--entry=")) {
                entryPoint = arg.substring(8);
            } else {
                kernelPaths.add(arg);
            }
        }

        if (kernelPaths.isEmpty()) {
            System.out.println("Error: No kernel files specified");
            System.exit(1);
        }

        FloatArray matrixA = new FloatArray(size * size);
        FloatArray matrixB = new FloatArray(size * size);
        FloatArray referenceC = new FloatArray(size * size);

        TornadoDevice device = getMetalDevice();

        fillRandomData(matrixA, -1.0f, 1.0f);
        fillRandomData(matrixB, -1.0f, 1.0f);

        System.out.println("Backend: Metal, device: " + device);
        System.out.println("Entry point: " + entryPoint);
        System.out.println("Computing sequential reference (" + size + "x" + size + ")...");
        long start = System.currentTimeMillis();
        matrixMultiplicationSequential(matrixA, matrixB, referenceC, size);
        System.out.println("Sequential computation took: " + (System.currentTimeMillis() - start) + " ms");

        System.out.println();
        System.out.println("Validation Results (tolerance=" + TOLERANCE + ")");
        System.out.println("=".repeat(50));

        for (String kernelPath : kernelPaths) {
            FloatArray kernelC = new FloatArray(size * size);

            AccessorParameters accessors = new AccessorParameters(4);
            accessors.set(0, matrixA, Access.READ_ONLY);
            accessors.set(1, matrixB, Access.READ_ONLY);
            accessors.set(2, kernelC, Access.WRITE_ONLY);
            accessors.set(3, Integer.valueOf(size), Access.NONE);

            TaskGraph graph = new TaskGraph("validate")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA, matrixB)
                    .prebuiltTask("t0", entryPoint, kernelPath, accessors)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, kernelC);

            ImmutableTaskGraph snapshot = graph.snapshot();

            WorkerGrid2D worker = new WorkerGrid2D(size, size);
            worker.setLocalWork(LOCAL_WORK_SIZE, LOCAL_WORK_SIZE, 1);
            GridScheduler scheduler = new GridScheduler("validate.t0", worker);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.withDevice(device).withGridScheduler(scheduler);
                plan.execute();
            }

            ValidationResult result = validate(referenceC, kernelC, size);
            System.out.printf("%s%n", kernelPath);
            if (result.valid) {
                System.out.printf("  PASSED (max diff: %.2e at [%d,%d])%n",
                        result.maxDiff, result.maxDiffRow, result.maxDiffCol);
            } else {
                System.out.printf("  FAILED: %d mismatches (max diff: %.2e at [%d,%d])%n",
                        result.mismatches, result.maxDiff, result.maxDiffRow, result.maxDiffCol);
                System.out.printf("    Expected: %.6f, Got: %.6f%n",
                        referenceC.get(result.maxDiffIndex), kernelC.get(result.maxDiffIndex));
            }
        }
    }

    private record ValidationResult(boolean valid, int mismatches, float maxDiff, int maxDiffIndex, int maxDiffRow, int maxDiffCol) {}
}
