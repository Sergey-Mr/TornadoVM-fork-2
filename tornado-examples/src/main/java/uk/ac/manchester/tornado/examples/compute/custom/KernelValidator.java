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
import uk.ac.manchester.tornado.examples.compute.MatrixVectorRowMajor;

/**
 * Validates kernel output against sequential baseline.
 * Separate from benchmark to avoid affecting GPU warmup state.
 *
 * Usage: java ... KernelValidator <kernel1.cl> [kernel2.cl] ...
 */
public class KernelValidator {

    private static final int INPUT_DIM = 8192;
    private static final int OUTPUT_DIM = 2048;
    private static final int LOCAL_WORK_GROUP_SIZE = 128;
    private static final String ENTRY_POINT = "matrixVectorGeneric";
    private static final float TOLERANCE = 1e-4f;
    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array, float min, float max) {
        float range = max - min;
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, min + RANDOM.nextFloat() * range);
        }
    }

    private static ValidationResult validate(FloatArray reference, FloatArray candidate) {
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
        return new ValidationResult(mismatches == 0, mismatches, maxDiff, maxDiffIndex);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: KernelValidator <kernel1.cl> [kernel2.cl] ...");
            System.exit(1);
        }

        FloatArray input = new FloatArray(INPUT_DIM);
        FloatArray weights = new FloatArray(INPUT_DIM * OUTPUT_DIM);
        FloatArray referenceOutput = new FloatArray(OUTPUT_DIM);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

        // Use same random seed as benchmark
        fillRandomData(input, -1.0f, 1.0f);
        fillRandomData(weights, -0.1f, 0.1f);

        // Compute sequential reference
        System.out.println("Computing sequential reference...");
        MatrixVectorRowMajor.matrixVectorSequential(input, referenceOutput, weights, INPUT_DIM, OUTPUT_DIM);

        System.out.println();
        System.out.println("Validation Results (tolerance=" + TOLERANCE + ")");
        System.out.println("=".repeat(50));

        // Validate each kernel
        for (String kernelPath : args) {
            FloatArray kernelOutput = new FloatArray(OUTPUT_DIM);

            // Set up and run kernel once
            AccessorParameters accessors = new AccessorParameters(6);
            accessors.set(0, input, Access.READ_ONLY);
            accessors.set(1, kernelOutput, Access.WRITE_ONLY);
            accessors.set(2, weights, Access.READ_ONLY);
            accessors.set(3, Integer.valueOf(INPUT_DIM), Access.NONE);
            accessors.set(4, Integer.valueOf(OUTPUT_DIM), Access.NONE);
            accessors.set(5, Integer.valueOf(LOCAL_WORK_GROUP_SIZE), Access.NONE);

            TaskGraph graph = new TaskGraph("validate")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, input, weights)
                    .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, kernelOutput);

            ImmutableTaskGraph snapshot = graph.snapshot();

            WorkerGrid1D worker = new WorkerGrid1D(OUTPUT_DIM * LOCAL_WORK_GROUP_SIZE);
            worker.setLocalWork(LOCAL_WORK_GROUP_SIZE, 1, 1);
            GridScheduler scheduler = new GridScheduler("validate.t0", worker);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.withDevice(device).withGridScheduler(scheduler);
                plan.execute();
            }

            // Validate
            ValidationResult result = validate(referenceOutput, kernelOutput);

            System.out.printf("%s%n", kernelPath);
            if (result.valid) {
                System.out.printf("  PASSED (max diff: %.2e at index %d)%n", result.maxDiff, result.maxDiffIndex);
            } else {
                System.out.printf("  FAILED: %d mismatches (max diff: %.2e at index %d)%n",
                        result.mismatches, result.maxDiff, result.maxDiffIndex);
            }
        }
    }

    private record ValidationResult(boolean valid, int mismatches, float maxDiff, int maxDiffIndex) {}
}
