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
 * Validates Reduction Add Floats kernel output against sequential baseline.
 * Separate from benchmark to avoid affecting GPU warmup state.
 *
 * Usage: java ... ReductionAddFloatsValidator <kernel1.cl> [kernel2.cl] ... [--size=N]
 */
public class ReductionAddFloatsValidator {

    private static final int DEFAULT_SIZE = 1000000; // 1M for validation
    private static final int LOCAL_WORK_SIZE = 256;
    private static final String ENTRY_POINT = "reductionAddFloats";
    private static final float RELATIVE_TOLERANCE = 1e-4f; // 0.01% relative error
    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array) {
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, RANDOM.nextFloat());
        }
    }

    // Sequential reduction for reference
    private static float reductionSequential(FloatArray input) {
        float sum = 0.0f;
        for (int i = 0; i < input.getSize(); i++) {
            sum += input.get(i);
        }
        return sum;
    }

    // Kahan summation for more accurate reference
    private static float reductionKahan(FloatArray input) {
        float sum = 0.0f;
        float c = 0.0f; // Compensation for lost low-order bits
        for (int i = 0; i < input.getSize(); i++) {
            float y = input.get(i) - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        return sum;
    }

    private static ValidationResult validate(float reference, float candidate, int size) {
        float absoluteDiff = Math.abs(reference - candidate);
        float relativeDiff = absoluteDiff / Math.abs(reference);

        boolean valid = relativeDiff <= RELATIVE_TOLERANCE;

        return new ValidationResult(valid, reference, candidate, absoluteDiff, relativeDiff);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: ReductionAddFloatsValidator <kernel1.cl> [kernel2.cl] ... [--size=N]");
            System.exit(1);
        }

        // Parse arguments
        int size = DEFAULT_SIZE;
        java.util.List<String> kernelPaths = new java.util.ArrayList<>();

        for (String arg : args) {
            if (arg.startsWith("--size=")) {
                size = Integer.parseInt(arg.substring(7));
            } else {
                kernelPaths.add(arg);
            }
        }

        if (kernelPaths.isEmpty()) {
            System.out.println("Error: No kernel files specified");
            System.exit(1);
        }

        int numWorkGroups = (size + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;

        FloatArray input = new FloatArray(size);
        fillRandomData(input);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

        // Compute sequential reference (using Kahan for accuracy)
        System.out.println("Computing sequential reduction reference (" + size + " elements)...");
        long start = System.currentTimeMillis();
        float referenceSum = reductionKahan(input);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("Sequential computation took: " + elapsed + " ms");
        System.out.printf("Reference sum: %.6f%n", referenceSum);

        System.out.println();
        System.out.println("Validation Results (relative tolerance=" + RELATIVE_TOLERANCE + ")");
        System.out.println("=".repeat(50));

        // Validate each kernel
        for (String kernelPath : kernelPaths) {
            FloatArray partialSums = new FloatArray(numWorkGroups);

            // Set up and run kernel
            AccessorParameters accessors = new AccessorParameters(2);
            accessors.set(0, input, Access.READ_ONLY);
            accessors.set(1, partialSums, Access.READ_WRITE);

            TaskGraph graph = new TaskGraph("validate")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, input)
                    .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, partialSums);

            ImmutableTaskGraph snapshot = graph.snapshot();

            WorkerGrid1D worker = new WorkerGrid1D(size);
            worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
            GridScheduler scheduler = new GridScheduler("validate.t0", worker);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.withDevice(device).withGridScheduler(scheduler);
                plan.execute();
            }

            // Sum partial results on host
            float kernelSum = 0.0f;
            for (int i = 0; i < numWorkGroups; i++) {
                kernelSum += partialSums.get(i);
            }

            // Validate
            ValidationResult result = validate(referenceSum, kernelSum, size);

            System.out.printf("%s%n", kernelPath);
            if (result.valid) {
                System.out.printf("  PASSED (sum=%.6f, rel_diff=%.2e)%n",
                        result.candidate, result.relativeDiff);
            } else {
                System.out.printf("  FAILED (rel_diff=%.2e > tolerance %.2e)%n",
                        result.relativeDiff, RELATIVE_TOLERANCE);
                System.out.printf("    Expected: %.6f, Got: %.6f (abs_diff: %.6f)%n",
                        result.reference, result.candidate, result.absoluteDiff);
            }
        }
    }

    private record ValidationResult(boolean valid, float reference, float candidate,
            float absoluteDiff, float relativeDiff) {}
}
