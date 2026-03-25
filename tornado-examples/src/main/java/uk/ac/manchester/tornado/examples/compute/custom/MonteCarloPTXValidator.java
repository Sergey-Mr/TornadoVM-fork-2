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
 * Validates PTX Monte Carlo kernel output against sequential baseline.
 * Separate from benchmark to avoid affecting GPU warmup state.
 *
 * Usage: java ... MonteCarloPTXValidator <kernel1.ptx> [kernel2.ptx] ... [--samples=N]
 */
public class MonteCarloPTXValidator {

    private static final int DEFAULT_NUM_SAMPLES = 1000000; // 1M for validation
    private static final int LOCAL_WORK_SIZE = 256;

    // PTX entry point from TornadoVM-generated kernel
    private static final String ENTRY_POINT = "s0_taskgraph_computemontecarlo_arrays_floatarray_16777216";

    // Sequential Monte Carlo using same LCG as TornadoVM
    private static void monteCarloSequential(FloatArray output, int iterations) {
        for (int j = 0; j < iterations; j++) {
            long seed = j;
            // Same LCG as TornadoVM kernel
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            float x = (seed & 0xFFFFFFF) / 268435455f;

            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            float y = (seed & 0xFFFFFFF) / 268435455f;

            float dist = (float) Math.sqrt(x * x + y * y);
            output.set(j, (dist <= 1.0f) ? 1.0f : 0.0f);
        }
    }

    private static double calculatePi(FloatArray output, int numSamples) {
        float sum = 0.0f;
        for (int i = 0; i < numSamples; i++) {
            sum += output.get(i);
        }
        return (4.0 * sum) / numSamples;
    }

    private static ValidationResult validate(FloatArray reference, FloatArray candidate, int numSamples) {
        int mismatches = 0;
        int firstMismatchIndex = -1;

        for (int i = 0; i < numSamples; i++) {
            // Binary output: should be exactly 0.0 or 1.0
            if (Math.abs(reference.get(i) - candidate.get(i)) > 0.001f) {
                if (firstMismatchIndex == -1) {
                    firstMismatchIndex = i;
                }
                mismatches++;
            }
        }

        double refPi = calculatePi(reference, numSamples);
        double candPi = calculatePi(candidate, numSamples);

        return new ValidationResult(mismatches == 0, mismatches, firstMismatchIndex, refPi, candPi);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: MonteCarloPTXValidator <kernel1.ptx> [kernel2.ptx] ... [--samples=N]");
            System.exit(1);
        }

        // Parse arguments
        int numSamples = DEFAULT_NUM_SAMPLES;
        java.util.List<String> kernelPaths = new java.util.ArrayList<>();

        for (String arg : args) {
            if (arg.startsWith("--samples=")) {
                numSamples = Integer.parseInt(arg.substring(10));
            } else {
                kernelPaths.add(arg);
            }
        }

        if (kernelPaths.isEmpty()) {
            System.out.println("Error: No kernel files specified");
            System.exit(1);
        }

        FloatArray referenceOutput = new FloatArray(numSamples);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

        // Compute sequential reference
        System.out.println("Computing sequential Monte Carlo reference (" + numSamples + " samples)...");
        long start = System.currentTimeMillis();
        monteCarloSequential(referenceOutput, numSamples);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("Sequential computation took: " + elapsed + " ms");

        double referencePi = calculatePi(referenceOutput, numSamples);
        System.out.printf("Reference PI estimate: %.8f (error: %.2e)%n", referencePi, Math.abs(Math.PI - referencePi));

        System.out.println();
        System.out.println("Validation Results");
        System.out.println("=".repeat(50));

        // Validate each kernel
        for (String kernelPath : kernelPaths) {
            FloatArray kernelOutput = new FloatArray(numSamples);

            // PTX kernel signature: (kernel_context, output, iterations)
            AccessorParameters accessors = new AccessorParameters(2);
            accessors.set(0, kernelOutput, Access.WRITE_ONLY);
            accessors.set(1, Long.valueOf(numSamples), Access.NONE);  // PTX uses .u64

            TaskGraph graph = new TaskGraph("validate")
                    .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, kernelOutput);

            ImmutableTaskGraph snapshot = graph.snapshot();

            WorkerGrid1D worker = new WorkerGrid1D(numSamples);
            worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
            GridScheduler scheduler = new GridScheduler("validate.t0", worker);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.withDevice(device).withGridScheduler(scheduler);
                plan.execute();
            }

            // Validate
            ValidationResult result = validate(referenceOutput, kernelOutput, numSamples);

            System.out.printf("%s%n", kernelPath);
            if (result.valid) {
                System.out.printf("  PASSED (PI: %.8f)%n", result.candidatePi);
            } else {
                System.out.printf("  FAILED: %d mismatches%n", result.mismatches);
                System.out.printf("    First mismatch at index %d: expected=%.1f, got=%.1f%n",
                        result.firstMismatchIndex,
                        referenceOutput.get(result.firstMismatchIndex),
                        kernelOutput.get(result.firstMismatchIndex));
                System.out.printf("    Reference PI: %.8f, Kernel PI: %.8f%n", result.referencePi, result.candidatePi);
            }
        }
    }

    private record ValidationResult(boolean valid, int mismatches, int firstMismatchIndex,
            double referencePi, double candidatePi) {}
}
