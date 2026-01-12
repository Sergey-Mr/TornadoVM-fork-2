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
 * Validates Black-Scholes kernel output against sequential baseline.
 * Separate from benchmark to avoid affecting GPU warmup state.
 *
 * Usage: java ... BlackScholesValidator <kernel1.cl> [kernel2.cl] ... [--options=N]
 */
public class BlackScholesValidator {

    private static final int DEFAULT_NUM_OPTIONS = 100000;
    private static final int LOCAL_WORK_SIZE = 256;
    private static final String ENTRY_POINT = "blackScholesKernel";
    private static final float TOLERANCE = 1e-4f;
    private static final Random RANDOM = new Random(42);

    // CND approximation coefficients (same as kernel)
    private static final float c1 = 0.319381530f;
    private static final float c2 = -0.356563782f;
    private static final float c3 = 1.781477937f;
    private static final float c4 = -1.821255978f;
    private static final float c5 = 1.330274429f;
    private static final float oneBySqrt2pi = 0.39894228040143267793994605993438f;

    private static void fillRandomData(FloatArray array) {
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, RANDOM.nextFloat());
        }
    }

    // Cumulative Normal Distribution
    private static float cnd(float X) {
        float L = Math.abs(X);
        float k = 1.0f / (1.0f + 0.2316419f * L);
        float k2 = k * k;
        float k3 = k2 * k;
        float k4 = k3 * k;
        float k5 = k4 * k;
        float w = c1 * k + c2 * k2 + c3 * k3 + c4 * k4 + c5 * k5;
        w = 1.0f - oneBySqrt2pi * (float) Math.exp(-0.5f * L * L) * w;
        return (X < 0.0f) ? 1.0f - w : w;
    }

    // Sequential Black-Scholes computation
    private static void blackScholesSequential(FloatArray input, FloatArray callResult, FloatArray putResult) {
        for (int idx = 0; idx < input.getSize(); idx++) {
            float rand = input.get(idx);

            // Generate option parameters from random input
            float S = 10.0f + rand * 90.0f;  // Stock price [10, 100]
            float K = 10.0f + rand * 90.0f;  // Strike price [10, 100]
            float T = 1.0f + rand * 9.0f;    // Time to expiration [1, 10]
            float r = 0.01f + rand * 0.04f;  // Risk-free rate [0.01, 0.05]
            float v = 0.01f + rand * 0.09f;  // Volatility [0.01, 0.10]

            float sqrtT = (float) Math.sqrt(T);
            float d1 = ((float) Math.log(S / K) + (r + 0.5f * v * v) * T) / (v * sqrtT);
            float d2 = d1 - v * sqrtT;

            float cndD1 = cnd(d1);
            float cndD2 = cnd(d2);

            float expRT = (float) Math.exp(-r * T);

            callResult.set(idx, S * cndD1 - K * expRT * cndD2);
            putResult.set(idx, K * expRT * (1.0f - cndD2) - S * (1.0f - cndD1));
        }
    }

    private static ValidationResult validate(FloatArray refCall, FloatArray refPut,
            FloatArray candCall, FloatArray candPut, int numOptions) {
        int callMismatches = 0;
        int putMismatches = 0;
        float maxCallDiff = 0;
        float maxPutDiff = 0;
        int maxCallDiffIndex = -1;
        int maxPutDiffIndex = -1;

        for (int i = 0; i < numOptions; i++) {
            float callDiff = Math.abs(refCall.get(i) - candCall.get(i));
            float putDiff = Math.abs(refPut.get(i) - candPut.get(i));

            if (callDiff > maxCallDiff) {
                maxCallDiff = callDiff;
                maxCallDiffIndex = i;
            }
            if (putDiff > maxPutDiff) {
                maxPutDiff = putDiff;
                maxPutDiffIndex = i;
            }

            if (callDiff > TOLERANCE) callMismatches++;
            if (putDiff > TOLERANCE) putMismatches++;
        }

        return new ValidationResult(
                callMismatches == 0 && putMismatches == 0,
                callMismatches, putMismatches,
                maxCallDiff, maxPutDiff,
                maxCallDiffIndex, maxPutDiffIndex);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: BlackScholesValidator <kernel1.cl> [kernel2.cl] ... [--options=N]");
            System.exit(1);
        }

        // Parse arguments
        int numOptions = DEFAULT_NUM_OPTIONS;
        java.util.List<String> kernelPaths = new java.util.ArrayList<>();

        for (String arg : args) {
            if (arg.startsWith("--options=")) {
                numOptions = Integer.parseInt(arg.substring(10));
            } else {
                kernelPaths.add(arg);
            }
        }

        if (kernelPaths.isEmpty()) {
            System.out.println("Error: No kernel files specified");
            System.exit(1);
        }

        FloatArray input = new FloatArray(numOptions);
        FloatArray refCall = new FloatArray(numOptions);
        FloatArray refPut = new FloatArray(numOptions);

        fillRandomData(input);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

        // Compute sequential reference
        System.out.println("Computing sequential Black-Scholes reference (" + numOptions + " options)...");
        long start = System.currentTimeMillis();
        blackScholesSequential(input, refCall, refPut);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("Sequential computation took: " + elapsed + " ms");

        System.out.println();
        System.out.println("Validation Results (tolerance=" + TOLERANCE + ")");
        System.out.println("=".repeat(50));

        // Validate each kernel
        for (String kernelPath : kernelPaths) {
            FloatArray kernelCall = new FloatArray(numOptions);
            FloatArray kernelPut = new FloatArray(numOptions);

            // Set up and run kernel once
            AccessorParameters accessors = new AccessorParameters(3);
            accessors.set(0, input, Access.READ_ONLY);
            accessors.set(1, kernelCall, Access.WRITE_ONLY);
            accessors.set(2, kernelPut, Access.WRITE_ONLY);

            TaskGraph graph = new TaskGraph("validate")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, input)
                    .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, kernelCall, kernelPut);

            ImmutableTaskGraph snapshot = graph.snapshot();

            WorkerGrid1D worker = new WorkerGrid1D(numOptions);
            worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
            GridScheduler scheduler = new GridScheduler("validate.t0", worker);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.withDevice(device).withGridScheduler(scheduler);
                plan.execute();
            }

            // Validate
            ValidationResult result = validate(refCall, refPut, kernelCall, kernelPut, numOptions);

            System.out.printf("%s%n", kernelPath);
            if (result.valid) {
                System.out.printf("  PASSED (max call diff: %.2e, max put diff: %.2e)%n",
                        result.maxCallDiff, result.maxPutDiff);
            } else {
                System.out.printf("  FAILED%n");
                if (result.callMismatches > 0) {
                    System.out.printf("    Call mismatches: %d (max diff: %.2e at index %d)%n",
                            result.callMismatches, result.maxCallDiff, result.maxCallDiffIndex);
                    System.out.printf("      Expected: %.6f, Got: %.6f%n",
                            refCall.get(result.maxCallDiffIndex), kernelCall.get(result.maxCallDiffIndex));
                }
                if (result.putMismatches > 0) {
                    System.out.printf("    Put mismatches: %d (max diff: %.2e at index %d)%n",
                            result.putMismatches, result.maxPutDiff, result.maxPutDiffIndex);
                    System.out.printf("      Expected: %.6f, Got: %.6f%n",
                            refPut.get(result.maxPutDiffIndex), kernelPut.get(result.maxPutDiffIndex));
                }
            }
        }
    }

    private record ValidationResult(boolean valid,
            int callMismatches, int putMismatches,
            float maxCallDiff, float maxPutDiff,
            int maxCallDiffIndex, int maxPutDiffIndex) {}
}
