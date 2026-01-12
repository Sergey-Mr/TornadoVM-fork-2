package uk.ac.manchester.tornado.examples.compute.custom;

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
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/**
 * Validates Mandelbrot kernel output against sequential baseline.
 * Separate from benchmark to avoid affecting GPU warmup state.
 *
 * Usage: java ... MandelbrotValidator <kernel1.cl> [kernel2.cl] ... [--size=N]
 */
public class MandelbrotValidator {

    private static final int DEFAULT_SIZE = 512;
    private static final int LOCAL_WORK_SIZE = 16;
    private static final int ITERATIONS = 10000;
    private static final String ENTRY_POINT = "mandelbrotTornado";

    // Sequential Mandelbrot computation for reference
    private static void mandelbrotSequential(int size, ShortArray output) {
        float space = 2.0f / size;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int idx = i * size + j;

                float Zr = 0.0f;
                float Zi = 0.0f;
                float Cr = (1 * j * space - 1.5f);
                float Ci = (1 * i * space - 1.0f);

                float ZrN = 0;
                float ZiN = 0;
                int count;

                for (count = 0; count < ITERATIONS && ZiN + ZrN <= 4.0f; count++) {
                    Zi = 2.0f * Zr * Zi + Ci;
                    Zr = ZrN - ZiN + Cr;
                    ZiN = Zi * Zi;
                    ZrN = Zr * Zr;
                }

                short iterCount = (short) ((((float) count / ITERATIONS) * 255));
                output.set(idx, iterCount);
            }
        }
    }

    private static ValidationResult validate(ShortArray reference, ShortArray candidate, int size) {
        int mismatches = 0;
        int maxDiff = 0;
        int maxDiffIndex = -1;

        for (int i = 0; i < reference.getSize(); i++) {
            int diff = Math.abs(reference.get(i) - candidate.get(i));
            if (diff > maxDiff) {
                maxDiff = diff;
                maxDiffIndex = i;
            }
            // Allow small tolerance for floating point differences
            if (diff > 1) {
                mismatches++;
            }
        }

        int maxDiffRow = maxDiffIndex / size;
        int maxDiffCol = maxDiffIndex % size;

        return new ValidationResult(mismatches == 0, mismatches, maxDiff, maxDiffIndex, maxDiffRow, maxDiffCol);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: MandelbrotValidator <kernel1.cl> [kernel2.cl] ... [--size=N]");
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

        ShortArray referenceOutput = new ShortArray(size * size);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

        // Compute sequential reference
        System.out.println("Computing sequential Mandelbrot reference (" + size + "x" + size + ")...");
        long start = System.currentTimeMillis();
        mandelbrotSequential(size, referenceOutput);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("Sequential computation took: " + elapsed + " ms");

        System.out.println();
        System.out.println("Validation Results (tolerance=1)");
        System.out.println("=".repeat(50));

        // Validate each kernel
        for (String kernelPath : kernelPaths) {
            ShortArray kernelOutput = new ShortArray(size * size);

            // Set up and run kernel once
            AccessorParameters accessors = new AccessorParameters(2);
            accessors.set(0, Integer.valueOf(size), Access.NONE);
            accessors.set(1, kernelOutput, Access.WRITE_ONLY);

            TaskGraph graph = new TaskGraph("validate")
                    .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, kernelOutput);

            ImmutableTaskGraph snapshot = graph.snapshot();

            WorkerGrid2D worker = new WorkerGrid2D(size, size);
            worker.setLocalWork(LOCAL_WORK_SIZE, LOCAL_WORK_SIZE, 1);
            GridScheduler scheduler = new GridScheduler("validate.t0", worker);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.withDevice(device).withGridScheduler(scheduler);
                plan.execute();
            }

            // Validate
            ValidationResult result = validate(referenceOutput, kernelOutput, size);

            System.out.printf("%s%n", kernelPath);
            if (result.valid) {
                System.out.printf("  PASSED (max diff: %d at [%d,%d])%n",
                        result.maxDiff, result.maxDiffRow, result.maxDiffCol);
            } else {
                System.out.printf("  FAILED: %d mismatches (max diff: %d at [%d,%d])%n",
                        result.mismatches, result.maxDiff, result.maxDiffRow, result.maxDiffCol);
                System.out.printf("    Expected: %d, Got: %d%n",
                        referenceOutput.get(result.maxDiffIndex), kernelOutput.get(result.maxDiffIndex));
            }
        }
    }

    private record ValidationResult(boolean valid, int mismatches, int maxDiff, int maxDiffIndex, int maxDiffRow, int maxDiffCol) {}
}
