package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.Random;

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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/**
 * Validates Blur Filter kernel output against sequential baseline.
 * Separate from benchmark to avoid affecting GPU warmup state.
 *
 * Usage: java ... BlurFilterValidator <kernel1.cl> [kernel2.cl] ... [--size=N] [--filter=W]
 */
public class BlurFilterValidator {

    private static final int DEFAULT_IMAGE_SIZE = 512;
    private static final int DEFAULT_FILTER_WIDTH = 15;
    private static final int LOCAL_WORK_SIZE = 16;
    private static final String ENTRY_POINT = "compute";
    private static final int TOLERANCE = 1; // Allow ±1 difference due to float rounding
    private static final Random RANDOM = new Random(42);

    private static void fillRandomImage(IntArray image) {
        for (int i = 0; i < image.getSize(); i++) {
            image.set(i, RANDOM.nextInt(256));
        }
    }

    private static void createGaussianFilter(FloatArray filter, int filterWidth) {
        float weight = 1.0f / (filterWidth * filterWidth);
        for (int i = 0; i < filterWidth * filterWidth; i++) {
            filter.set(i, weight);
        }
    }

    // Sequential blur filter computation
    private static void blurFilterSequential(IntArray input, IntArray output,
            int numRows, int numCols, FloatArray filter, int filterWidth) {

        int halfFilter = filterWidth / 2;

        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                float result = 0.0f;

                for (int filterR = -halfFilter; filterR <= halfFilter; filterR++) {
                    for (int filterC = -halfFilter; filterC <= halfFilter; filterC++) {
                        // Clamp to image boundaries
                        int imageR = Math.min(Math.max(r + filterR, 0), numRows - 1);
                        int imageC = Math.min(Math.max(c + filterC, 0), numCols - 1);

                        int filterIdx = (filterR + halfFilter) * filterWidth + (filterC + halfFilter);
                        result += input.get(imageR * numCols + imageC) * filter.get(filterIdx);
                    }
                }

                // Clamp result to [0, 255]
                int finalValue = Math.min(Math.max((int) result, 0), 255);
                output.set(r * numCols + c, finalValue);
            }
        }
    }

    private static ValidationResult validate(IntArray reference, IntArray candidate,
            int numRows, int numCols) {
        int mismatches = 0;
        int maxDiff = 0;
        int maxDiffIndex = -1;

        for (int i = 0; i < reference.getSize(); i++) {
            int diff = Math.abs(reference.get(i) - candidate.get(i));
            if (diff > maxDiff) {
                maxDiff = diff;
                maxDiffIndex = i;
            }
            if (diff > TOLERANCE) {
                mismatches++;
            }
        }

        int maxDiffRow = maxDiffIndex / numCols;
        int maxDiffCol = maxDiffIndex % numCols;

        return new ValidationResult(mismatches == 0, mismatches, maxDiff,
                maxDiffIndex, maxDiffRow, maxDiffCol);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: BlurFilterValidator <kernel1.cl> [kernel2.cl] ... [--size=N] [--filter=W]");
            System.exit(1);
        }

        // Parse arguments
        int imageSize = DEFAULT_IMAGE_SIZE;
        int filterWidth = DEFAULT_FILTER_WIDTH;
        java.util.List<String> kernelPaths = new java.util.ArrayList<>();

        for (String arg : args) {
            if (arg.startsWith("--size=")) {
                imageSize = Integer.parseInt(arg.substring(7));
            } else if (arg.startsWith("--filter=")) {
                filterWidth = Integer.parseInt(arg.substring(9));
            } else {
                kernelPaths.add(arg);
            }
        }

        if (kernelPaths.isEmpty()) {
            System.out.println("Error: No kernel files specified");
            System.exit(1);
        }

        int numRows = imageSize;
        int numCols = imageSize;

        IntArray inputChannel = new IntArray(numRows * numCols);
        IntArray referenceOutput = new IntArray(numRows * numCols);
        FloatArray filter = new FloatArray(filterWidth * filterWidth);

        fillRandomImage(inputChannel);
        createGaussianFilter(filter, filterWidth);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

        // Compute sequential reference
        System.out.println("Computing sequential Blur Filter reference (" + numRows + "x" + numCols + ", filter=" + filterWidth + "x" + filterWidth + ")...");
        long start = System.currentTimeMillis();
        blurFilterSequential(inputChannel, referenceOutput, numRows, numCols, filter, filterWidth);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("Sequential computation took: " + elapsed + " ms");

        System.out.println();
        System.out.println("Validation Results (tolerance=" + TOLERANCE + ")");
        System.out.println("=".repeat(50));

        // Validate each kernel
        for (String kernelPath : kernelPaths) {
            IntArray kernelOutput = new IntArray(numRows * numCols);

            // Set up and run kernel once
            AccessorParameters accessors = new AccessorParameters(6);
            accessors.set(0, inputChannel, Access.READ_ONLY);
            accessors.set(1, kernelOutput, Access.WRITE_ONLY);
            accessors.set(2, Integer.valueOf(numRows), Access.NONE);
            accessors.set(3, Integer.valueOf(numCols), Access.NONE);
            accessors.set(4, filter, Access.READ_ONLY);
            accessors.set(5, Integer.valueOf(filterWidth), Access.NONE);

            TaskGraph graph = new TaskGraph("validate")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, inputChannel, filter)
                    .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, kernelOutput);

            ImmutableTaskGraph snapshot = graph.snapshot();

            WorkerGrid2D worker = new WorkerGrid2D(numRows, numCols);
            worker.setLocalWork(LOCAL_WORK_SIZE, LOCAL_WORK_SIZE, 1);
            GridScheduler scheduler = new GridScheduler("validate.t0", worker);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.withDevice(device).withGridScheduler(scheduler);
                plan.execute();
            }

            // Validate
            ValidationResult result = validate(referenceOutput, kernelOutput, numRows, numCols);

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

    private record ValidationResult(boolean valid, int mismatches, int maxDiff,
            int maxDiffIndex, int maxDiffRow, int maxDiffCol) {}
}
