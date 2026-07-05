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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/**
 * Validates Flash Attention kernel output against sequential baseline.
 * Separate from benchmark to avoid affecting GPU warmup state.
 *
 * Usage: java ... FlashAttentionValidator <kernel1.cl> [kernel2.cl] ... [--heads=N] [--dim=N] [--ctx=N]
 */
public class FlashAttentionValidator {

    // MUST match the hardcoded values in the generated kernel!
    private static final int DEFAULT_N_HEADS = 32;
    private static final int DEFAULT_HEAD_SIZE = 128;
    private static final int DEFAULT_CONTEXT_LENGTH = 256;  // Can be smaller for validation
    private static final int DEFAULT_KV_HEADS = 8;  // kvDim = 8 * 128 = 1024

    private static final String ENTRY_POINT = "processHeadsFlashAttention";
    private static final float TOLERANCE = 1e-2f;  // Relative tolerance
    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array) {
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, (RANDOM.nextFloat() - 0.5f) * 0.1f);
        }
    }

    /**
     * Sequential Flash Attention implementation for reference.
     * Implements standard scaled dot-product attention with online softmax.
     */
    private static void flashAttentionSequential(
            FloatArray q, FloatArray keyCache, FloatArray valueCache, FloatArray output,
            int nHeads, int headSize, int kvDim, int kvMul,
            int position, int layer, int contextLength) {

        int loff = layer * contextLength * kvDim;

        for (int h = 0; h < nHeads; h++) {
            int kvHeadIdx = h / kvMul;

            // Compute attention scores and apply softmax
            float[] scores = new float[position + 1];
            float maxScore = Float.NEGATIVE_INFINITY;

            // Compute Q * K^T / sqrt(headSize)
            for (int t = 0; t <= position; t++) {
                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    int qIdx = h * headSize + d;
                    int kIdx = loff + t * kvDim + kvHeadIdx * headSize + d;
                    score += q.get(qIdx) * keyCache.get(kIdx);
                }
                score /= (float) Math.sqrt(headSize);
                scores[t] = score;
                if (score > maxScore) {
                    maxScore = score;
                }
            }

            // Softmax with numerical stability
            float sumExp = 0.0f;
            for (int t = 0; t <= position; t++) {
                scores[t] = (float) Math.exp(scores[t] - maxScore);
                sumExp += scores[t];
            }
            for (int t = 0; t <= position; t++) {
                scores[t] /= sumExp;
            }

            // Compute attention output: softmax(scores) * V
            for (int d = 0; d < headSize; d++) {
                float sum = 0.0f;
                for (int t = 0; t <= position; t++) {
                    int vIdx = loff + t * kvDim + kvHeadIdx * headSize + d;
                    sum += scores[t] * valueCache.get(vIdx);
                }
                output.set(h * headSize + d, sum);
            }
        }
    }

    private static ValidationResult validate(FloatArray reference, FloatArray candidate, int size) {
        int mismatches = 0;
        float maxRelDiff = 0;
        int maxDiffIndex = -1;

        for (int i = 0; i < size; i++) {
            float ref = reference.get(i);
            float cand = candidate.get(i);
            float absDiff = Math.abs(ref - cand);
            float relDiff = (Math.abs(ref) > 1e-6f) ? absDiff / Math.abs(ref) : absDiff;

            if (relDiff > maxRelDiff) {
                maxRelDiff = relDiff;
                maxDiffIndex = i;
            }
            if (relDiff > TOLERANCE) {
                mismatches++;
            }
        }

        return new ValidationResult(mismatches == 0, mismatches, maxRelDiff, maxDiffIndex);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: FlashAttentionValidator <kernel1.cl> [kernel2.cl] ... [--heads=N] [--dim=N] [--ctx=N]");
            System.exit(1);
        }

        // Parse arguments
        int nHeads = DEFAULT_N_HEADS;
        int headSize = DEFAULT_HEAD_SIZE;
        int contextLength = DEFAULT_CONTEXT_LENGTH;
        java.util.List<String> kernelPaths = new java.util.ArrayList<>();

        for (String arg : args) {
            if (arg.startsWith("--heads=")) {
                nHeads = Integer.parseInt(arg.substring(8));
            } else if (arg.startsWith("--dim=")) {
                headSize = Integer.parseInt(arg.substring(6));
            } else if (arg.startsWith("--ctx=")) {
                contextLength = Integer.parseInt(arg.substring(6));
            } else {
                kernelPaths.add(arg);
            }
        }

        if (kernelPaths.isEmpty()) {
            System.out.println("Error: No kernel files specified");
            System.exit(1);
        }

        // Derived dimensions
        int nKvHeads = Math.max(1, nHeads / 4);  // GQA ratio
        int kvDim = nKvHeads * headSize;
        int kvMul = nHeads / nKvHeads;
        int numLayers = 1;  // Single layer for validation
        int layer = 0;
        int position = contextLength - 1;

        System.out.println("=== Flash Attention Validator ===");
        System.out.println("nHeads: " + nHeads);
        System.out.println("headSize: " + headSize);
        System.out.println("contextLength: " + contextLength);
        System.out.println("kvDim: " + kvDim);
        System.out.println("position: " + position);

        // Allocate arrays
        FloatArray q = new FloatArray(nHeads * headSize);
        FloatArray keyCache = new FloatArray(numLayers * contextLength * kvDim);
        FloatArray valueCache = new FloatArray(numLayers * contextLength * kvDim);
        FloatArray referenceOutput = new FloatArray(nHeads * headSize);
        IntArray positionHolder = new IntArray(1);

        // Initialize data with same seed for reproducibility
        fillRandomData(q);
        fillRandomData(keyCache);
        fillRandomData(valueCache);
        positionHolder.set(0, position);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

        // Compute sequential reference
        System.out.println("\nComputing sequential reference...");
        long start = System.currentTimeMillis();
        flashAttentionSequential(q, keyCache, valueCache, referenceOutput,
                nHeads, headSize, kvDim, kvMul, position, layer, contextLength);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("Sequential computation took: " + elapsed + " ms");

        System.out.println();
        System.out.println("Validation Results (tolerance=" + TOLERANCE + ")");
        System.out.println("=".repeat(50));

        // Validate each kernel
        for (String kernelPath : kernelPaths) {
            FloatArray kernelOutput = new FloatArray(nHeads * headSize);

            // Set up kernel
            AccessorParameters accessors = new AccessorParameters(11);
            accessors.set(0, q, Access.READ_ONLY);
            accessors.set(1, keyCache, Access.READ_ONLY);
            accessors.set(2, valueCache, Access.READ_ONLY);
            accessors.set(3, kernelOutput, Access.WRITE_ONLY);
            accessors.set(4, Integer.valueOf(nHeads), Access.NONE);
            accessors.set(5, Integer.valueOf(headSize), Access.NONE);
            accessors.set(6, Integer.valueOf(kvDim), Access.NONE);
            accessors.set(7, Integer.valueOf(kvMul), Access.NONE);
            accessors.set(8, positionHolder, Access.READ_ONLY);
            accessors.set(9, Integer.valueOf(layer), Access.NONE);
            accessors.set(10, Integer.valueOf(contextLength), Access.NONE);

            TaskGraph graph = new TaskGraph("validate")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, q, keyCache, valueCache, positionHolder)
                    .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, kernelOutput);

            ImmutableTaskGraph snapshot = graph.snapshot();

            WorkerGrid1D worker = new WorkerGrid1D(nHeads * headSize);
            worker.setLocalWork(headSize, 1, 1);
            GridScheduler scheduler = new GridScheduler("validate.t0", worker);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.withDevice(device).withGridScheduler(scheduler);
                plan.execute();
            }

            // Validate
            ValidationResult result = validate(referenceOutput, kernelOutput, nHeads * headSize);

            System.out.printf("%s%n", kernelPath);
            if (result.valid) {
                System.out.printf("  PASSED (max rel diff: %.2e at index %d)%n",
                        result.maxRelDiff, result.maxDiffIndex);
            } else {
                System.out.printf("  FAILED: %d mismatches (max rel diff: %.2e at index %d)%n",
                        result.mismatches, result.maxRelDiff, result.maxDiffIndex);
                if (result.maxDiffIndex >= 0) {
                    System.out.printf("    Expected: %.6f, Got: %.6f%n",
                            referenceOutput.get(result.maxDiffIndex),
                            kernelOutput.get(result.maxDiffIndex));
                }
            }
        }
    }

    private record ValidationResult(boolean valid, int mismatches, float maxRelDiff, int maxDiffIndex) {}
}
