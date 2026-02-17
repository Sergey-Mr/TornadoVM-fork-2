package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;
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
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

/**
 * PTX Single-kernel benchmark for Flash Attention (from GPULlama3.java).
 * Measures ONLY kernel execution time - run once per kernel for fair comparison.
 *
 * Usage: java ... FlashAttentionPTXBenchmark <kernel.ptx> [nHeads] [headSize] [contextLength]
 *
 * Defaults: nHeads=32, headSize=128, contextLength=2048
 *
 * NOTE: Update ENTRY_POINT to match your generated PTX kernel function name.
 */
public class FlashAttentionPTXBenchmark {

    // Default Llama-3 8B dimensions
    private static final int DEFAULT_N_HEADS = 32;
    private static final int DEFAULT_HEAD_SIZE = 128;
    private static final int DEFAULT_CONTEXT_LENGTH = 2048;
    private static final int DEFAULT_KV_HEADS = 8;  // GQA: 8 KV heads for 32 query heads

    private static final int WARM_UP_ITERATIONS = 20;
    private static final int BENCHMARK_ITERATIONS = 50;

    // TODO: Update this after generating PTX kernel
    private static final String ENTRY_POINT = "processHeadsFlashAttention";

    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array) {
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, (RANDOM.nextFloat() - 0.5f) * 0.1f);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: FlashAttentionPTXBenchmark <kernel.ptx> [nHeads] [headSize] [contextLength]");
            System.out.println("  Defaults: nHeads=32, headSize=128, contextLength=2048");
            System.exit(1);
        }

        String kernelPath = args[0];
        int nHeads = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_N_HEADS;
        int headSize = (args.length >= 3) ? Integer.parseInt(args[2]) : DEFAULT_HEAD_SIZE;
        int contextLength = (args.length >= 4) ? Integer.parseInt(args[3]) : DEFAULT_CONTEXT_LENGTH;

        // Derived dimensions
        int nKvHeads = DEFAULT_KV_HEADS;
        int kvDim = nKvHeads * headSize;
        int kvMul = nHeads / nKvHeads;
        int numLayers = 32;  // Typical for Llama-3 8B
        int layer = 0;
        int position = contextLength - 1;  // Process at last position (worst case)

        System.out.println("=== Flash Attention PTX Benchmark ===");
        System.out.println("Kernel: " + kernelPath);
        System.out.println("nHeads: " + nHeads);
        System.out.println("headSize: " + headSize);
        System.out.println("contextLength: " + contextLength);
        System.out.println("kvDim: " + kvDim);
        System.out.println("kvMul: " + kvMul);
        System.out.println("position: " + position);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        // Allocate arrays
        FloatArray q = new FloatArray(nHeads * headSize);
        FloatArray keyCache = new FloatArray(numLayers * contextLength * kvDim);
        FloatArray valueCache = new FloatArray(numLayers * contextLength * kvDim);
        FloatArray xb = new FloatArray(nHeads * headSize);  // Output
        IntArray positionHolder = new IntArray(1);

        // Initialize data
        fillRandomData(q);
        fillRandomData(keyCache);
        fillRandomData(valueCache);
        positionHolder.set(0, position);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // Kernel signature: processHeadsFlashAttention(q, key_cache, value_cache, xb,
        //                   nHeads, headSize, kvDim, kvMul, positionHolder, layer, contextLength)
        AccessorParameters accessors = new AccessorParameters(11);
        accessors.set(0, q, Access.READ_ONLY);
        accessors.set(1, keyCache, Access.READ_ONLY);
        accessors.set(2, valueCache, Access.READ_ONLY);
        accessors.set(3, xb, Access.WRITE_ONLY);
        accessors.set(4, Integer.valueOf(nHeads), Access.NONE);
        accessors.set(5, Integer.valueOf(headSize), Access.NONE);
        accessors.set(6, Integer.valueOf(kvDim), Access.NONE);
        accessors.set(7, Integer.valueOf(kvMul), Access.NONE);
        accessors.set(8, positionHolder, Access.READ_ONLY);
        accessors.set(9, Integer.valueOf(layer), Access.NONE);
        accessors.set(10, Integer.valueOf(contextLength), Access.NONE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, q, keyCache, valueCache, positionHolder)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, xb);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // Grid: nHeads workgroups, headSize local threads each
        WorkerGrid1D worker = new WorkerGrid1D(nHeads * headSize);
        worker.setLocalWork(headSize, 1, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        ArrayList<Long> kernelTimes = new ArrayList<>();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler);

            // Warmup
            System.out.println("Warming up...");
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                plan.execute();
            }

            // Measure kernel time only
            System.out.println("Measuring kernel time...");
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                TornadoExecutionResult result = plan
                        .withProfiler(ProfilerMode.SILENT)
                        .execute();

                TornadoProfilerResult profilerResult = result.getProfilerResult();
                long kernelTime = profilerResult.getDeviceKernelTime();
                kernelTimes.add(kernelTime);
            }
        }

        LongSummaryStatistics stats = kernelTimes.stream().mapToLong(Long::longValue).summaryStatistics();

        // Calculate metrics
        // FLOPs per attention: ~4 * nHeads * headSize * contextLength (approximate)
        long flopsPerIteration = 4L * nHeads * headSize * (position + 1);
        double gflops = (flopsPerIteration * 1e-9) / (stats.getAverage() * 1e-9);

        // Tokens per second (how many positions could be processed)
        double tokensPerSec = 1.0 / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY)");
        System.out.println("==========================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Configuration: %d heads x %d dim, context=%d%n", nHeads, headSize, contextLength);
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("GFLOP/s: %.2f%n", gflops);
        System.out.printf("Theoretical tokens/s: %.0f%n", tokensPerSec);
    }
}
