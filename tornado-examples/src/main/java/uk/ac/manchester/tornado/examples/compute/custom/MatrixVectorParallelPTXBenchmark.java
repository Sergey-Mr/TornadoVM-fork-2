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
import uk.ac.manchester.tornado.api.types.arrays.LongArray;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

/**
 * PTX Benchmark for matrixVectorParallel kernel (simpler version without context_unused).
 *
 * PTX function signature:
 * s1_t0_matrixvectorparallel_arrays_floatarray_arrays_floatarray_arrays_floatarray_8192_2048(
 *     kernel_context, x, hb, w, n, d)
 *
 * Usage: MatrixVectorParallelPTXBenchmark <kernel.ptx> [inputDim] [outputDim]
 */
public class MatrixVectorParallelPTXBenchmark {

    private static final int DEFAULT_INPUT_DIM = 8192;
    private static final int DEFAULT_OUTPUT_DIM = 2048;
    private static final int LOCAL_WORK_SIZE = 256;
    private static final int WARM_UP_ITERATIONS = 100;
    private static final int BENCHMARK_ITERATIONS = 200;

    // Entry point - TornadoVM will construct: s1_t0_matrixvectorparallel_...
    private static final String ENTRY_POINT = "matrixVectorParallel";

    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array, float min, float max) {
        float range = max - min;
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, min + RANDOM.nextFloat() * range);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: MatrixVectorParallelPTXBenchmark <kernel.ptx> [inputDim] [outputDim]");
            System.out.println("  Default inputDim: " + DEFAULT_INPUT_DIM);
            System.out.println("  Default outputDim: " + DEFAULT_OUTPUT_DIM);
            System.exit(1);
        }

        String kernelPath = args[0];
        int inputDim = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_INPUT_DIM;
        int outputDim = (args.length >= 3) ? Integer.parseInt(args[2]) : DEFAULT_OUTPUT_DIM;

        System.out.println("=== Matrix-Vector Parallel PTX Benchmark ===");
        System.out.println("Kernel: " + kernelPath);
        System.out.println("Input dimension: " + inputDim);
        System.out.println("Output dimension: " + outputDim);
        System.out.println("Local work size: " + LOCAL_WORK_SIZE);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        // Data arrays
        FloatArray input = new FloatArray(inputDim);
        FloatArray weights = new FloatArray(inputDim * outputDim);
        FloatArray output = new FloatArray(outputDim);

        // KernelContext placeholder
        LongArray kernelContext = new LongArray(64);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        fillRandomData(input, -1.0f, 1.0f);
        fillRandomData(weights, -0.1f, 0.1f);

        // Parameters match PTX signature: (kernel_context, x, hb, w, n, d)
        AccessorParameters accessors = new AccessorParameters(6);
        accessors.set(0, kernelContext, Access.READ_ONLY);      // kernel_context
        accessors.set(1, input, Access.READ_ONLY);              // x
        accessors.set(2, output, Access.WRITE_ONLY);            // hb
        accessors.set(3, weights, Access.READ_ONLY);            // w
        accessors.set(4, Long.valueOf(inputDim), Access.NONE);  // n
        accessors.set(5, Long.valueOf(outputDim), Access.NONE); // d

        // Use "s1" to match PTX naming: s1_t0_matrixvectorparallel_...
        TaskGraph graph = new TaskGraph("s1")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, kernelContext, input, weights)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();

        WorkerGrid1D worker = new WorkerGrid1D(outputDim);
        worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
        GridScheduler scheduler = new GridScheduler("s1.t0", worker);

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
        long totalFlops = 2L * inputDim * outputDim;
        double gflops = (totalFlops * 1e-9) / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY)");
        System.out.println("==========================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Dimensions: %d x %d%n", inputDim, outputDim);
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("GFLOP/s: %.2f%n", gflops);

        // Print sample output values
        System.out.println();
        System.out.println("Sample output values:");
        System.out.printf("  output[0]: %.6f%n", output.get(0));
        System.out.printf("  output[%d]: %.6f%n", outputDim/2, output.get(outputDim/2));
        System.out.printf("  output[%d]: %.6f%n", outputDim-1, output.get(outputDim-1));
    }
}
