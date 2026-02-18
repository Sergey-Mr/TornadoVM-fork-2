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
 * PTX Single-kernel benchmark for Matrix-Vector Row Major multiplication.
 * This version handles KernelContext-based PTX kernels which have extra parameters.
 *
 * Usage: java ... MatrixVectorRowMajorKernelContextPTXBenchmark <kernel.ptx> [inputDim] [outputDim]
 *
 * IMPORTANT - PTX File Preparation:
 * TornadoVM's prebuiltTask() automatically adds PTX headers (.version, .target, .address_size).
 * If your PTX file already contains these headers, you MUST strip them first:
 *
 *   sed -i '/^\.version/d; /^\.target/d; /^\.address_size/d' kernel.ptx
 *
 * Otherwise you'll get: cuModuleLoadData -> Returned: 218 (CUDA_ERROR_INVALID_PTX)
 *
 * NOTE: The generated PTX for KernelContext-based kernels has this signature:
 *   s0_t0_matrixvectorgeneric_uk_ac_manchester_tornado_api_kernelcontext_...(
 *       kernel_context, context_unused, x, hb, w, n, d, localWorkGroupSize)
 *
 * The first two parameters (kernel_context, context_unused) are TornadoVM internal pointers.
 */
public class MatrixVectorRowMajorKernelContextPTXBenchmark {

    private static final int DEFAULT_INPUT_DIM = 8192;
    private static final int DEFAULT_OUTPUT_DIM = 2048;
    private static final int LOCAL_WORK_GROUP_SIZE = 32;  // Must match PTX kernel naming
    private static final int WARM_UP_ITERATIONS = 100;
    private static final int BENCHMARK_ITERATIONS = 200;

    // Entry point must match the base function name (TornadoVM adds prefixes)
    private static final String ENTRY_POINT = "matrixVectorGeneric";

    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array, float min, float max) {
        float range = max - min;
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, min + RANDOM.nextFloat() * range);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: MatrixVectorRowMajorKernelContextPTXBenchmark <kernel.ptx> [inputDim] [outputDim]");
            System.out.println("  Default inputDim: " + DEFAULT_INPUT_DIM);
            System.out.println("  Default outputDim: " + DEFAULT_OUTPUT_DIM);
            System.out.println();
            System.out.println("NOTE: This benchmark is for KernelContext-based PTX kernels.");
            System.out.println("The PTX function signature includes kernel_context and context_unused parameters.");
            System.exit(1);
        }

        String kernelPath = args[0];
        int inputDim = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_INPUT_DIM;
        int outputDim = (args.length >= 3) ? Integer.parseInt(args[2]) : DEFAULT_OUTPUT_DIM;

        System.out.println("=== Matrix-Vector Row Major KernelContext PTX Benchmark ===");
        System.out.println("Kernel: " + kernelPath);
        System.out.println("Input dimension: " + inputDim);
        System.out.println("Output dimension: " + outputDim);
        System.out.println("Local work group size: " + LOCAL_WORK_GROUP_SIZE);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        // User data arrays
        FloatArray input = new FloatArray(inputDim);
        FloatArray weights = new FloatArray(inputDim * outputDim);
        FloatArray output = new FloatArray(outputDim);

        // KernelContext placeholder arrays (TornadoVM internal)
        // These are dummy arrays that the KernelContext-based PTX expects
        LongArray kernelContext = new LongArray(64);  // Placeholder for kernel context
        LongArray contextUnused = new LongArray(64);  // Placeholder for unused context

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        fillRandomData(input, -1.0f, 1.0f);
        fillRandomData(weights, -0.1f, 0.1f);

        // Set up kernel parameters to match KernelContext PTX signature:
        // (kernel_context, context_unused, x, hb, w, n, d, localWorkGroupSize)
        AccessorParameters accessors = new AccessorParameters(8);
        accessors.set(0, kernelContext, Access.READ_ONLY);       // kernel_context
        accessors.set(1, contextUnused, Access.READ_ONLY);       // context_unused
        accessors.set(2, input, Access.READ_ONLY);               // x (input vector)
        accessors.set(3, output, Access.WRITE_ONLY);             // hb (output vector)
        accessors.set(4, weights, Access.READ_ONLY);             // w (weight matrix)
        accessors.set(5, Long.valueOf(inputDim), Access.NONE);   // n (input dimension)
        accessors.set(6, Long.valueOf(outputDim), Access.NONE);  // d (output dimension)
        accessors.set(7, Long.valueOf(LOCAL_WORK_GROUP_SIZE), Access.NONE); // localWorkGroupSize

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, kernelContext, contextUnused, input, weights)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // Global work size = outputDim * LOCAL_WORK_GROUP_SIZE (each output element gets a workgroup)
        WorkerGrid1D worker = new WorkerGrid1D(outputDim * LOCAL_WORK_GROUP_SIZE);
        worker.setLocalWork(LOCAL_WORK_GROUP_SIZE, 1, 1);
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

        // Print sample output values for verification
        System.out.println();
        System.out.println("Sample output values:");
        System.out.printf("  output[0]: %.6f%n", output.get(0));
        System.out.printf("  output[%d]: %.6f%n", outputDim/2, output.get(outputDim/2));
        System.out.printf("  output[%d]: %.6f%n", outputDim-1, output.get(outputDim-1));
    }
}
