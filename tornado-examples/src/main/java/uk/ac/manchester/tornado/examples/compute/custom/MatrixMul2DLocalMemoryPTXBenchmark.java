package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;
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
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

/**
 * PTX Single-kernel benchmark for Matrix Multiplication with Local Memory (2D tiled).
 * Measures ONLY kernel execution time - run once per kernel for fair comparison.
 *
 * Usage: java ... MatrixMul2DLocalMemoryPTXBenchmark <kernel.ptx> [size]
 *
 * Default size: 1024x1024
 * Tile size (TS): 16 or 32 depending on kernel
 *
 * IMPORTANT - PTX File Preparation:
 * TornadoVM's prebuiltTask() automatically adds PTX headers (.version, .target, .address_size).
 * If your PTX file already contains these headers, you MUST strip them first:
 *
 *   sed -i '/^\.version/d; /^\.target/d; /^\.address_size/d' kernel.ptx
 *
 * Otherwise you'll get: cuModuleLoadData -> Returned: 218 (CUDA_ERROR_INVALID_PTX)
 *
 * NOTE: Update ENTRY_POINT to match your generated PTX kernel function name.
 */
public class MatrixMul2DLocalMemoryPTXBenchmark {

    private static final int DEFAULT_SIZE = 1024;
    private static final int TS = 16;  // Tile size - must match kernel
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;

    // TODO: Update this after generating PTX kernel
    private static final String ENTRY_POINT = "matrixMultiplication";

    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array, float min, float max) {
        float range = max - min;
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, min + RANDOM.nextFloat() * range);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: MatrixMul2DLocalMemoryPTXBenchmark <kernel.ptx> [size]");
            System.out.println("  size must be divisible by tile size (TS=" + TS + ")");
            System.out.println("  Default size: " + DEFAULT_SIZE);
            System.exit(1);
        }

        String kernelPath = args[0];
        int size = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_SIZE;

        if (size % TS != 0) {
            System.err.println("Error: size (" + size + ") must be divisible by tile size (" + TS + ")");
            System.exit(1);
        }

        System.out.println("=== Matrix Multiplication 2D Local Memory PTX Benchmark ===");
        System.out.println("Kernel: " + kernelPath);
        System.out.println("Matrix size: " + size + "x" + size);
        System.out.println("Tile size: " + TS);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        FloatArray matrixA = new FloatArray(size * size);
        FloatArray matrixB = new FloatArray(size * size);
        FloatArray matrixC = new FloatArray(size * size);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        fillRandomData(matrixA, -1.0f, 1.0f);
        fillRandomData(matrixB, -1.0f, 1.0f);

        // Set up kernel parameters
        AccessorParameters accessors = new AccessorParameters(4);
        accessors.set(0, matrixA, Access.READ_ONLY);
        accessors.set(1, matrixB, Access.READ_ONLY);
        accessors.set(2, matrixC, Access.WRITE_ONLY);
        accessors.set(3, Integer.valueOf(size), Access.NONE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA, matrixB)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, matrixC);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // 2D grid: global work = (size, size), local work = (TS, TS)
        WorkerGrid2D worker = new WorkerGrid2D(size, size);
        worker.setLocalWork(TS, TS, 1);
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

        // FLOPS for matrix multiplication: 2 * N^3
        long totalFlops = 2L * size * size * size;
        double gflops = (totalFlops * 1e-9) / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY)");
        System.out.println("==========================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Matrix size: %dx%d%n", size, size);
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("GFLOP/s: %.2f%n", gflops);
    }
}
