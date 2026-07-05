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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

/**
 * Single-kernel benchmark for Blur Filter (2D convolution).
 * Measures ONLY kernel execution time - run once per kernel for fair comparison.
 *
 * Usage: java ... BlurFilterSingleKernelBenchmark <kernel.cl> [imageSize] [filterWidth]
 *
 * Default imageSize: 2048x2048
 * Default filterWidth: 31
 */
public class BlurFilterSingleKernelBenchmark {

    private static final int DEFAULT_IMAGE_SIZE = 2048;
    private static final int DEFAULT_FILTER_WIDTH = 31;
    private static final int LOCAL_WORK_SIZE = 16;
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;
    private static final String ENTRY_POINT = "compute";
    private static final Random RANDOM = new Random(42);

    private static void fillRandomImage(IntArray image) {
        for (int i = 0; i < image.getSize(); i++) {
            image.set(i, RANDOM.nextInt(256)); // Grayscale 0-255
        }
    }

    private static void createGaussianFilter(FloatArray filter, int filterWidth) {
        // Simple box filter (uniform weights)
        float weight = 1.0f / (filterWidth * filterWidth);
        for (int i = 0; i < filterWidth * filterWidth; i++) {
            filter.set(i, weight);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: BlurFilterSingleKernelBenchmark <kernel.cl> [imageSize] [filterWidth]");
            System.exit(1);
        }

        String kernelPath = args[0];
        int imageSize = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_IMAGE_SIZE;
        int filterWidth = (args.length >= 3) ? Integer.parseInt(args[2]) : DEFAULT_FILTER_WIDTH;

        int numRows = imageSize;
        int numCols = imageSize;

        System.out.println("Kernel: " + kernelPath);
        System.out.println("Image size: " + numRows + "x" + numCols);
        System.out.println("Filter width: " + filterWidth + "x" + filterWidth);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        // Input: single channel image (grayscale)
        IntArray inputChannel = new IntArray(numRows * numCols);
        IntArray outputChannel = new IntArray(numRows * numCols);
        FloatArray filter = new FloatArray(filterWidth * filterWidth);

        fillRandomImage(inputChannel);
        createGaussianFilter(filter, filterWidth);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // Kernel signature: compute(rgbChannel, channelBlurred, numRows, numCols, filter, filterWidth)
        AccessorParameters accessors = new AccessorParameters(6);
        accessors.set(0, inputChannel, Access.READ_ONLY);
        accessors.set(1, outputChannel, Access.WRITE_ONLY);
        accessors.set(2, Integer.valueOf(numRows), Access.NONE);
        accessors.set(3, Integer.valueOf(numCols), Access.NONE);
        accessors.set(4, filter, Access.READ_ONLY);
        accessors.set(5, Integer.valueOf(filterWidth), Access.NONE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, inputChannel, filter)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputChannel);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // 2D grid: one thread per pixel
        WorkerGrid2D worker = new WorkerGrid2D(numRows, numCols);
        worker.setLocalWork(LOCAL_WORK_SIZE, LOCAL_WORK_SIZE, 1);
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

        // Pixels per second metric
        long totalPixels = (long) numRows * numCols;
        double megaPixelsPerSec = (totalPixels * 1e-6) / (stats.getAverage() * 1e-9);

        // FLOPS: each pixel requires filterWidth*filterWidth multiply-adds (2 ops each)
        long opsPerPixel = 2L * filterWidth * filterWidth;
        long totalOps = totalPixels * opsPerPixel;
        double gflops = (totalOps * 1e-9) / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY)");
        System.out.println("==========================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Image size: %dx%d (%d pixels)%n", numRows, numCols, totalPixels);
        System.out.printf("Filter size: %dx%d%n", filterWidth, filterWidth);
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("MPixels/s: %.2f%n", megaPixelsPerSec);
        System.out.printf("GFLOP/s: %.2f%n", gflops);
    }
}
