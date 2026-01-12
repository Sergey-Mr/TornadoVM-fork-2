package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;

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
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

/**
 * Single-kernel benchmark for Mandelbrot set computation.
 * Measures ONLY kernel execution time - run once per kernel for fair comparison.
 *
 * Usage: java ... MandelbrotSingleKernelBenchmark <kernel.cl> [size]
 *
 * Default size: 1024x1024
 */
public class MandelbrotSingleKernelBenchmark {

    private static final int DEFAULT_SIZE = 1024;
    private static final int LOCAL_WORK_SIZE = 16;
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;
    private static final String ENTRY_POINT = "mandelbrotTornado";

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: MandelbrotSingleKernelBenchmark <kernel.cl> [size]");
            System.exit(1);
        }

        String kernelPath = args[0];
        int size = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_SIZE;

        System.out.println("Kernel: " + kernelPath);
        System.out.println("Image size: " + size + "x" + size);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        // Output array: size * size shorts (iteration counts as colors)
        ShortArray output = new ShortArray(size * size);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // Kernel signature: mandelbrotTornado(size, output)
        AccessorParameters accessors = new AccessorParameters(2);
        accessors.set(0, Integer.valueOf(size), Access.NONE);
        accessors.set(1, output, Access.WRITE_ONLY);

        TaskGraph graph = new TaskGraph("s0")
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // 2D grid: one thread per pixel
        WorkerGrid2D worker = new WorkerGrid2D(size, size);
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
        long totalPixels = (long) size * size;
        double megaPixelsPerSec = (totalPixels * 1e-6) / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY)");
        System.out.println("==========================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Image size: %dx%d (%d pixels)%n", size, size, totalPixels);
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("MPixels/s: %.2f%n", megaPixelsPerSec);
    }
}
