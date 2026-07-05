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
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

/**
 * Single-kernel benchmark for NBody simulation.
 * Measures ONLY kernel execution time - run once per kernel for fair comparison.
 *
 * Usage: java ... NBodySingleKernelBenchmark <kernel.cl> [numBodies]
 *
 * Default numBodies: 16384
 */
public class NBodySingleKernelBenchmark {

    private static final int DEFAULT_NUM_BODIES = 16384;
    private static final int LOCAL_WORK_SIZE = 256;
    private static final int WARM_UP_ITERATIONS = 20;
    private static final int BENCHMARK_ITERATIONS = 50;
    private static final String ENTRY_POINT = "nBody";

    // NBody simulation parameters
    private static final float DEL_T = 0.005f;
    private static final float ESP_SQR = 500.0f;

    private static final Random RANDOM = new Random(42);

    private static void initializeBodies(FloatArray pos, FloatArray vel, int numBodies) {
        // Position: x, y, z, mass (4 floats per body)
        for (int i = 0; i < numBodies; i++) {
            pos.set(4 * i + 0, RANDOM.nextFloat() * 2.0f - 1.0f);     // x
            pos.set(4 * i + 1, RANDOM.nextFloat() * 2.0f - 1.0f);     // y
            pos.set(4 * i + 2, RANDOM.nextFloat() * 2.0f - 1.0f);     // z
            pos.set(4 * i + 3, RANDOM.nextFloat() * 0.5f + 0.5f);     // mass
        }
        // Velocity: vx, vy, vz, padding (4 floats per body)
        for (int i = 0; i < numBodies * 4; i++) {
            vel.set(i, 0.0f);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: NBodySingleKernelBenchmark <kernel.cl> [numBodies]");
            System.out.println("  Default numBodies: " + DEFAULT_NUM_BODIES);
            System.exit(1);
        }

        String kernelPath = args[0];
        int numBodies = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_NUM_BODIES;

        System.out.println("Kernel: " + kernelPath);
        System.out.println("Number of bodies: " + numBodies);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        // Allocate arrays: 4 floats per body (x, y, z, mass/padding)
        FloatArray pos = new FloatArray(numBodies * 4);
        FloatArray vel = new FloatArray(numBodies * 4);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        initializeBodies(pos, vel, numBodies);

        // Set up kernel parameters
        // Kernel signature: nBody(numBodies, pos, vel, delT, espSqr)
        AccessorParameters accessors = new AccessorParameters(5);
        accessors.set(0, Integer.valueOf(numBodies), Access.NONE);
        accessors.set(1, pos, Access.READ_WRITE);
        accessors.set(2, vel, Access.READ_WRITE);
        accessors.set(3, Float.valueOf(DEL_T), Access.NONE);
        accessors.set(4, Float.valueOf(ESP_SQR), Access.NONE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, pos, vel)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, pos, vel);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // 1D grid: one thread per body
        WorkerGrid1D worker = new WorkerGrid1D(numBodies);
        worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        ArrayList<Long> kernelTimes = new ArrayList<>();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler);

            // Warmup (includes compilation)
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

        // FLOPS for NBody: approximately 20 FLOPS per pair interaction
        // Total: numBodies * numBodies * 20
        long totalFlops = 20L * numBodies * numBodies;
        double gflops = (totalFlops * 1e-9) / (stats.getAverage() * 1e-9);

        // Interactions per second
        long interactions = (long) numBodies * numBodies;
        double billionInteractions = (interactions * 1e-9) / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY)");
        System.out.println("==========================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Bodies: %d%n", numBodies);
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("GFLOP/s: %.2f%n", gflops);
        System.out.printf("Billion Interactions/s: %.2f%n", billionInteractions);
    }
}
