package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;

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
 * PTX benchmark for Monte Carlo PI estimation.
 * Measures ONLY kernel execution time - run once per kernel for fair comparison.
 *
 * Usage: java ... MonteCarloPTXBenchmark <kernel.ptx> [numSamples]
 *
 * Default numSamples: 16777216 (16M)
 */
public class MonteCarloPTXBenchmark {

    private static final int DEFAULT_NUM_SAMPLES = 16777216; // 16M samples
    private static final int LOCAL_WORK_SIZE = 256;
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;

    // PTX entry point from TornadoVM-generated kernel
    private static final String ENTRY_POINT = "s0_taskgraph_computemontecarlo_arrays_floatarray_16777216";

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: MonteCarloPTXBenchmark <kernel.ptx> [numSamples]");
            System.exit(1);
        }

        String kernelPath = args[0];
        int numSamples = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_NUM_SAMPLES;

        System.out.println("Kernel: " + kernelPath);
        System.out.println("Entry point: " + ENTRY_POINT);
        System.out.println("Number of samples: " + numSamples);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        // Output array: one float per sample (1.0 if inside circle, 0.0 if outside)
        FloatArray output = new FloatArray(numSamples);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // PTX kernel signature: (kernel_context, output, iterations)
        // But AccessorParameters only needs user data: output, iterations
        AccessorParameters accessors = new AccessorParameters(2);
        accessors.set(0, output, Access.WRITE_ONLY);
        accessors.set(1, Long.valueOf(numSamples), Access.NONE);  // PTX uses .u64 for iterations

        TaskGraph graph = new TaskGraph("s0")
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // 1D grid: one thread per sample
        WorkerGrid1D worker = new WorkerGrid1D(numSamples);
        worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
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

        // Calculate PI from last run's output
        float sum = 0.0f;
        for (int i = 0; i < numSamples; i++) {
            sum += output.get(i);
        }
        double piEstimate = (4.0 * sum) / numSamples;

        LongSummaryStatistics stats = kernelTimes.stream().mapToLong(Long::longValue).summaryStatistics();

        // Million samples per second metric
        double millionSamplesPerSec = (numSamples * 1e-6) / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY)");
        System.out.println("==========================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Number of samples: %d%n", numSamples);
        System.out.printf("PI estimate: %.8f (error: %.2e)%n", piEstimate, Math.abs(Math.PI - piEstimate));
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("MSamples/s: %.2f%n", millionSamplesPerSec);
    }
}
