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
 * Single-kernel benchmark for Black-Scholes option pricing.
 * Measures ONLY kernel execution time - run once per kernel for fair comparison.
 *
 * Usage: java ... BlackScholesSingleKernelBenchmark <kernel.cl> [numOptions]
 *
 * Default numOptions: 4194304 (4M)
 */
public class BlackScholesSingleKernelBenchmark {

    private static final int DEFAULT_NUM_OPTIONS = 4194304; // 4M options
    private static final int LOCAL_WORK_SIZE = 256;
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;
    private static final String ENTRY_POINT = "blackScholesKernel";
    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array) {
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, RANDOM.nextFloat());
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: BlackScholesSingleKernelBenchmark <kernel.cl> [numOptions]");
            System.exit(1);
        }

        String kernelPath = args[0];
        int numOptions = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_NUM_OPTIONS;

        System.out.println("Kernel: " + kernelPath);
        System.out.println("Number of options: " + numOptions);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        // Input: random values for generating option parameters
        // Output: call and put option prices
        FloatArray input = new FloatArray(numOptions);
        FloatArray callResult = new FloatArray(numOptions);
        FloatArray putResult = new FloatArray(numOptions);

        fillRandomData(input);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // Kernel signature: blackScholesKernel(input, callResult, putResult)
        AccessorParameters accessors = new AccessorParameters(3);
        accessors.set(0, input, Access.READ_ONLY);
        accessors.set(1, callResult, Access.WRITE_ONLY);
        accessors.set(2, putResult, Access.WRITE_ONLY);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, callResult, putResult);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // 1D grid: one thread per option
        WorkerGrid1D worker = new WorkerGrid1D(numOptions);
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

        LongSummaryStatistics stats = kernelTimes.stream().mapToLong(Long::longValue).summaryStatistics();

        // Million options per second metric
        double millionOptionsPerSec = (numOptions * 1e-6) / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY)");
        System.out.println("==========================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Number of options: %d%n", numOptions);
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("MOptions/s: %.2f%n", millionOptionsPerSec);
    }
}
