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
 * PTX Single-kernel benchmark for Pi Computation (Leibniz series reduction).
 * Measures ONLY kernel execution time - run once per kernel for fair comparison.
 *
 * Usage: java ... PiComputationPTXBenchmark <kernel.ptx> [size]
 *
 * Default size: 16777216 (16M terms)
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
public class PiComputationPTXBenchmark {

    private static final int DEFAULT_SIZE = 16777216;  // 16M terms
    private static final int LOCAL_WORK_SIZE = 256;
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;

    private static final String ENTRY_POINT = "computePi";

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: PiComputationPTXBenchmark <kernel.ptx> [size]");
            System.out.println("  Default size: " + DEFAULT_SIZE);
            System.exit(1);
        }

        String kernelPath = args[0];
        int size = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_SIZE;

        System.out.println("=== Pi Computation PTX Benchmark ===");
        System.out.println("Kernel: " + kernelPath);
        System.out.println("Number of terms: " + size);
        System.out.println("Local work size: " + LOCAL_WORK_SIZE);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        FloatArray input = new FloatArray(size);
        FloatArray result = new FloatArray(1);

        // Initialize input (zeros for this computation - values computed in kernel)
        input.init(0.0f);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // Set up kernel parameters
        AccessorParameters accessors = new AccessorParameters(2);
        accessors.set(0, input, Access.READ_ONLY);
        accessors.set(1, result, Access.READ_WRITE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, result)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, result);

        ImmutableTaskGraph snapshot = graph.snapshot();

        WorkerGrid1D worker = new WorkerGrid1D(size);
        worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        ArrayList<Long> kernelTimes = new ArrayList<>();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler);

            // Warmup
            System.out.println("Warming up...");
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                result.init(0.0f);
                plan.execute();
            }

            // Measure kernel time only
            System.out.println("Measuring kernel time...");
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                result.init(0.0f);

                TornadoExecutionResult execResult = plan
                        .withProfiler(ProfilerMode.SILENT)
                        .execute();

                TornadoProfilerResult profilerResult = execResult.getProfilerResult();
                long kernelTime = profilerResult.getDeviceKernelTime();
                kernelTimes.add(kernelTime);
            }
        }

        LongSummaryStatistics stats = kernelTimes.stream().mapToLong(Long::longValue).summaryStatistics();

        // Calculate Pi from result
        float piValue = result.get(0) * 4.0f;

        // Throughput: terms processed per second
        double mtermsPerSec = (size * 1e-6) / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY)");
        System.out.println("==========================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Terms: %d%n", size);
        System.out.printf("Pi value: %.10f%n", piValue);
        System.out.printf("Pi error: %.2e%n", Math.abs(Math.PI - piValue));
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("Million Terms/s: %.2f%n", mtermsPerSec);
    }
}
