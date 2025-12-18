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
import uk.ac.manchester.tornado.examples.compute.MatrixVectorRowMajor;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

public class MatrixVectorRowCustomBenchmark {

    private static final int INPUT_DIM = 8192;
    private static final int OUTPUT_DIM = 2048;
    private static final int LOCAL_WORK_GROUP_SIZE = 128;
    private static final int WARM_UP_ITERATIONS = 60;
    private static final int BENCHMARK_ITERATIONS = 120;
    private static final int GPU_STABILIZATION_ITERATIONS = 100; // Extra warmup to stabilize GPU state before first measurement

    private static final String DEFAULT_GENERATED_KERNEL = "kernels/matrixvector_generated.cl";
    private static final String DEFAULT_CUSTOM_KERNEL = "kernels/matrixvector_custom.cl";
    private static final String ENTRY_POINT = "matrixVectorGeneric";

    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array, float min, float max) {
        float range = max - min;
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, min + RANDOM.nextFloat() * range);
        }
    }

    private static boolean validate(FloatArray reference, FloatArray candidate, float delta) {
        boolean valid = true;
        for (int i = 0; i < reference.getSize(); i++) {
            float diff = Math.abs(reference.get(i) - candidate.get(i));
            if (diff > delta) {
                System.out.printf("Mismatch at %d: ref=%.6f cand=%.6f diff=%.6f%n", i, reference.get(i), candidate.get(i), diff);
                valid = false;
                break;
            }
        }
        return valid;
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        String generatedKernelPath = (args.length >= 1) ? args[0] : DEFAULT_GENERATED_KERNEL;
        String customKernelPath = (args.length >= 2) ? args[1] : DEFAULT_CUSTOM_KERNEL;

        System.out.println("Generated kernel: " + generatedKernelPath);
        System.out.println("Custom kernel   : " + customKernelPath);
        FloatArray input = new FloatArray(INPUT_DIM);
        FloatArray weights = new FloatArray(INPUT_DIM * OUTPUT_DIM);
        FloatArray outputSequential = new FloatArray(OUTPUT_DIM);
        FloatArray outputKernelContext = new FloatArray(OUTPUT_DIM);
        FloatArray outputCustom = new FloatArray(OUTPUT_DIM);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

        fillRandomData(input, -1.0f, 1.0f);
        fillRandomData(weights, -0.1f, 0.1f);

        // Sequential baseline
        for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
            MatrixVectorRowMajor.matrixVectorSequential(input, outputSequential, weights, INPUT_DIM, OUTPUT_DIM);
        }

        ArrayList<Long> sequentialTimes = new ArrayList<>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            MatrixVectorRowMajor.matrixVectorSequential(input, outputSequential, weights, INPUT_DIM, OUTPUT_DIM);
            long end = System.nanoTime();
            sequentialTimes.add(end - start);
        }

        // Fair comparison: set up both plans, interleave warmup and measurement
        FairBenchmarkResult fairResult = benchmarkKernelsFairly(
                generatedKernelPath, customKernelPath,
                input, weights, outputKernelContext, outputCustom, device);

        boolean generatedValid = validate(outputSequential, outputKernelContext, 1e-4f);
        boolean customValid = validate(outputSequential, outputCustom, 1e-4f);

        if (!generatedValid) {
            System.out.println("Generated kernel result does not match sequential baseline.");
        }
        if (!customValid) {
            System.out.println("Custom kernel result does not match sequential baseline.");
        }

        LongSummaryStatistics seqStats = sequentialTimes.stream().mapToLong(Long::longValue).summaryStatistics();
        LongSummaryStatistics generatedStats = fairResult.generatedStats();
        LongSummaryStatistics customStats = fairResult.customStats();

        long totalFlops = 2L * INPUT_DIM * OUTPUT_DIM;

        double seqGFlops = (totalFlops * 1e-9) / (seqStats.getAverage() * 1e-9);
        double generatedGFlops = (totalFlops * 1e-9) / (generatedStats.getAverage() * 1e-9);
        double customGFlops = (totalFlops * 1e-9) / (customStats.getAverage() * 1e-9);

        System.out.println("Benchmark Results (problem-specific)");
        System.out.println("====================================");
        System.out.printf("Sequential: avg=%.3f ms min=%.3f ms max=%.3f ms GFLOP/s=%.2f%n",
                seqStats.getAverage() / 1_000_000.0,
                seqStats.getMin() / 1_000_000.0,
                seqStats.getMax() / 1_000_000.0,
                seqGFlops);

        System.out.printf("Generated OpenCL: avg=%.3f ms min=%.3f ms max=%.3f ms GFLOP/s=%.2f%n",
                generatedStats.getAverage() / 1_000_000.0,
                generatedStats.getMin() / 1_000_000.0,
                generatedStats.getMax() / 1_000_000.0,
                generatedGFlops);

        System.out.printf("Custom OpenCL: avg=%.3f ms min=%.3f ms max=%.3f ms GFLOP/s=%.2f%n",
                customStats.getAverage() / 1_000_000.0,
                customStats.getMin() / 1_000_000.0,
                customStats.getMax() / 1_000_000.0,
                customGFlops);

        System.out.printf("Speedup (Generated vs Java): %.2fx%n", seqStats.getAverage() / generatedStats.getAverage());
        System.out.printf("Speedup (Custom vs Java): %.2fx%n", seqStats.getAverage() / customStats.getAverage());
        System.out.printf("Speedup (Custom vs Generated): %.2fx%n", generatedStats.getAverage() / customStats.getAverage());

        if (customStats.getAverage() < generatedStats.getAverage()) {
            System.out.println("Custom kernel is faster by " +
                    String.format("%.2fx", generatedStats.getAverage() / customStats.getAverage()) +
                    " over the generated kernel.");
        } else {
            System.out.println("Generated kernel remains faster by " +
                    String.format("%.2fx", customStats.getAverage() / generatedStats.getAverage()) +
                    " over the custom kernel.");
        }
    }

    /**
     * Sequential benchmark: creates both execution plans, performs complete warmup and measurement
     * for each kernel sequentially to reduce context switches.
     */
    /**
     * Sequential benchmark using TornadoVM profiler to measure kernel execution time only.
     * This eliminates overhead from data transfers and other host-side operations.
     */
    private static FairBenchmarkResult benchmarkKernelsFairly(
            String generatedKernelPath, String customKernelPath,
            FloatArray input, FloatArray weights,
            FloatArray outputGenerated, FloatArray outputCustom,
            TornadoDevice device) throws TornadoExecutionPlanException {

        // Set up generated kernel plan
        AccessorParameters accessorsGenerated = new AccessorParameters(6);
        accessorsGenerated.set(0, input, Access.READ_ONLY);
        accessorsGenerated.set(1, outputGenerated, Access.WRITE_ONLY);
        accessorsGenerated.set(2, weights, Access.READ_ONLY);
        accessorsGenerated.set(3, Integer.valueOf(INPUT_DIM), Access.NONE);
        accessorsGenerated.set(4, Integer.valueOf(OUTPUT_DIM), Access.NONE);
        accessorsGenerated.set(5, Integer.valueOf(LOCAL_WORK_GROUP_SIZE), Access.NONE);

        TaskGraph graphGenerated = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input, weights)
                .prebuiltTask("t0", ENTRY_POINT, generatedKernelPath, accessorsGenerated)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputGenerated);

        ImmutableTaskGraph snapshotGenerated = graphGenerated.snapshot();

        WorkerGrid1D workerGenerated = new WorkerGrid1D(OUTPUT_DIM * LOCAL_WORK_GROUP_SIZE);
        workerGenerated.setLocalWork(LOCAL_WORK_GROUP_SIZE, 1, 1);
        GridScheduler schedulerGenerated = new GridScheduler("s0.t0", workerGenerated);

        // Set up custom kernel plan
        AccessorParameters accessorsCustom = new AccessorParameters(6);
        accessorsCustom.set(0, input, Access.READ_ONLY);
        accessorsCustom.set(1, outputCustom, Access.WRITE_ONLY);
        accessorsCustom.set(2, weights, Access.READ_ONLY);
        accessorsCustom.set(3, Integer.valueOf(INPUT_DIM), Access.NONE);
        accessorsCustom.set(4, Integer.valueOf(OUTPUT_DIM), Access.NONE);
        accessorsCustom.set(5, Integer.valueOf(LOCAL_WORK_GROUP_SIZE), Access.NONE);

        TaskGraph graphCustom = new TaskGraph("s1")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input, weights)
                .prebuiltTask("t0", ENTRY_POINT, customKernelPath, accessorsCustom)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputCustom);

        ImmutableTaskGraph snapshotCustom = graphCustom.snapshot();

        WorkerGrid1D workerCustom = new WorkerGrid1D(OUTPUT_DIM * LOCAL_WORK_GROUP_SIZE);
        workerCustom.setLocalWork(LOCAL_WORK_GROUP_SIZE, 1, 1);
        GridScheduler schedulerCustom = new GridScheduler("s1.t0", workerCustom);

        ArrayList<Long> generatedKernelTimes = new ArrayList<>();
        ArrayList<Long> customKernelTimes = new ArrayList<>();

        try (TornadoExecutionPlan planGenerated = new TornadoExecutionPlan(snapshotGenerated);
             TornadoExecutionPlan planCustom = new TornadoExecutionPlan(snapshotCustom)) {

            planGenerated.withDevice(device).withGridScheduler(schedulerGenerated);
            planCustom.withDevice(device).withGridScheduler(schedulerCustom);

            // Sequential approach: Complete warmup and measurement for kernel 1, then kernel 2

            // GPU stabilization phase - ensure GPU is in stable boosted state before any measurement
            System.out.println("Stabilizing GPU state...");
            for (int i = 0; i < GPU_STABILIZATION_ITERATIONS; i++) {
                planGenerated.execute();
            }

            System.out.println("Warming up and measuring generated kernel (kernel time only)...");

            // Warmup generated kernel
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                planGenerated.execute();
            }
        
            // Measure generated kernel using profiler
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                TornadoExecutionResult result = planGenerated
                        .withProfiler(ProfilerMode.SILENT)
                        .execute();
            
                TornadoProfilerResult profilerResult = result.getProfilerResult();
                long kernelTime = profilerResult.getDeviceKernelTime();
                generatedKernelTimes.add(kernelTime);
            }

            System.out.println("Warming up and measuring custom kernel (kernel time only)...");
        
            // Warmup custom kernel
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                planCustom.execute();
            }
        
            // Measure custom kernel using profiler
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                TornadoExecutionResult result = planCustom
                        .withProfiler(ProfilerMode.SILENT)
                        .execute();
            
                TornadoProfilerResult profilerResult = result.getProfilerResult();
                long kernelTime = profilerResult.getDeviceKernelTime();
                customKernelTimes.add(kernelTime);
            }
        }

        return new FairBenchmarkResult(
                generatedKernelTimes.stream().mapToLong(Long::longValue).summaryStatistics(),
                customKernelTimes.stream().mapToLong(Long::longValue).summaryStatistics()
        );
    }

    private record FairBenchmarkResult(LongSummaryStatistics generatedStats, LongSummaryStatistics customStats) {
    }
}