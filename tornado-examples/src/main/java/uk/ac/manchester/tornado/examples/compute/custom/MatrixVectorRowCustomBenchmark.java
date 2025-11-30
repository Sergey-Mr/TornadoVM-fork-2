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

public class MatrixVectorRowCustomBenchmark {

    private static final int INPUT_DIM = 8192;
    private static final int OUTPUT_DIM = 2048;
    private static final int LOCAL_WORK_GROUP_SIZE = 128;
    private static final int WARM_UP_ITERATIONS = 60;
    private static final int BENCHMARK_ITERATIONS = 120;

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

        BenchmarkResult generatedResult = benchmarkPrebuiltKernel("s0", generatedKernelPath, input, weights, outputKernelContext, device);
        BenchmarkResult customResult = benchmarkPrebuiltKernel("s1", customKernelPath, input, weights, outputCustom, device);

        boolean generatedValid = validate(outputSequential, outputKernelContext, 1e-4f);
        boolean customValid = validate(outputSequential, outputCustom, 1e-4f);

        if (!generatedValid) {
            System.out.println("Generated kernel result does not match sequential baseline.");
        }
        if (!customValid) {
            System.out.println("Custom kernel result does not match sequential baseline.");
        }

        LongSummaryStatistics seqStats = sequentialTimes.stream().mapToLong(Long::longValue).summaryStatistics();
        LongSummaryStatistics generatedStats = generatedResult.timingStats();
        LongSummaryStatistics customStats = customResult.timingStats();

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

    private static BenchmarkResult benchmarkPrebuiltKernel(String graphName, String kernelPath, FloatArray input, FloatArray weights, FloatArray output, TornadoDevice device)
            throws TornadoExecutionPlanException {
        AccessorParameters accessors = new AccessorParameters(6);
        accessors.set(0, input, Access.READ_ONLY);
        accessors.set(1, output, Access.WRITE_ONLY);
        accessors.set(2, weights, Access.READ_ONLY);
        accessors.set(3, Integer.valueOf(INPUT_DIM), Access.NONE);
        accessors.set(4, Integer.valueOf(OUTPUT_DIM), Access.NONE);
        accessors.set(5, Integer.valueOf(LOCAL_WORK_GROUP_SIZE), Access.NONE);

        TaskGraph graph = new TaskGraph(graphName)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input, weights)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();

        WorkerGrid1D worker = new WorkerGrid1D(OUTPUT_DIM * LOCAL_WORK_GROUP_SIZE);
        worker.setLocalWork(LOCAL_WORK_GROUP_SIZE, 1, 1);
        GridScheduler scheduler = new GridScheduler(graphName + ".t0", worker);

        ArrayList<Long> times = new ArrayList<>();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device);
            plan.withGridScheduler(scheduler);
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                plan.execute();
            }
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                long start = System.nanoTime();
                plan.execute();
                long end = System.nanoTime();
                times.add(end - start);
            }
        }

        return new BenchmarkResult(times.stream().mapToLong(Long::longValue).summaryStatistics());
    }

    private record BenchmarkResult(LongSummaryStatistics timingStats) {
    }
}
