package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;
import java.util.Random;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.TornadoRuntime;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Single-kernel benchmark for Matrix Multiplication 1D on the METAL backend.
 * Measures ONLY kernel execution time - run once per kernel in a fresh JVM for
 * a fair comparison, following the same methodology as the OpenCL/PTX variants.
 *
 * Usage: java ... MatrixMultiplication1DMetalSingleKernelBenchmark <kernel.metal> [size] [local] [--entry=NAME]
 *
 * The entry point defaults to the fully-qualified name TornadoVM emits for the
 * Metal backend; override with --entry= if your kernel uses a different name.
 */
public class MatrixMultiplication1DMetalSingleKernelBenchmark {

    private static final int DEFAULT_SIZE = 1024;
    private static final int DEFAULT_LOCAL = 16; // Metal supports up to 1024 threads/threadgroup
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;
    private static final String DEFAULT_ENTRY = "uk_ac_manchester_tornado_examples_compute_MatrixMultiplication1D_matrixMultiplication";
    private static final Random RANDOM = new Random(42);

    private static void fillRandomData(FloatArray array, float min, float max) {
        float range = max - min;
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, min + RANDOM.nextFloat() * range);
        }
    }

    private static TornadoDevice getMetalDevice() {
        TornadoRuntime runtime = TornadoRuntimeProvider.getTornadoRuntime();
        for (int i = 0; i < runtime.getNumBackends(); i++) {
            if (runtime.getBackendType(i) == TornadoVMBackendType.METAL) {
                return runtime.getBackend(i).getDevice(0);
            }
        }
        throw new RuntimeException("No Metal backend available. Rebuild TornadoVM with --backend metal.");
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: MatrixMultiplication1DMetalSingleKernelBenchmark <kernel.metal> [size] [local] [--entry=NAME]");
            System.exit(1);
        }

        String entryPoint = DEFAULT_ENTRY;
        java.util.List<String> pos = new java.util.ArrayList<>();
        for (String a : args) {
            if (a.startsWith("--entry=")) {
                entryPoint = a.substring(8);
            } else {
                pos.add(a);
            }
        }

        String kernelPath = pos.get(0);
        int size = (pos.size() >= 2) ? Integer.parseInt(pos.get(1)) : DEFAULT_SIZE;
        int local = (pos.size() >= 3) ? Integer.parseInt(pos.get(2)) : DEFAULT_LOCAL;

        System.out.println("Kernel: " + kernelPath);
        System.out.println("Backend: Metal");
        System.out.println("Matrix size: " + size + "x" + size);
        System.out.println("Local work size: " + local + "x" + local);
        System.out.println("Entry point: " + entryPoint);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        FloatArray matrixA = new FloatArray(size * size);
        FloatArray matrixB = new FloatArray(size * size);
        FloatArray matrixC = new FloatArray(size * size);

        TornadoDevice device = getMetalDevice();
        System.out.println("Device: " + device);

        fillRandomData(matrixA, -1.0f, 1.0f);
        fillRandomData(matrixB, -1.0f, 1.0f);

        AccessorParameters accessors = new AccessorParameters(4);
        accessors.set(0, matrixA, Access.READ_ONLY);
        accessors.set(1, matrixB, Access.READ_ONLY);
        accessors.set(2, matrixC, Access.WRITE_ONLY);
        accessors.set(3, Integer.valueOf(size), Access.NONE);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA, matrixB)
                .prebuiltTask("t0", entryPoint, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, matrixC);

        ImmutableTaskGraph snapshot = graph.snapshot();

        WorkerGrid2D worker = new WorkerGrid2D(size, size);
        worker.setLocalWork(local, local, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        ArrayList<Long> kernelTimes = new ArrayList<>();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler);

            System.out.println("Warming up...");
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                plan.execute();
            }

            System.out.println("Measuring kernel time...");
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                TornadoExecutionResult result = plan
                        .withProfiler(ProfilerMode.SILENT)
                        .execute();
                TornadoProfilerResult profilerResult = result.getProfilerResult();
                kernelTimes.add(profilerResult.getDeviceKernelTime());
            }
        }

        LongSummaryStatistics stats = kernelTimes.stream().mapToLong(Long::longValue).summaryStatistics();
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
