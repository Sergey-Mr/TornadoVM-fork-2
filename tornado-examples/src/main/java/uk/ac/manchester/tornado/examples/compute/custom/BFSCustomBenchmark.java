package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;
import java.util.Random;
import java.util.stream.IntStream;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Benchmarks TornadoVM's generated BFS runBFS kernel against a user-provided
 * OpenCL implementation using {@code prebuiltTask}.
 *
 * <p>
 * How to run:
 * </p>
 * <code>
 * # Using the comparison script:
 * ./scripts/compare_kernels.sh BFSCustomBenchmark bfs_runBFS
 *
 * # Or directly:
 * java --enable-preview @tornado-argfile -cp /tmp/tornado-custom-classes:dist/tornado-sdk/tornado-sdk-1.1.2-dev-1992554/share/java/tornado/* \
 *   uk.ac.manchester.tornado.examples.compute.custom.BFSCustomBenchmark \
 *   kernels/bfs_runBFS_generated.cl kernels/bfs_runBFS_custom.cl
 * </code>
 */
public final class BFSCustomBenchmark {

    private static final String DEFAULT_GENERATED_KERNEL = "kernels/bfs_runBFS_generated.cl";
    private static final String DEFAULT_CUSTOM_KERNEL = "kernels/bfs_runBFS_custom.cl";
    private static final String ENTRY_POINT = "runBFS";

    private static final int DEFAULT_NUM_NODES = 1000;
    private static final int WARM_UP_ITERATIONS = 10;
    private static final int BENCHMARK_ITERATIONS = 30;
    private static final int MAX_BFS_ITERATIONS = 50;

    private static final Random RANDOM = new Random(42);

    private BFSCustomBenchmark() {
        // utility class
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int numNodes = (args.length >= 1) ? Integer.parseInt(args[0]) : DEFAULT_NUM_NODES;
        String generatedKernelPath = (args.length >= 2) ? args[1] : DEFAULT_GENERATED_KERNEL;
        String customKernelPath = (args.length >= 3) ? args[2] : DEFAULT_CUSTOM_KERNEL;

        System.out.println("BFS Benchmark Configuration");
        System.out.println("===========================");
        System.out.println("Number of nodes  : " + numNodes);
        System.out.println("Generated kernel : " + generatedKernelPath);
        System.out.println("Custom kernel    : " + customKernelPath);
        System.out.println();

        // Create graph data
        IntArray adjacencyMatrix = new IntArray(numNodes * numNodes);
        IntArray verticesSeq = new IntArray(numNodes);
        IntArray verticesGenerated = new IntArray(numNodes);
        IntArray verticesCustom = new IntArray(numNodes);

        int rootNode = 0;
        generateRandomGraph(adjacencyMatrix, numNodes, rootNode);

        // Run sequential baseline
        long seqTime = runSequentialBFS(adjacencyMatrix, verticesSeq, numNodes, rootNode);

        // Run generated kernel
        BenchmarkResult generatedResult = benchmarkKernel("bfsGenerated", generatedKernelPath,
                adjacencyMatrix, verticesGenerated, numNodes, rootNode);

        // Run custom kernel
        BenchmarkResult customResult = benchmarkKernel("bfsCustom", customKernelPath,
                adjacencyMatrix, verticesCustom, numNodes, rootNode);

        // Validate results
        boolean generatedValid = validate(verticesSeq, verticesGenerated, numNodes);
        boolean customValid = validate(verticesSeq, verticesCustom, numNodes);

        if (!generatedValid) {
            System.out.println("[WARN] Generated kernel result does not match the sequential baseline.");
        }
        if (!customValid) {
            System.out.println("[WARN] Custom kernel result does not match the sequential baseline.");
        }

        // Print results
        LongSummaryStatistics generatedStats = generatedResult.timingStats();
        LongSummaryStatistics customStats = customResult.timingStats();

        double sequentialMs = seqTime / 1_000_000.0;
        double generatedMs = generatedStats.getAverage() / 1_000_000.0;
        double customMs = customStats.getAverage() / 1_000_000.0;

        System.out.println();
        System.out.println("Benchmark Results");
        System.out.println("=================");
        System.out.printf("Sequential: %.3f ms%n", sequentialMs);
        System.out.printf("Generated : avg=%.3f ms min=%.3f ms max=%.3f ms%n",
                generatedMs, generatedStats.getMin() / 1_000_000.0, generatedStats.getMax() / 1_000_000.0);
        System.out.printf("Custom    : avg=%.3f ms min=%.3f ms max=%.3f ms%n",
                customMs, customStats.getMin() / 1_000_000.0, customStats.getMax() / 1_000_000.0);

        System.out.println();
        System.out.printf("Speedup (Generated vs Sequential): %.2fx%n", seqTime / (double) generatedStats.getAverage());
        System.out.printf("Speedup (Custom vs Sequential)   : %.2fx%n", seqTime / (double) customStats.getAverage());
        System.out.printf("Speedup (Custom vs Generated)    : %.2fx%n", generatedStats.getAverage() / (double) customStats.getAverage());

        if (customStats.getAverage() < generatedStats.getAverage()) {
            System.out.printf("Custom kernel is faster by %.2fx%n", generatedStats.getAverage() / (double) customStats.getAverage());
        } else {
            System.out.printf("Generated kernel is faster by %.2fx%n", customStats.getAverage() / (double) generatedStats.getAverage());
        }
    }

    private static void connect(int from, int to, IntArray graph, int N) {
        if (from != to && (graph.get(from * N + to) == 0)) {
            graph.set(from * N + to, 1);
        }
    }

    private static int[] generateIntRandomArray(int numNodes) {
        Random r = new Random(42);
        int bound = Math.min(10, numNodes);
        IntStream streamArray = r.ints(bound, 0, numNodes);
        return streamArray.toArray();
    }

    private static void generateRandomGraph(IntArray adjacencyMatrix, int numNodes, int root) {
        adjacencyMatrix.init(0);
        Random r = new Random(42);
        int bound = Math.min(numNodes / 10, 100);
        IntStream fromStream = r.ints(bound, 0, numNodes);
        int[] f = fromStream.toArray();

        for (int k = 0; k < f.length; k++) {
            int from = f[k];
            if (k == 0) {
                from = root;
            }
            int[] toArray = generateIntRandomArray(numNodes);
            for (int i = 0; i < toArray.length; i++) {
                connect(from, toArray[i], adjacencyMatrix, numNodes);
            }
        }
    }

    private static void initializeVertices(IntArray vertices, int numNodes, int root) {
        for (int i = 0; i < numNodes; i++) {
            if (i == root) {
                vertices.set(i, 0);
            } else {
                vertices.set(i, -1);
            }
        }
    }

    private static void runBFSSequential(IntArray vertices, IntArray adjacencyMatrix,
            int numNodes, IntArray modify, IntArray currentDepth) {
        for (int from = 0; from < numNodes; from++) {
            for (int to = 0; to < numNodes; to++) {
                int elementAccess = from * numNodes + to;
                if (adjacencyMatrix.get(elementAccess) == 1) {
                    int dfirst = vertices.get(from);
                    int dsecond = vertices.get(to);
                    if ((currentDepth.get(0) == dfirst) && (dsecond == -1)) {
                        vertices.set(to, dfirst + 1);
                        modify.set(0, 0);
                    }
                }
            }
        }
    }

    private static long runSequentialBFS(IntArray adjacencyMatrix, IntArray vertices,
            int numNodes, int root) {
        initializeVertices(vertices, numNodes, root);

        IntArray modify = new IntArray(1);
        IntArray currentDepth = new IntArray(1);

        long startTime = System.nanoTime();

        boolean done = false;
        int iterations = 0;
        while (!done && iterations < MAX_BFS_ITERATIONS) {
            modify.init(1);
            currentDepth.set(0, iterations);

            runBFSSequential(vertices, adjacencyMatrix, numNodes, modify, currentDepth);

            if (modify.get(0) == 1) {
                done = true;
            }
            iterations++;
        }

        long endTime = System.nanoTime();
        return endTime - startTime;
    }

    private static BenchmarkResult benchmarkKernel(String graphName, String kernelPath,
            IntArray adjacencyMatrix, IntArray vertices, int numNodes, int root)
            throws TornadoExecutionPlanException {

        initializeVertices(vertices, numNodes, root);

        IntArray modify = new IntArray(1);
        IntArray currentDepth = new IntArray(1);

        AccessorParameters accessors = new AccessorParameters(5);
        accessors.set(0, vertices, Access.READ_WRITE);
        accessors.set(1, adjacencyMatrix, Access.READ_ONLY);
        accessors.set(2, Integer.valueOf(numNodes), Access.NONE);
        accessors.set(3, modify, Access.READ_WRITE);
        accessors.set(4, currentDepth, Access.READ_ONLY);

        TaskGraph graph = new TaskGraph(graphName)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, vertices, adjacencyMatrix, modify, currentDepth)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, vertices, modify);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // Calculate local work size that divides evenly into numNodes
        int localSize = 16;
        while (numNodes % localSize != 0 && localSize > 1) {
            localSize--;
        }
        if (localSize < 1) {
            localSize = 1;
        }

        WorkerGrid2D workerGrid = new WorkerGrid2D(numNodes, numNodes);
        workerGrid.setLocalWork(localSize, localSize, 1);
        GridScheduler scheduler = new GridScheduler(graphName + ".t0", workerGrid);

        ArrayList<Long> times = new ArrayList<>(BENCHMARK_ITERATIONS);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withGridScheduler(scheduler);

            // Warm up
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                initializeVertices(vertices, numNodes, root);
                boolean done = false;
                int iterations = 0;

                while (!done && iterations < MAX_BFS_ITERATIONS) {
                    modify.init(1);
                    currentDepth.set(0, iterations);
                    plan.execute();

                    if (modify.get(0) == 1) {
                        done = true;
                    }
                    iterations++;
                }
            }

            // Benchmark
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                initializeVertices(vertices, numNodes, root);
                boolean done = false;
                int iterations = 0;

                long start = System.nanoTime();

                while (!done && iterations < MAX_BFS_ITERATIONS) {
                    modify.init(1);
                    currentDepth.set(0, iterations);
                    plan.execute();

                    if (modify.get(0) == 1) {
                        done = true;
                    }
                    iterations++;
                }

                long end = System.nanoTime();
                times.add(end - start);
            }
        }

        return new BenchmarkResult(times.stream().mapToLong(Long::longValue).summaryStatistics());
    }

    private static boolean validate(IntArray reference, IntArray candidate, int numNodes) {
        for (int i = 0; i < numNodes; i++) {
            if (reference.get(i) != candidate.get(i)) {
                System.out.printf("Mismatch at node %d: ref=%d cand=%d%n",
                        i, reference.get(i), candidate.get(i));
                return false;
            }
        }
        return true;
    }

    private static class BenchmarkResult {
        private final LongSummaryStatistics timingStats;

        public BenchmarkResult(LongSummaryStatistics timingStats) {
            this.timingStats = timingStats;
        }

        public LongSummaryStatistics timingStats() {
            return timingStats;
        }
    }
}
