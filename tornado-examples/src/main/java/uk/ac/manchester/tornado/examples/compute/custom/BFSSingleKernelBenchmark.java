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
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

/**
 * Single-kernel benchmark for BFS (Breadth-First Search).
 * Measures ONLY accumulated kernel execution time across all BFS levels.
 * Run once per kernel for fair comparison - no ordering bias.
 *
 * Usage: java ... BFSSingleKernelBenchmark <kernel.cl> [numNodes]
 *
 * Default numNodes: 2000
 */
public class BFSSingleKernelBenchmark {

    private static final int DEFAULT_NUM_NODES = 2000;
    private static final int LOCAL_WORK_SIZE = 16;
    private static final int WARM_UP_ITERATIONS = 5;
    private static final int BENCHMARK_ITERATIONS = 20;
    private static final int MAX_BFS_LEVELS = 100;
    private static final String ENTRY_POINT = "runBFS";
    private static final Random RANDOM = new Random(42);

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
            vertices.set(i, (i == root) ? 0 : -1);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: BFSSingleKernelBenchmark <kernel.cl> [numNodes]");
            System.out.println("  Default numNodes: " + DEFAULT_NUM_NODES);
            System.exit(1);
        }

        String kernelPath = args[0];
        int numNodes = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_NUM_NODES;
        int rootNode = 0;

        System.out.println("Kernel: " + kernelPath);
        System.out.println("Number of nodes: " + numNodes);
        System.out.println("Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);

        // Create graph data
        IntArray adjacencyMatrix = new IntArray(numNodes * numNodes);
        IntArray vertices = new IntArray(numNodes);
        IntArray modify = new IntArray(1);
        IntArray currentDepth = new IntArray(1);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        generateRandomGraph(adjacencyMatrix, numNodes, rootNode);

        // Set up kernel parameters
        AccessorParameters accessors = new AccessorParameters(5);
        accessors.set(0, vertices, Access.READ_WRITE);
        accessors.set(1, adjacencyMatrix, Access.READ_ONLY);
        accessors.set(2, Integer.valueOf(numNodes), Access.NONE);
        accessors.set(3, modify, Access.READ_WRITE);
        accessors.set(4, currentDepth, Access.READ_ONLY);

        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, vertices, adjacencyMatrix, modify, currentDepth)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, vertices, modify);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // Calculate local work size
        int localSize = LOCAL_WORK_SIZE;
        while (numNodes % localSize != 0 && localSize > 1) {
            localSize--;
        }

        WorkerGrid2D worker = new WorkerGrid2D(numNodes, numNodes);
        worker.setLocalWork(localSize, localSize, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        ArrayList<Long> totalKernelTimes = new ArrayList<>();
        ArrayList<Integer> bfsLevels = new ArrayList<>();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler);

            // Warmup - run full BFS traversals
            System.out.println("Warming up...");
            for (int w = 0; w < WARM_UP_ITERATIONS; w++) {
                initializeVertices(vertices, numNodes, rootNode);
                boolean done = false;
                int level = 0;

                while (!done && level < MAX_BFS_LEVELS) {
                    modify.init(1);
                    currentDepth.set(0, level);
                    plan.execute();

                    if (modify.get(0) == 1) {
                        done = true;
                    }
                    level++;
                }
            }

            // Benchmark - measure kernel time only
            System.out.println("Measuring kernel time...");
            for (int b = 0; b < BENCHMARK_ITERATIONS; b++) {
                initializeVertices(vertices, numNodes, rootNode);
                boolean done = false;
                int level = 0;
                long accumulatedKernelTime = 0;

                while (!done && level < MAX_BFS_LEVELS) {
                    modify.init(1);
                    currentDepth.set(0, level);

                    TornadoExecutionResult result = plan
                            .withProfiler(ProfilerMode.SILENT)
                            .execute();

                    TornadoProfilerResult profilerResult = result.getProfilerResult();
                    accumulatedKernelTime += profilerResult.getDeviceKernelTime();

                    if (modify.get(0) == 1) {
                        done = true;
                    }
                    level++;
                }

                totalKernelTimes.add(accumulatedKernelTime);
                bfsLevels.add(level);
            }
        }

        LongSummaryStatistics stats = totalKernelTimes.stream().mapToLong(Long::longValue).summaryStatistics();
        double avgLevels = bfsLevels.stream().mapToInt(Integer::intValue).average().orElse(0);

        // Edge traversals: approximate TEPS (Traversed Edges Per Second)
        // Each level processes potentially all edges in adjacency matrix
        long totalEdgeTraversals = (long) numNodes * numNodes; // per BFS
        double mteps = (totalEdgeTraversals * 1e-6) / (stats.getAverage() * 1e-9); // Million TEPS

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY - accumulated across all BFS levels)");
        System.out.println("==============================================================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Nodes: %d%n", numNodes);
        System.out.printf("Avg BFS levels: %.1f%n", avgLevels);
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("MTEPS (Million Traversed Edges/s): %.2f%n", mteps);
    }
}
