package uk.ac.manchester.tornado.examples.compute.custom;

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

/**
 * PTX Validator for BFS (Breadth-First Search) kernels.
 * Compares outputs from two PTX kernels to verify they produce identical results.
 *
 * Usage: BFSPTXValidator <kernel1.ptx> <kernel2.ptx> [numNodes]
 *
 * Default numNodes: 500 (smaller for faster validation)
 */
public class BFSPTXValidator {

    private static final int DEFAULT_NUM_NODES = 500;
    private static final int LOCAL_WORK_SIZE_X = 16;
    private static final int LOCAL_WORK_SIZE_Y = 16;
    private static final int MAX_BFS_LEVELS = 100;
    private static final String ENTRY_POINT = "runBFS";

    private static void connect(int from, int to, IntArray graph, int N) {
        if (from != to && (graph.get(from * N + to) == 0)) {
            graph.set(from * N + to, 1);
        }
    }

    private static int[] generateIntRandomArray(int numNodes, Random r) {
        int bound = Math.min(10, numNodes);
        IntStream streamArray = r.ints(bound, 0, numNodes);
        return streamArray.toArray();
    }

    private static void generateRandomGraph(IntArray adjacencyMatrix, int numNodes, int root, long seed) {
        adjacencyMatrix.init(0);
        Random r = new Random(seed);
        int bound = Math.min(numNodes / 10, 100);
        IntStream fromStream = r.ints(bound, 0, numNodes);
        int[] f = fromStream.toArray();

        for (int k = 0; k < f.length; k++) {
            int from = f[k];
            if (k == 0) {
                from = root;
            }
            int[] toArray = generateIntRandomArray(numNodes, r);
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

    private static void runBFS(String kernelPath, IntArray vertices, IntArray adjacencyMatrix,
                                int numNodes, TornadoDevice device) throws TornadoExecutionPlanException {
        IntArray modify = new IntArray(1);
        IntArray currentDepth = new IntArray(1);

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

        int localSizeX = LOCAL_WORK_SIZE_X;
        int localSizeY = LOCAL_WORK_SIZE_Y;
        while (numNodes % localSizeX != 0 && localSizeX > 1) {
            localSizeX--;
        }
        while (numNodes % localSizeY != 0 && localSizeY > 1) {
            localSizeY--;
        }

        WorkerGrid2D worker = new WorkerGrid2D(numNodes, numNodes);
        worker.setLocalWork(localSizeX, localSizeY, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler);

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
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 2) {
            System.out.println("Usage: BFSPTXValidator <kernel1.ptx> <kernel2.ptx> [numNodes]");
            System.out.println("  Default numNodes: " + DEFAULT_NUM_NODES);
            System.out.println();
            System.out.println("Example:");
            System.out.println("  BFSPTXValidator kernels/ptx/bfs_generated.ptx kernels/ptx/bfs_custom.ptx 500");
            System.exit(1);
        }

        String kernel1Path = args[0];
        String kernel2Path = args[1];
        int numNodes = (args.length >= 3) ? Integer.parseInt(args[2]) : DEFAULT_NUM_NODES;
        int rootNode = 0;
        long seed = 42;

        System.out.println("=== BFS PTX Validator ===");
        System.out.println("Kernel 1: " + kernel1Path);
        System.out.println("Kernel 2: " + kernel2Path);
        System.out.println("Number of nodes: " + numNodes);

        // Allocate arrays
        IntArray adjacencyMatrix = new IntArray(numNodes * numNodes);
        IntArray vertices1 = new IntArray(numNodes);
        IntArray vertices2 = new IntArray(numNodes);

        // Generate identical graph
        generateRandomGraph(adjacencyMatrix, numNodes, rootNode, seed);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // Run kernel 1
        System.out.println("\nRunning kernel 1...");
        initializeVertices(vertices1, numNodes, rootNode);
        try {
            runBFS(kernel1Path, vertices1, adjacencyMatrix, numNodes, device);
            System.out.println("Kernel 1 completed.");
        } catch (Exception e) {
            System.out.println("Kernel 1 FAILED: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }

        // Run kernel 2
        System.out.println("Running kernel 2...");
        initializeVertices(vertices2, numNodes, rootNode);
        try {
            runBFS(kernel2Path, vertices2, adjacencyMatrix, numNodes, device);
            System.out.println("Kernel 2 completed.");
        } catch (Exception e) {
            System.out.println("Kernel 2 FAILED: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }

        // Compare results
        System.out.println("\nComparing BFS results...");
        int errors = 0;
        int reachable1 = 0, reachable2 = 0;

        for (int i = 0; i < numNodes; i++) {
            int depth1 = vertices1.get(i);
            int depth2 = vertices2.get(i);

            if (depth1 >= 0) reachable1++;
            if (depth2 >= 0) reachable2++;

            if (depth1 != depth2) {
                errors++;
                if (errors <= 5) {
                    System.out.printf("  Mismatch at node %d: depth=%d vs depth=%d%n", i, depth1, depth2);
                }
            }
        }

        // Print summary
        System.out.println();
        System.out.println("=== Validation Results ===");
        System.out.printf("Reachable nodes - K1: %d, K2: %d%n", reachable1, reachable2);
        System.out.printf("Mismatches: %d / %d nodes%n", errors, numNodes);
        System.out.println();

        if (errors == 0) {
            System.out.println("VALIDATION PASSED - Kernels produce identical BFS results");
        } else {
            System.out.println("VALIDATION FAILED - Kernels produce different BFS results");
            if (errors > 5) {
                System.out.println("(Only first 5 errors shown)");
            }
        }

        // Print sample values
        System.out.println();
        System.out.println("Sample BFS depths:");
        for (int i = 0; i < Math.min(5, numNodes); i++) {
            System.out.printf("  Node %d: K1=%d, K2=%d%n", i, vertices1.get(i), vertices2.get(i));
        }
    }
}
