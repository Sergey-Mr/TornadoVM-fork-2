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
 * Validates BFS kernel output against sequential baseline.
 * Separate from benchmark to avoid affecting GPU warmup state.
 *
 * Usage: java ... BFSValidator <kernel1.cl> [kernel2.cl] ... [--nodes=N]
 */
public class BFSValidator {

    private static final int DEFAULT_NUM_NODES = 1000;
    private static final int LOCAL_WORK_SIZE = 16;
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

    // Sequential BFS for reference
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

    private static void runFullSequentialBFS(IntArray adjacencyMatrix, IntArray vertices,
            int numNodes, int root) {
        initializeVertices(vertices, numNodes, root);

        IntArray modify = new IntArray(1);
        IntArray currentDepth = new IntArray(1);

        boolean done = false;
        int level = 0;

        while (!done && level < MAX_BFS_LEVELS) {
            modify.init(1);
            currentDepth.set(0, level);
            runBFSSequential(vertices, adjacencyMatrix, numNodes, modify, currentDepth);

            if (modify.get(0) == 1) {
                done = true;
            }
            level++;
        }
    }

    private static ValidationResult validate(IntArray reference, IntArray candidate, int numNodes) {
        int mismatches = 0;
        int firstMismatchNode = -1;
        int firstMismatchRef = 0;
        int firstMismatchCand = 0;

        for (int i = 0; i < numNodes; i++) {
            if (reference.get(i) != candidate.get(i)) {
                if (firstMismatchNode == -1) {
                    firstMismatchNode = i;
                    firstMismatchRef = reference.get(i);
                    firstMismatchCand = candidate.get(i);
                }
                mismatches++;
            }
        }

        return new ValidationResult(mismatches == 0, mismatches, firstMismatchNode,
                firstMismatchRef, firstMismatchCand);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: BFSValidator <kernel1.cl> [kernel2.cl] ... [--nodes=N]");
            System.exit(1);
        }

        // Parse arguments
        int numNodes = DEFAULT_NUM_NODES;
        java.util.List<String> kernelPaths = new java.util.ArrayList<>();

        for (String arg : args) {
            if (arg.startsWith("--nodes=")) {
                numNodes = Integer.parseInt(arg.substring(8));
            } else {
                kernelPaths.add(arg);
            }
        }

        if (kernelPaths.isEmpty()) {
            System.out.println("Error: No kernel files specified");
            System.exit(1);
        }

        int rootNode = 0;

        // Create graph data
        IntArray adjacencyMatrix = new IntArray(numNodes * numNodes);
        IntArray referenceVertices = new IntArray(numNodes);

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();

        generateRandomGraph(adjacencyMatrix, numNodes, rootNode);

        // Compute sequential reference
        System.out.println("Computing sequential BFS reference (" + numNodes + " nodes)...");
        long start = System.currentTimeMillis();
        runFullSequentialBFS(adjacencyMatrix, referenceVertices, numNodes, rootNode);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("Sequential computation took: " + elapsed + " ms");

        // Count reachable nodes
        int reachable = 0;
        int maxDepth = 0;
        for (int i = 0; i < numNodes; i++) {
            if (referenceVertices.get(i) >= 0) {
                reachable++;
                maxDepth = Math.max(maxDepth, referenceVertices.get(i));
            }
        }
        System.out.println("Reachable nodes: " + reachable + "/" + numNodes);
        System.out.println("Max BFS depth: " + maxDepth);

        System.out.println();
        System.out.println("Validation Results");
        System.out.println("=".repeat(50));

        // Validate each kernel
        for (String kernelPath : kernelPaths) {
            IntArray kernelVertices = new IntArray(numNodes);
            IntArray modify = new IntArray(1);
            IntArray currentDepth = new IntArray(1);

            initializeVertices(kernelVertices, numNodes, rootNode);

            // Set up kernel
            AccessorParameters accessors = new AccessorParameters(5);
            accessors.set(0, kernelVertices, Access.READ_WRITE);
            accessors.set(1, adjacencyMatrix, Access.READ_ONLY);
            accessors.set(2, Integer.valueOf(numNodes), Access.NONE);
            accessors.set(3, modify, Access.READ_WRITE);
            accessors.set(4, currentDepth, Access.READ_ONLY);

            TaskGraph graph = new TaskGraph("validate")
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, kernelVertices, adjacencyMatrix, modify, currentDepth)
                    .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, kernelVertices, modify);

            ImmutableTaskGraph snapshot = graph.snapshot();

            int localSize = LOCAL_WORK_SIZE;
            while (numNodes % localSize != 0 && localSize > 1) {
                localSize--;
            }

            WorkerGrid2D worker = new WorkerGrid2D(numNodes, numNodes);
            worker.setLocalWork(localSize, localSize, 1);
            GridScheduler scheduler = new GridScheduler("validate.t0", worker);

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.withDevice(device).withGridScheduler(scheduler);

                // Run full BFS
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

            // Validate
            ValidationResult result = validate(referenceVertices, kernelVertices, numNodes);

            System.out.printf("%s%n", kernelPath);
            if (result.valid) {
                System.out.printf("  PASSED%n");
            } else {
                System.out.printf("  FAILED: %d mismatches%n", result.mismatches);
                System.out.printf("    First mismatch at node %d: expected=%d, got=%d%n",
                        result.firstMismatchNode, result.firstMismatchRef, result.firstMismatchCand);
            }
        }
    }

    private record ValidationResult(boolean valid, int mismatches, int firstMismatchNode,
            int firstMismatchRef, int firstMismatchCand) {}
}
