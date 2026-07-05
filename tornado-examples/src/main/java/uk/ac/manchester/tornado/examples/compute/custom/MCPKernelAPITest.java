/*
 * Test for the new MCP Kernel Comparison API
 *
 * This demonstrates:
 * 1. Running a kernel with TornadoVM
 * 2. Extracting the generated kernel source AFTER execution
 * 3. Replacing the kernel with a modified version
 * 4. Re-running to compare
 */
package uk.ac.manchester.tornado.examples.compute.custom;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class MCPKernelAPITest {

    // Simple vector addition kernel
    public static void vectorAdd(FloatArray a, FloatArray b, FloatArray c) {
        for (@Parallel int i = 0; i < a.getSize(); i++) {
            c.set(i, a.get(i) + b.get(i));
        }
    }

    public static void main(String[] args) {
        final int size = 1024;

        // Initialize data
        FloatArray a = new FloatArray(size);
        FloatArray b = new FloatArray(size);
        FloatArray c = new FloatArray(size);

        for (int i = 0; i < size; i++) {
            a.set(i, i);
            b.set(i, i * 2);
        }

        // Create TaskGraph
        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                .task("t0", MCPKernelAPITest::vectorAdd, a, b, c)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);

        System.out.println("=== MCP Kernel API Test ===\n");

        // Step 1: Execute the original kernel
        System.out.println("Step 1: Executing original kernel...");
        executor.execute();

        // Verify result
        boolean correct = true;
        for (int i = 0; i < 10; i++) {
            float expected = i + (i * 2);
            if (Math.abs(c.get(i) - expected) > 0.001f) {
                correct = false;
                break;
            }
        }
        System.out.println("Original kernel result: " + (correct ? "CORRECT" : "INCORRECT"));
        System.out.println("Sample output: c[0]=" + c.get(0) + ", c[1]=" + c.get(1) + ", c[2]=" + c.get(2));

        // Step 2: Get the generated kernel source
        System.out.println("\nStep 2: Extracting generated kernel source...");
        String kernelSource = executor.getGeneratedKernelSource("t0");

        if (kernelSource != null) {
            System.out.println("\n--- Generated Kernel Source ---");
            System.out.println(kernelSource);
            System.out.println("--- End of Kernel Source ---\n");

            // Step 3: Demonstrate kernel replacement
            // For this test, we'll just add a comment to show replacement works
            System.out.println("Step 3: Replacing kernel with modified version...");

            // Add a comment to the kernel (simple modification to prove it works)
            String modifiedKernel = "// MCP OPTIMIZED VERSION\n" + kernelSource;

            boolean replaced = executor.replaceKernelSource("t0", modifiedKernel);
            System.out.println("Kernel replacement: " + (replaced ? "SUCCESS" : "FAILED"));

            if (replaced) {
                // Step 4: Re-execute with the modified kernel
                System.out.println("\nStep 4: Re-executing with modified kernel...");

                // Reset output array
                for (int i = 0; i < size; i++) {
                    c.set(i, 0);
                }

                executor.execute();

                // Verify result again
                correct = true;
                for (int i = 0; i < 10; i++) {
                    float expected = i + (i * 2);
                    if (Math.abs(c.get(i) - expected) > 0.001f) {
                        correct = false;
                        break;
                    }
                }
                System.out.println("Modified kernel result: " + (correct ? "CORRECT" : "INCORRECT"));
                System.out.println("Sample output: c[0]=" + c.get(0) + ", c[1]=" + c.get(1) + ", c[2]=" + c.get(2));
            }
        } else {
            System.out.println("ERROR: Could not retrieve kernel source!");
            System.out.println("This might mean the kernel hasn't been compiled yet or taskId is incorrect.");
        }

        System.out.println("\n=== Test Complete ===");
    }
}
