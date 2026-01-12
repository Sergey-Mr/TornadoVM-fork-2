/*
 * Headless version of Mandelbrot for kernel generation on servers without X11
 */
package uk.ac.manchester.tornado.examples.compute;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;

/**
 * Headless Mandelbrot for kernel generation.
 *
 * Usage:
 *   tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.MandelbrotHeadless
 */
public class MandelbrotHeadless {

    public static final int SIZE = 1024;

    private static void mandelbrotTornado(int size, ShortArray output) {
        final int iterations = 10000;
        float space = 2.0f / size;

        for (@Parallel int i = 0; i < size; i++) {
            for (@Parallel int j = 0; j < size; j++) {
                float Zr = 0.0f;
                float Zi = 0.0f;
                float Cr = (1 * j * space - 1.5f);
                float Ci = (1 * i * space - 1.0f);
                float ZrN = 0;
                float ZiN = 0;
                int y = 0;
                for (int ii = 0; ii < iterations; ii++) {
                    if (ZiN + ZrN <= 4.0f) {
                        Zi = 2.0f * Zr * Zi + Ci;
                        Zr = 1 * ZrN - ZiN + Cr;
                        ZiN = Zi * Zi;
                        ZrN = Zr * Zr;
                        y++;
                    } else {
                        ii = iterations;
                    }
                }
                short r = (short) ((y * 255) / iterations);
                output.set(i * size + j, r);
            }
        }
    }

    public static void main(String[] args) {
        ShortArray result = new ShortArray(SIZE * SIZE);

        TaskGraph taskGraph = new TaskGraph("s0")
                .task("t0", MandelbrotHeadless::mandelbrotTornado, SIZE, result)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, result);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);
        executor.execute();

        // Verify a few values
        System.out.println("Mandelbrot computation complete");
        System.out.println("Sample values: [0]=" + result.get(0) + ", [512*512]=" + result.get(512*512));
    }
}
