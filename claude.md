# TornadoVM Single Kernel Benchmark Guide

## Overview

This document explains the Single Kernel Benchmark methodology for fairly comparing TornadoVM-generated OpenCL kernels against hand-optimized custom kernels.

## System Information

### Remote Server
- **Host**: `serhii@storm`
- **Location**: `~/TornadoVM-fork-2`
- **GPU**: NVIDIA GeForce RTX 4090
- **Platform**: Linux (Ubuntu)
- **Java**: GraalVM JDK 21.0.9+7.1

### Environment Setup
```bash
# Set Java home
export JAVA_HOME=$HOME/graalvm-jdk-21.0.9+7.1
export PATH=$JAVA_HOME/bin:$PATH

# Source TornadoVM environment (REQUIRED after each login)
source setvars.sh

# Verify setup
echo $TORNADO_SDK
```

### Build Command
```bash
make
# or: bin/compile --jdk jdk21 --backend opencl
```

---

## Why Single Kernel Benchmarks?

### The Problem: Ordering Bias

When comparing two kernels in the same benchmark run:
```
[Kernel A warmup] → [Kernel A measure] → [Kernel B warmup] → [Kernel B measure]
```

**Kernel B always benefits from:**
- GPU being in boosted clock state
- Driver optimizations cached
- Memory controllers warmed up

**Result:** Even identical kernels show 20-30% difference due to ordering, not actual performance.

### The Solution: Single Kernel Per Run

```
Run 1: [cold GPU] → [Kernel A warmup] → [Kernel A measure]
Run 2: [cold GPU] → [Kernel B warmup] → [Kernel B measure]
```

**Benefits:**
- Both kernels start from identical cold GPU state
- Both get identical warmup treatment
- Fair, reproducible comparison
- Measures actual kernel performance difference

### What We Measure: KERNEL_TIME Only

Using `getDeviceKernelTime()` from TornadoVM profiler:
- Measures ONLY GPU kernel execution time
- Excludes data transfers
- Excludes host-side overhead
- Excludes compilation time (handled by warmup)

---

## Available Benchmarks

| Algorithm | Benchmark Class | Validator Class | Metric |
|-----------|-----------------|-----------------|--------|
| Matrix-Vector Row | `MatrixVectorRowMajorSingleKernelBenchmark` | `KernelValidator` | GFLOP/s |
| MatrixMul 2D Local | `MatrixMul2DLocalMemorySingleKernelBenchmark` | `MatrixMul2DLocalMemoryValidator` | GFLOP/s |
| MatrixMul 1D | `MatrixMultiplication1DSingleKernelBenchmark` | `MatrixMultiplication1DValidator` | GFLOP/s |
| NBody | `NBodySingleKernelBenchmark` | `NBodyValidator` | GFLOP/s, Billion Interactions/s |
| BFS | `BFSSingleKernelBenchmark` | `BFSValidator` | MTEPS |
| Mandelbrot | `MandelbrotSingleKernelBenchmark` | `MandelbrotValidator` | MPixels/s |
| MonteCarlo | `MonteCarloSingleKernelBenchmark` | `MonteCarloValidator` | MSamples/s |
| BlackScholes | `BlackScholesSingleKernelBenchmark` | `BlackScholesValidator` | MOptions/s |
| BlurFilter | `BlurFilterSingleKernelBenchmark` | `BlurFilterValidator` | MPixels/s, GFLOP/s |

---

## How to Run Benchmarks

### General Command Format
```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.<BenchmarkClass> \
  <kernel.cl> [size_params]
```

### Matrix-Vector Row Major
```bash
# Benchmark
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixVectorRowMajorSingleKernelBenchmark \
  kernels/matrixvectorrow_generated.cl

java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixVectorRowMajorSingleKernelBenchmark \
  kernels/matrixvectorrow_custom.cl

# Validate
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.KernelValidator \
  kernels/matrixvectorrow_generated.cl kernels/matrixvectorrow_custom.cl
```

### Matrix Multiplication 2D (Local Memory)
```bash
# Benchmark (default size 1024)
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMul2DLocalMemorySingleKernelBenchmark \
  kernels/matrixmul2dlocalmemory_generated.cl 1024

# Validate
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMul2DLocalMemoryValidator \
  kernels/matrixmul2dlocalmemory_generated.cl kernels/matrixmul2dlocalmemory_custom.cl
```

### Matrix Multiplication 1D
```bash
# Benchmark
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMultiplication1DSingleKernelBenchmark \
  kernels/matrixmultiplication1d_generated.cl 1024

# Validate
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMultiplication1DValidator \
  kernels/matrixmultiplication1d_generated.cl kernels/matrixmultiplication1d_custom.cl
```

### NBody
```bash
# Benchmark (default 16384 bodies)
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.NBodySingleKernelBenchmark \
  kernels/nbody_generated.cl 16384

# Validate (smaller size for speed)
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.NBodyValidator \
  kernels/nbody_generated.cl kernels/nbody_optimized.cl --bodies=1024
```

### BFS
```bash
# Benchmark (default 2000 nodes)
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.BFSSingleKernelBenchmark \
  kernels/bfs_generated.cl 2000

# Validate
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.BFSValidator \
  kernels/bfs_generated.cl kernels/bfs_custom.cl --nodes=1000
```

### Mandelbrot
```bash
# Benchmark (default 1024x1024)
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MandelbrotSingleKernelBenchmark \
  kernels/mandelbrot_generated.cl 1024

# Validate
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MandelbrotValidator \
  kernels/mandelbrot_generated.cl kernels/mandelbrot_custom.cl --size=512
```

### MonteCarlo
```bash
# Benchmark (default 16M samples)
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MonteCarloSingleKernelBenchmark \
  kernels/montecarlo_generated.cl 16777216

# Validate
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MonteCarloValidator \
  kernels/montecarlo_generated.cl kernels/montecarlo_custom.cl --samples=1000000
```

### BlackScholes
```bash
# Benchmark (default 4M options)
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.BlackScholesSingleKernelBenchmark \
  kernels/blackscholes_generated.cl 4194304

# Validate
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.BlackScholesValidator \
  kernels/blackscholes_generated.cl kernels/blackscholes_custom.cl --options=100000
```

### BlurFilter
```bash
# Benchmark (default 2048x2048, filter 31x31)
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.BlurFilterSingleKernelBenchmark \
  kernels/blurfilter_generated.cl 2048 31

# Validate
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.BlurFilterValidator \
  kernels/blurfilter_generated.cl kernels/blurfilter_custom.cl --size=512 --filter=15
```

---

## How to Generate Kernels from TornadoVM

Use `--printKernel` flag to see generated OpenCL:
```bash
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.<ExampleClass> 2>&1 | tee output.txt
```

### Examples:
```bash
# Matrix Multiplication 2D Local Memory
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.kernelcontext.matrices.MatrixMul2DLocalMemory

# NBody
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.NBody --params="2048 1"

# BFS
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.BFS

# Mandelbrot
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.Mandelbrot

# MonteCarlo
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.MonteCarlo

# BlackScholes
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.BlackScholes

# BlurFilter
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.BlurFilter
```

Extract the `__kernel void ...` section and save to `kernels/<name>_generated.cl`.

### Kernel Entry Points

| Algorithm | Entry Point Function | Notes |
|-----------|---------------------|-------|
| Matrix-Vector Row | `matrixVectorRowMajor` | 1D parallel |
| MatrixMul 2D Local | `matrixMultiplication` | 2D parallel, local memory |
| MatrixMul 1D | `matrixMultiplication` | 2D parallel |
| NBody | `nBody` | 1D parallel |
| BFS | `runBFS` | 2D parallel, iterative |
| Mandelbrot | `mandelbrotTornado` | 2D parallel |
| MonteCarlo | `computeMontecarlo` | 1D parallel |
| BlackScholes | `blackScholesKernel` | 1D parallel |
| BlurFilter | `compute` | 2D parallel |

---

## How to Create a New Single Kernel Benchmark

### Template Structure

```java
package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;
import java.util.Random;

import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;  // or WorkerGrid2D
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;  // or IntArray
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

public class MyAlgorithmSingleKernelBenchmark {

    private static final int DEFAULT_SIZE = 1024;
    private static final int LOCAL_WORK_SIZE = 256;
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;
    private static final String ENTRY_POINT = "myKernelFunction";  // Must match .cl file
    private static final Random RANDOM = new Random(42);

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length < 1) {
            System.out.println("Usage: MyAlgorithmSingleKernelBenchmark <kernel.cl> [size]");
            System.exit(1);
        }

        String kernelPath = args[0];
        int size = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_SIZE;

        // 1. Print configuration
        System.out.println("Kernel: " + kernelPath);
        System.out.println("Size: " + size);

        // 2. Allocate data arrays
        FloatArray input = new FloatArray(size);
        FloatArray output = new FloatArray(size);
        // ... initialize data ...

        TornadoDevice device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        System.out.println("Device: " + device);

        // 3. Set up AccessorParameters (must match kernel signature)
        AccessorParameters accessors = new AccessorParameters(N);  // N = number of params
        accessors.set(0, input, Access.READ_ONLY);
        accessors.set(1, output, Access.WRITE_ONLY);
        accessors.set(2, Integer.valueOf(size), Access.NONE);
        // ... more parameters ...

        // 4. Create TaskGraph
        TaskGraph graph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input)
                .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // 5. Configure grid (1D or 2D)
        WorkerGrid1D worker = new WorkerGrid1D(size);
        worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        ArrayList<Long> kernelTimes = new ArrayList<>();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler);

            // 6. Warmup phase
            System.out.println("Warming up...");
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                plan.execute();
            }

            // 7. Measurement phase - KERNEL TIME ONLY
            System.out.println("Measuring kernel time...");
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                TornadoExecutionResult result = plan
                        .withProfiler(ProfilerMode.SILENT)
                        .execute();

                TornadoProfilerResult profilerResult = result.getProfilerResult();
                long kernelTime = profilerResult.getDeviceKernelTime();
                kernelTimes.add(kernelTime);
            }
        }

        // 8. Calculate and print statistics
        LongSummaryStatistics stats = kernelTimes.stream()
                .mapToLong(Long::longValue).summaryStatistics();

        long totalFlops = /* calculate based on algorithm */;
        double gflops = (totalFlops * 1e-9) / (stats.getAverage() * 1e-9);

        System.out.println();
        System.out.println("Results (KERNEL TIME ONLY)");
        System.out.println("==========================");
        System.out.printf("Kernel: %s%n", kernelPath);
        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
        System.out.printf("GFLOP/s: %.2f%n", gflops);
    }
}
```

### Key Points

1. **Entry Point**: Must match the kernel function name in the `.cl` file
2. **AccessorParameters**: Must match kernel signature exactly (order matters)
3. **Grid Configuration**: Use WorkerGrid1D or WorkerGrid2D based on algorithm
4. **Warmup**: Essential to handle JIT compilation and GPU warm-up
5. **ProfilerMode.SILENT**: Gets kernel time without printing profiler output

---

## Kernel File Locations

All kernels are stored in `~/TornadoVM-fork-2/kernels/`:

```
kernels/
├── matrixvectorrow_generated.cl
├── matrixvectorrow_custom.cl
├── matrixmul2dlocalmemory_generated.cl
├── matrixmul2dlocalmemory_custom.cl
├── matrixmultiplication1d_generated.cl
├── matrixmultiplication1d_custom.cl
├── nbody_generated.cl
├── nbody_optimized.cl
├── nbody/
│   ├── nbody_opt1_fp32_rsqrt.cl
│   ├── nbody_opt2_restrict.cl
│   ├── nbody_opt3_workgroup.cl
│   ├── nbody_opt4_unroll.cl
│   ├── nbody_opt5_register_cache.cl
│   ├── nbody_opt6_scalar_accum.cl
│   └── nbody_opt7_local_memory.cl
├── bfs_generated.cl
├── bfs_custom.cl
├── mandelbrot_generated.cl
├── mandelbrot_custom.cl
├── montecarlo_generated.cl
├── montecarlo_custom.cl
├── blackscholes_generated.cl
├── blackscholes_custom.cl
├── blurfilter_generated.cl
├── blurfilter_custom.cl
├── opt1_restrict.cl
├── opt2_unroll.cl
├── opt3_workgroup.cl
├── opt4_local_memory.cl
└── opt5_local_unroll.cl
```

---

## TornadoVM Kernel Signature

Generated kernels have TornadoVM wrapper parameters:

```c
__kernel void myFunction(
    __global long *_kernel_context,      // TornadoVM internal
    __constant uchar *_constant_region,  // TornadoVM internal
    __local uchar *_local_region,        // TornadoVM internal
    __global int *_atomics,              // TornadoVM internal
    // User parameters start here:
    __global uchar *inputArray,
    __global uchar *outputArray,
    __private int size
)
```

**Important**: When creating custom kernels, keep the same signature as generated kernels.

---

## Optimization Techniques Tested

### Matrix Multiplication
1. `restrict` keyword - pointer aliasing hints
2. Loop unrolling (4x)
3. Explicit work-group size
4. Local memory tiling (biggest impact ~22%)
5. Local memory + unrolling

### NBody
1. `native_rsqrt()` instead of `1.0f/sqrt()`
2. `restrict` keyword
3. Explicit work-group size
4. Loop unrolling
5. Register caching
6. Scalar accumulators
7. Local memory tiling

---

## Troubleshooting

### "Error: could not open tornado-argfile"
```bash
source setvars.sh
```

### "ClassNotFoundException"
```bash
make  # Rebuild after adding new Java files
```

### "clCreateKernel -> Returned: -46" (Invalid kernel name)
Entry point name in benchmark doesn't match function name in `.cl` file.

### "file does not exist: kernels/..."
Check kernel file path and name.

---

## Quick Reference

```bash
# After SSH to server
cd ~/TornadoVM-fork-2
source setvars.sh

# Rebuild
make

# Run any benchmark
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.<ClassName> \
  kernels/<kernel>.cl [params]
```
