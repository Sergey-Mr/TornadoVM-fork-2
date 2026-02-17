# TornadoVM OpenCL Kernel Benchmarking Methodology

This document provides comprehensive documentation of the benchmarking infrastructure created to fairly compare TornadoVM-generated OpenCL kernels against hand-optimized custom kernels.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution: Single Kernel Benchmarking](#solution-single-kernel-benchmarking)
3. [What We Measure](#what-we-measure)
4. [Infrastructure Overview](#infrastructure-overview)
5. [Benchmark Implementation](#benchmark-implementation)
6. [Validator Implementation](#validator-implementation)
7. [OpenCL Kernel Structure](#opencl-kernel-structure)
8. [Optimization Techniques](#optimization-techniques)
9. [Running Benchmarks](#running-benchmarks)
10. [Creating New Benchmarks](#creating-new-benchmarks)
11. [Results Interpretation](#results-interpretation)

---

## Problem Statement

### The Ordering Bias Problem

When benchmarking two OpenCL kernels in the same JVM execution:

```
[Kernel A warmup] → [Kernel A measure] → [Kernel B warmup] → [Kernel B measure]
```

**Kernel B always benefits from:**
- GPU being in boosted clock state (thermal throttling not yet engaged)
- OpenCL driver optimizations already cached
- GPU memory controllers fully warmed up
- PCIe link in high-bandwidth state

**Measured Impact:** Even when running the exact same kernel twice, the second execution shows 20-30% better performance due purely to ordering - not any actual code difference.

### Why This Matters

If you compare a TornadoVM-generated kernel (run first) against a custom-optimized kernel (run second), you cannot distinguish between:
- Actual optimization improvements
- Artificial ordering bias

This makes it impossible to validate whether optimizations actually help.

---

## Solution: Single Kernel Benchmarking

### Core Principle

**Run each kernel in a completely separate JVM execution:**

```bash
# Execution 1: Generated kernel (cold GPU)
java ... MatrixMultiplication1DSingleKernelBenchmark kernels/matrix1d_generated.cl 1024
# Output: Avg 5.234 ms, 412.5 GFLOP/s

# Execution 2: Custom kernel (cold GPU)
java ... MatrixMultiplication1DSingleKernelBenchmark kernels/matrix1d_custom.cl 1024
# Output: Avg 4.123 ms, 523.8 GFLOP/s
```

### Benefits

| Aspect | Single-Kernel Method | Traditional Method |
|--------|---------------------|-------------------|
| GPU State | Identical cold start | Second kernel benefits |
| Ordering Bias | Eliminated | 20-30% artificial difference |
| Reproducibility | High | Varies by order |
| Fair Comparison | Yes | No |

### The Warmup Phase

Each benchmark includes a warmup phase **within** its own execution:

```java
// Warmup (handles JIT compilation and initial GPU warm-up)
for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
    plan.execute();
}

// Measurement (stable state)
for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
    // Measure kernel time
}
```

This ensures the kernel being tested reaches stable performance, but doesn't give unfair advantage to any other kernel.

---

## What We Measure

### KERNEL_TIME Only

We use TornadoVM's profiler to measure **only GPU kernel execution time**:

```java
TornadoExecutionResult result = plan
        .withProfiler(ProfilerMode.SILENT)
        .execute();

TornadoProfilerResult profilerResult = result.getProfilerResult();
long kernelTime = profilerResult.getDeviceKernelTime();  // Nanoseconds
```

### What's Included vs Excluded

| Included | Excluded |
|----------|----------|
| GPU kernel execution | Host-to-device data transfer |
| On-device memory access | Device-to-host data transfer |
| GPU compute operations | JIT compilation time |
| | Host-side Java overhead |
| | TaskGraph setup time |

### Why Kernel Time Only?

- **Data transfer is constant** - Same data transferred regardless of kernel optimization
- **Compilation is one-time** - Handled by warmup phase
- **Isolates optimization impact** - Shows actual compute improvement
- **Comparable across runs** - Eliminates variable host overhead

---

## Infrastructure Overview

### File Structure

```
TornadoVM-fork-2/
├── CLAUDE.md                           # AI assistant context
├── docs/
│   └── BENCHMARKING.md                 # This documentation
├── kernels/                            # OpenCL kernel files
│   ├── *_generated.cl                  # Extracted from TornadoVM
│   ├── *_custom.cl                     # Hand-optimized versions
│   ├── opt1_restrict.cl                # Progressive optimizations
│   ├── opt2_unroll.cl
│   ├── opt3_workgroup.cl
│   ├── opt4_local_memory.cl
│   ├── opt5_local_unroll.cl
│   ├── nbody/                          # NBody optimization series
│   ├── matrixrowmajor/                 # Matrix-vector optimizations
│   └── macbook/                        # Platform-specific kernels
└── tornado-examples/.../compute/custom/
    ├── *SingleKernelBenchmark.java     # Benchmark classes (11)
    └── *Validator.java                 # Validation classes (11)
```

### Component Roles

| Component | Purpose |
|-----------|---------|
| `*SingleKernelBenchmark.java` | Measures kernel time with warmup/measure phases |
| `*Validator.java` | Verifies kernel correctness against CPU reference |
| `*_generated.cl` | OpenCL extracted from TornadoVM |
| `*_custom.cl` | Hand-optimized OpenCL |
| `opt*.cl` | Progressive optimization test variants |

---

## Benchmark Implementation

### Standard Benchmark Structure

Every benchmark follows this pattern:

```java
public class AlgorithmSingleKernelBenchmark {

    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;
    private static final String ENTRY_POINT = "kernelFunctionName";

    public static void main(String[] args) throws TornadoExecutionPlanException {
        // 1. Parse arguments
        String kernelPath = args[0];
        int size = Integer.parseInt(args[1]);

        // 2. Allocate and initialize data
        FloatArray input = new FloatArray(size);
        FloatArray output = new FloatArray(size);
        fillRandomData(input);

        // 3. Get device
        TornadoDevice device = TornadoRuntimeProvider
            .getTornadoRuntime().getDefaultDevice();

        // 4. Set up AccessorParameters (must match kernel signature)
        AccessorParameters accessors = new AccessorParameters(3);
        accessors.set(0, input, Access.READ_ONLY);
        accessors.set(1, output, Access.WRITE_ONLY);
        accessors.set(2, Integer.valueOf(size), Access.NONE);

        // 5. Build TaskGraph with prebuiltTask
        TaskGraph graph = new TaskGraph("s0")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, input)
            .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();

        // 6. Configure grid scheduler
        WorkerGrid1D worker = new WorkerGrid1D(size);
        worker.setLocalWork(256, 1, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        ArrayList<Long> kernelTimes = new ArrayList<>();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withDevice(device).withGridScheduler(scheduler);

            // 7. WARMUP PHASE (no measurement)
            for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
                plan.execute();
            }

            // 8. MEASUREMENT PHASE (kernel time only)
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                TornadoExecutionResult result = plan
                    .withProfiler(ProfilerMode.SILENT)
                    .execute();

                long kernelTime = result.getProfilerResult().getDeviceKernelTime();
                kernelTimes.add(kernelTime);
            }
        }

        // 9. Calculate and report statistics
        LongSummaryStatistics stats = kernelTimes.stream()
            .mapToLong(Long::longValue).summaryStatistics();

        System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
        System.out.printf("Min: %.3f ms%n", stats.getMin() / 1_000_000.0);
        System.out.printf("Max: %.3f ms%n", stats.getMax() / 1_000_000.0);
    }
}
```

### Key Implementation Details

1. **`prebuiltTask()`** - Loads external .cl file instead of JIT-compiling Java
2. **`ENTRY_POINT`** - Must exactly match the `__kernel void` function name
3. **`ProfilerMode.SILENT`** - Enables profiling without printing
4. **`getDeviceKernelTime()`** - Returns nanoseconds of GPU execution only

### Benchmark Parameters by Algorithm

| Algorithm | Warmup | Iterations | Grid | Local Size |
|-----------|--------|------------|------|------------|
| MatrixMul 1D | 50 | 100 | 2D (NxN) | 4x8 |
| MatrixMul 2D Local | 50 | 100 | 2D (NxN) | 16x16 |
| NBody | 20 | 50 | 1D (bodies) | 256 |
| BFS | 5 | 20 | 2D (NxN) | 4x8 |
| Mandelbrot | 50 | 100 | 2D (WxH) | 16x16 |
| FlashAttention | 50 | 100 | 1D (heads*dim) | 128 |

---

## Validator Implementation

### Purpose

Validators ensure kernel correctness **before** trusting benchmark results. A faster kernel that produces wrong results is useless.

### Validator Structure

```java
public class AlgorithmValidator {

    private static final float TOLERANCE = 1e-4f;

    public static void main(String[] args) throws TornadoExecutionPlanException {
        // 1. Use SMALLER problem size for faster validation
        int size = 1024;  // vs 16384 for benchmarks

        // 2. Compute sequential CPU reference
        float[] reference = computeSequential(input, size);

        // 3. Run kernel ONCE (no warmup needed for correctness)
        plan.execute();

        // 4. Compare with tolerance
        ValidationResult result = validate(reference, kernelOutput, TOLERANCE);

        // 5. Report pass/fail
        if (result.valid) {
            System.out.println("PASSED");
        } else {
            System.out.printf("FAILED: %d mismatches (max diff: %.2e)%n",
                result.mismatches, result.maxDiff);
        }
    }
}
```

### Tolerance Values by Algorithm

| Algorithm | Tolerance | Reason |
|-----------|-----------|--------|
| Matrix operations | 1e-4f | FP32 accumulation errors |
| NBody | 0.1f | Large accumulated errors from iterations |
| BFS | exact (0) | Integer results must match exactly |
| FlashAttention | 1e-2f | Softmax numerical stability |

### Validation Workflow

```bash
# ALWAYS validate before benchmarking
java ... Validator kernels/generated.cl kernels/custom.cl

# If PASSED, then benchmark
java ... Benchmark kernels/generated.cl
java ... Benchmark kernels/custom.cl
```

---

## OpenCL Kernel Structure

### TornadoVM Kernel Signature

All TornadoVM-generated kernels have this signature:

```c
__kernel void functionName(
    __global long *_kernel_context,      // Contains problem dimensions
    __constant uchar *_constant_region,  // TornadoVM constants
    __local uchar *_local_region,        // Shared memory region
    __global int *_atomics,              // Atomic counters
    // User parameters start here:
    __global uchar *arrayA,              // Cast to actual type
    __global uchar *arrayB,
    __private int size
)
```

### Accessing Data

```c
// Get problem size from kernel context
const int N = (int)_kernel_context[0];

// Cast uchar* to actual type WITH OFFSET
__global const float *a = ((__global const float *)arrayA) + 4;
__global const float *b = ((__global const float *)arrayB) + 4;
__global float *c = ((__global float *)arrayC) + 4;
```

**Critical**: The `+4` offset is required because TornadoVM stores metadata in the first 16 bytes.

### Custom Kernel Requirements

When creating optimized kernels:
1. **Keep the same signature** - All 4 TornadoVM parameters plus user parameters
2. **Keep the same function name** - Must match benchmark's `ENTRY_POINT`
3. **Keep the `+4` offset** - For array data access
4. **Get N from `_kernel_context[0]`** - Not from user parameters

---

## Optimization Techniques

### Matrix Multiplication Optimizations

**Progressive optimization series (opt1-opt5):**

| File | Technique | Impact |
|------|-----------|--------|
| opt1_restrict.cl | `restrict` keyword | ~5% (compiler hints) |
| opt2_unroll.cl | `#pragma unroll 4` | ~10% (reduced branch overhead) |
| opt3_workgroup.cl | `reqd_work_group_size` | ~5% (compiler optimizations) |
| opt4_local_memory.cl | Local memory tiling | ~22% (reduced memory traffic) |
| opt5_local_unroll.cl | Local + unroll | ~25% (combined benefits) |

**Local Memory Tiling Example:**

```c
#define TS 16

__local float As[TS][TS];
__local float Bs[TS][TS];

// Cooperative load into local memory
As[ly][lx] = a[row * N + (k0 + lx)];
Bs[ly][lx] = b[(k0 + ly) * N + col];

barrier(CLK_LOCAL_MEM_FENCE);

// Compute from fast local memory
for (int k = 0; k < TS; k++) {
    acc = fma(As[ly][k], Bs[k][lx], acc);
}

barrier(CLK_LOCAL_MEM_FENCE);
```

### NBody Optimizations

**Progressive optimization series (nbody_opt1-opt7):**

| File | Technique | Key Change |
|------|-----------|------------|
| nbody_opt1_fp32_rsqrt.cl | FP32 rsqrt | `rsqrt((float)dist)` vs FP64 |
| nbody_opt2_restrict.cl | restrict | Pointer aliasing hints |
| nbody_opt3_workgroup.cl | Work-group size | `reqd_work_group_size(256,1,1)` |
| nbody_opt4_unroll.cl | Loop unrolling | `#pragma unroll 4` |
| nbody_opt5_register_cache.cl | Register cache | Cache position in registers |
| nbody_opt6_scalar_accum.cl | Scalar accumulators | Separate accX, accY, accZ |
| nbody_opt7_local_memory.cl | Local memory | Tile positions in `__local` |

### General Optimization Patterns

| Pattern | When to Use | Example |
|---------|-------------|---------|
| Local memory | Memory-bound kernels | Matrix multiplication |
| Loop unrolling | Tight inner loops | `#pragma unroll 4` |
| Register caching | Frequently accessed values | `const float myX = pos[idx]` |
| fma() | Multiply-add sequences | `fma(a, b, c)` vs `a*b+c` |
| native_* functions | Platform supports it | `native_rsqrt()` on NVIDIA |
| Scalar accumulators | Vector reductions | Separate x,y,z accumulators |

---

## Running Benchmarks

### Environment Setup

```bash
# SSH to server
ssh serhii@storm
cd ~/TornadoVM-fork-2

# REQUIRED: Source environment
source setvars.sh

# Verify
echo $TORNADO_SDK
```

### Benchmark Commands

```bash
# General format
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.<BenchmarkClass> \
  <kernel.cl> [size]
```

### Complete Benchmark Workflow

```bash
# Step 1: Validate kernels produce correct results
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMultiplication1DValidator \
  kernels/matrix1d_generated.cl kernels/matrix1d_custom.cl

# Step 2: Benchmark generated kernel (SEPARATE RUN)
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMultiplication1DSingleKernelBenchmark \
  kernels/matrix1d_generated.cl 1024

# Step 3: Benchmark custom kernel (SEPARATE RUN)
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMultiplication1DSingleKernelBenchmark \
  kernels/matrix1d_custom.cl 1024

# Step 4: Compare results
# Generated: 5.234 ms, 412.5 GFLOP/s
# Custom:    4.123 ms, 523.8 GFLOP/s
# Speedup:   1.27x
```

---

## Creating New Benchmarks

### Step 1: Generate Kernel from TornadoVM

```bash
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.MyAlgorithm 2>&1 | tee output.txt
```

Extract `__kernel void ...` section to `kernels/myalgorithm_generated.cl`.

### Step 2: Create Benchmark Class

Copy template from existing benchmark (e.g., `MatrixMultiplication1DSingleKernelBenchmark.java`):

1. Update class name
2. Set correct `ENTRY_POINT` matching kernel function name
3. Set appropriate `LOCAL_WORK_SIZE` for target hardware
4. Update `AccessorParameters` to match kernel signature
5. Use correct `WorkerGrid1D` or `WorkerGrid2D`
6. Calculate domain-specific metrics (GFLOP/s, MTEPS, etc.)

### Step 3: Create Validator Class

1. Implement sequential CPU reference
2. Use smaller problem size
3. Set appropriate tolerance
4. Report detailed error information

### Step 4: Build and Test

```bash
make
java ... MyAlgorithmValidator kernels/myalgorithm_generated.cl
java ... MyAlgorithmSingleKernelBenchmark kernels/myalgorithm_generated.cl
```

---

## Results Interpretation

### Understanding Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| GFLOP/s | `(2 * N^3) / (time_ns * 1e-9) * 1e-9` | Higher = better compute throughput |
| MTEPS | `edges / (time_ns * 1e-9) * 1e-6` | Graph traversal rate |
| GB/s | `bytes / (time_ns * 1e-9) * 1e-9` | Memory bandwidth utilization |
| MPixels/s | `pixels / (time_ns * 1e-9) * 1e-6` | Image processing rate |

### Comparing Results

```
Generated kernel: 5.234 ms, 412.5 GFLOP/s
Custom kernel:    4.123 ms, 523.8 GFLOP/s

Speedup = 5.234 / 4.123 = 1.27x
Improvement = (523.8 - 412.5) / 412.5 * 100 = 27%
```

### Statistical Significance

Each benchmark reports:
- **Avg**: Mean kernel time (primary metric)
- **Min**: Best-case performance
- **Max**: Worst-case performance

For reliable comparisons:
- Ensure low variance (Max - Min) / Avg < 10%
- Run multiple times to verify consistency
- Compare Avg values, not Min

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `could not open tornado-argfile` | Environment not set | `source setvars.sh` |
| `ClassNotFoundException` | Java not compiled | `make` |
| `clCreateKernel -> -46` | Wrong entry point | Match ENTRY_POINT to kernel function |
| `file does not exist` | Wrong path | Check kernel file path |
| Large numerical differences | Missing +4 offset | Add `+ 4` to array pointer casts |
| Validation fails | Algorithm mismatch | Verify sequential reference matches kernel |

---

## References

- TornadoVM Documentation: https://tornadovm.readthedocs.io/
- OpenCL Programming Guide
- GPU Performance Optimization Best Practices
