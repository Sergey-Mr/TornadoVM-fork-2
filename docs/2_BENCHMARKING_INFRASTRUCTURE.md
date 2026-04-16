# TornadoVM Benchmarking Infrastructure

> Detailed documentation of the custom benchmarking framework for fair GPU kernel performance comparison.

## Table of Contents

1. [Why a Custom Benchmarking Approach](#why-a-custom-benchmarking-approach)
2. [The Separate-Runs Methodology](#the-separate-runs-methodology)
3. [Measurement: KERNEL_TIME Only](#measurement-kernel_time-only)
4. [Benchmark Architecture](#benchmark-architecture)
5. [The prebuiltTask Mechanism](#the-prebuilttask-mechanism)
6. [Grid Configuration](#grid-configuration)
7. [Validation Framework](#validation-framework)
8. [Benchmark Catalogue](#benchmark-catalogue)
9. [Kernel Files Organisation](#kernel-files-organisation)
10. [Platform-Specific Considerations](#platform-specific-considerations)
11. [Build and Run](#build-and-run)

---

## Why a Custom Benchmarking Approach

### The Problem with Standard Benchmarks

Standard GPU benchmarking approaches have a fundamental flaw when comparing two kernels: **ordering bias**. If you run Kernel A then Kernel B in the same process, Kernel B consistently runs 20-30% faster because:

1. **GPU clock boost** - Modern GPUs dynamically increase clock frequency under sustained load. By the time Kernel B runs, the GPU is already in a boosted state.
2. **Driver caches** - OpenCL/CUDA drivers cache compilation artifacts, memory allocation metadata, and scheduling decisions. The second kernel benefits from a warmed driver.
3. **Memory controller warm-up** - GPU memory controllers and DRAM banks are in an active state, reducing initial latency.
4. **JIT compilation** - TornadoVM's JIT compiler is already warm, and class loading is complete.

This means a naive "run both kernels in a loop" benchmark would systematically make the second kernel look better, regardless of its actual quality.

### The Problem with End-to-End Timing

Another common mistake is measuring **total execution time** including data transfers:

```
Total time = Data transfer (CPU→GPU) + Kernel execution + Data transfer (GPU→CPU)
```

For small kernels, data transfer dominates (>80% of total time). An optimization that makes the kernel 2x faster might only show as 10% improvement in total time, masking the real benefit. Conversely, a kernel that "looks" similar in total time might have a severely degraded compute phase hidden by fast transfers.

### Our Solution

This project addresses both problems:

1. **Separate JVM executions** - Each kernel runs in its own fresh JVM process with a cold GPU
2. **Kernel-time-only measurement** - Using TornadoVM's profiler to isolate GPU kernel execution time

---

## The Separate-Runs Methodology

### Principle

```
Run 1 (fresh JVM, cold GPU):
┌─────────────────────────────────────────────────────────┐
│  JVM starts → TornadoVM init → Warmup (50 iter) →       │
│  Benchmark (100 iter, kernel time only) → Report stats   │
│                                                          │
│  Kernel: matrix1d_generated.cl                           │
│  Result: Avg 5.234 ms, 412.5 GFLOP/s                    │
└─────────────────────────────────────────────────────────┘

Run 2 (fresh JVM, cold GPU):
┌─────────────────────────────────────────────────────────┐
│  JVM starts → TornadoVM init → Warmup (50 iter) →       │
│  Benchmark (100 iter, kernel time only) → Report stats   │
│                                                          │
│  Kernel: matrix1d_custom.cl                              │
│  Result: Avg 4.123 ms, 523.8 GFLOP/s                    │
└─────────────────────────────────────────────────────────┘

Comparison: custom is 1.27x faster (derived, not measured in-process)
```

### Why This Works

Both kernels start from identical conditions:
- Fresh JVM (no JIT compilation residue)
- Cold GPU (clocks at base frequency, caches empty)
- Clean driver state (no cached compilation artifacts)
- Same data (same random seed, deterministic initialization)

The only variable is the kernel code itself.

### Implementation Pattern

Every `SingleKernelBenchmark` class follows this structure:

```java
public static void main(String[] args) {
    // 1. Parse command-line args (kernel path, problem size)
    String kernelPath = args[0];
    int size = Integer.parseInt(args[1]);

    // 2. Deterministic data initialisation (same seed = same data)
    Random random = new Random(42);
    FloatArray data = new FloatArray(size * size);
    fillRandom(data, random);

    // 3. Setup TornadoVM TaskGraph with prebuiltTask
    TaskGraph graph = new TaskGraph("s0")
        .transferToDevice(DataTransferMode.FIRST_EXECUTION, data)
        .prebuiltTask("t0", ENTRY_POINT, kernelPath, accessors)
        .transferToHost(DataTransferMode.EVERY_EXECUTION, result);

    // 4. Warmup phase (50 iterations, no profiling)
    for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
        plan.execute();
    }

    // 5. Measurement phase (100 iterations, kernel time only)
    ArrayList<Long> kernelTimes = new ArrayList<>();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        TornadoExecutionResult result = plan
            .withProfiler(ProfilerMode.SILENT)
            .execute();
        long kernelTime = result.getProfilerResult().getDeviceKernelTime();
        kernelTimes.add(kernelTime);
    }

    // 6. Statistics and reporting
    LongSummaryStatistics stats = kernelTimes.stream()
        .mapToLong(Long::longValue)
        .summaryStatistics();
    System.out.printf("Avg: %.3f ms%n", stats.getAverage() / 1_000_000.0);
}
```

---

## Measurement: KERNEL_TIME Only

### What `getDeviceKernelTime()` Measures

TornadoVM's profiler measures **only the GPU kernel execution time** via OpenCL/CUDA event timing:

```
                    NOT MEASURED                  MEASURED              NOT MEASURED
               ◄────────────────────►  ◄───────────────────────►  ◄─────────────────►
╔══════════════╦══════════════════════╦═════════════════════════╦══════════════════════╗
║   JVM Code   ║   Copy In (H→D)      ║    GPU Kernel Exec      ║   Copy Out (D→H)     ║
║  (setup,     ║   (host to device    ║    (actual compute on   ║   (device to host    ║
║   overhead)  ║    memory transfer)  ║     GPU cores)          ║    memory transfer)  ║
╚══════════════╩══════════════════════╩═════════════════════════╩══════════════════════╝
```

This is achieved through hardware-level event timing:
- **OpenCL:** `clGetEventProfilingInfo(CL_PROFILING_COMMAND_END - CL_PROFILING_COMMAND_START)`
- **CUDA/PTX:** `cuEventElapsedTime(start, end)`

### Why KERNEL_TIME Only?

| Metric | What it Shows | Problem |
|--------|--------------|---------|
| Total execution | Kernel + transfers + JVM overhead | Transfer-dominated for small kernels |
| Kernel + transfer | Compute + data movement | Conflates two independent concerns |
| **Kernel time only** | **Pure GPU compute performance** | **Isolates the optimisation target** |

Since our optimisations target the kernel code (loop unrolling, tiling, vectorisation), we need to measure the kernel code's execution in isolation. Data transfer time is irrelevant - it's the same for both the generated and optimised kernel (same data, same device).

### ProfilerMode.SILENT

```java
TornadoExecutionResult result = plan
    .withProfiler(ProfilerMode.SILENT)
    .execute();
```

`ProfilerMode.SILENT` enables profiling instrumentation but suppresses console output. This gives us access to `getDeviceKernelTime()` without cluttering the benchmark output with per-iteration profiler dumps.

---

## Benchmark Architecture

### Class Hierarchy

```
                    ┌───────────────────────────┐
                    │   SingleKernelBenchmark    │
                    │   (one kernel per run)     │
                    │                           │
                    │ - WARM_UP_ITERATIONS = 50  │
                    │ - BENCHMARK_ITERATIONS=100 │
                    │ - ENTRY_POINT = "..."      │
                    │ - prebuiltTask()           │
                    │ - ProfilerMode.SILENT      │
                    └───────────┬───────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
   ┌──────┴──────┐    ┌───────┴───────┐    ┌───────┴───────┐
   │  OpenCL     │    │   PTX         │    │   MCP         │
   │  Benchmarks │    │   Benchmarks  │    │   Benchmarks  │
   │             │    │               │    │               │
   │ .cl files   │    │ .ptx files    │    │ HTTP call to  │
   │ WorkerGrid  │    │ WorkerGrid    │    │ MCP server    │
   │             │    │               │    │               │
   └─────────────┘    └───────────────┘    └───────────────┘
```

### Data Flow Through a Benchmark

```
Command Line Args
    │
    ├── kernel path: "kernels/matrix1d_custom.cl"
    ├── size: 1024
    │
    v
┌───────────────────┐
│  Data Init         │
│  Random(seed=42)   │──> Deterministic: same data every run
│  FloatArray alloc  │
└────────┬──────────┘
         │
         v
┌───────────────────┐
│  TaskGraph Setup   │
│                   │
│  transferToDevice  │──> FIRST_EXECUTION (one-time copy)
│  prebuiltTask      │──> Loads kernel from .cl/.ptx file
│  transferToHost    │──> EVERY_EXECUTION (copy result back)
└────────┬──────────┘
         │
         v
┌───────────────────┐
│  GridScheduler     │
│                   │
│  WorkerGrid1D/2D   │──> Defines global and local work sizes
│  setLocalWork()    │──> Sets work-group dimensions
└────────┬──────────┘
         │
         v
┌───────────────────┐     ┌───────────────────┐
│  Warmup Phase      │     │  Benchmark Phase   │
│  50 iterations     │────>│  100 iterations    │
│  No profiling      │     │  ProfilerMode.SILENT│
│  GPU reaches       │     │  Collect kernel    │
│  steady state      │     │  time per iteration│
└───────────────────┘     └────────┬──────────┘
                                    │
                                    v
                          ┌───────────────────┐
                          │  Statistics         │
                          │  Avg, Min, Max (ms) │
                          │  Domain metric      │
                          │  (GFLOP/s, etc.)    │
                          └───────────────────┘
```

---

## The prebuiltTask Mechanism

### What is prebuiltTask?

TornadoVM normally compiles Java code to GPU kernels at runtime. `prebuiltTask` bypasses this: it loads a pre-written kernel from a file and executes it directly on the GPU.

This is essential for our benchmarking because:
1. We need to run **specific** kernel versions (generated vs. hand-optimized)
2. We need to control the **exact kernel code** executing on the GPU
3. We need to test kernels that were optimized **outside TornadoVM** (by the MCP server)

### How it Works

```java
// Define parameter mapping
AccessorParameters accessors = new AccessorParameters(4);
accessors.set(0, matrixA, Access.READ_ONLY);    // Maps to kernel param 0
accessors.set(1, matrixB, Access.READ_ONLY);     // Maps to kernel param 1
accessors.set(2, matrixC, Access.WRITE_ONLY);    // Maps to kernel param 2
accessors.set(3, Integer.valueOf(size), Access.NONE);  // Scalar param

// Register the prebuilt task
TaskGraph graph = new TaskGraph("s0")
    .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA, matrixB)
    .prebuiltTask("t0",              // Task ID
                  "matrixMultiplication",  // Entry point (kernel function name)
                  "kernels/matrix1d_custom.cl",  // Kernel file path
                  accessors)          // Parameter bindings
    .transferToHost(DataTransferMode.EVERY_EXECUTION, matrixC);
```

### TornadoVM Kernel Signature Convention

All kernels (whether generated by TornadoVM or hand-written) follow this signature pattern:

```c
__kernel void functionName(
    __global long *_kernel_context,      // TornadoVM internal (contains N at index 0)
    __constant uchar *_constant_region,  // TornadoVM internal
    __local uchar *_local_region,        // TornadoVM internal
    __global int *_atomics,              // TornadoVM internal
    // User parameters start here:
    __global uchar *arrayA,              // Cast to actual type inside kernel
    __global uchar *arrayB,
    __private int size
)
```

**The +4 offset rule:** TornadoVM arrays have a 4-element header (16 bytes for float). Data starts at index 4:
```c
__global float *a = ((__global float *)arrayA) + 4;
```

---

## Grid Configuration

### 1D Grid (Vector/Element-wise Operations)

```java
WorkerGrid1D worker = new WorkerGrid1D(totalElements);
worker.setLocalWork(256, 1, 1);
GridScheduler scheduler = new GridScheduler("s0.t0", worker);
```

Used by: NBody, BlackScholes, MonteCarlo, Reduction

### 2D Grid (Matrix Operations)

```java
WorkerGrid2D worker = new WorkerGrid2D(rows, cols);
worker.setLocalWork(16, 16, 1);  // 16x16 = 256 threads per work-group
GridScheduler scheduler = new GridScheduler("s0.t0", worker);
```

Used by: MatrixMul 2D, Mandelbrot, BlurFilter, BFS

### Work-Group Size Considerations

| Device | Max Work-Group | Optimal Range | Reason |
|--------|---------------|---------------|--------|
| NVIDIA RTX 4090 | 1024 | 256-512 | Warp size 32, good occupancy |
| Apple M4 | 1024 (theoretical) | 32-256 | Register pressure at higher counts |
| AMD RDNA3 | 1024 | 256 | Wavefront size 32 |

---

## Validation Framework

### Why Validate?

An "optimised" kernel that produces wrong results is worse than useless. Every kernel must pass validation before its benchmark results are trusted.

### Validation Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    VALIDATION FLOW                               │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                    │
│  │  CPU Sequential   │    │  GPU Kernel       │                   │
│  │  Reference         │    │  (under test)     │                   │
│  │                   │    │                   │                   │
│  │  Same input data  │    │  Same input data  │                   │
│  │  Known-correct    │    │  prebuiltTask()   │                   │
│  │  implementation   │    │                   │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │                       │                              │
│           v                       v                              │
│  ┌──────────────────────────────────────────┐                    │
│  │  Element-by-Element Comparison            │                    │
│  │                                          │                    │
│  │  for (int i = 0; i < N; i++) {            │                   │
│  │      float diff = |gpu[i] - cpu[i]|;     │                   │
│  │      if (diff > TOLERANCE) {              │                   │
│  │          report MISMATCH at index i;      │                   │
│  │      }                                    │                   │
│  │  }                                        │                   │
│  └──────────────────────────────────────────┘                    │
│                                                                  │
│  Result: PASS (all elements within tolerance)                    │
│      or: FAIL (N mismatches, first at index X, max error Y)     │
└────────────────────────────────────────────────────────────────┘
```

### Tolerance Values

Different algorithms accumulate floating-point error differently:

| Algorithm | Tolerance | Reason |
|-----------|----------|--------|
| Matrix operations | 1e-3f | Accumulated FP32 multiply-add error |
| NBody | 0.1f | Physics simulation error accumulates across timesteps |
| Pi/MonteCarlo | 0.001f | Statistical convergence, not exact match |
| BlackScholes | 1e-4f | Numerical stability of exp/log operations |
| BlurFilter | 1e-3f | Floating-point convolution error |

### Validator Classes

Each benchmark has a corresponding validator:

| Benchmark | Validator | Special Logic |
|-----------|-----------|---------------|
| MatrixMultiplication1D | MatrixMultiplication1DValidator | Sequential O(n^3) reference |
| MatrixMul2DLocalMemory | MatrixMul2DLocalMemoryValidator | 2D tiled sequential reference |
| NBody | NBodyValidator | Validates position AND velocity |
| BFS | BFSValidator | Graph level validation, reachable nodes |
| FlashAttention | FlashAttentionValidator | Multi-head attention reference |
| BlackScholes | BlackScholesValidator | Black-Scholes formula reference |
| MonteCarlo | MonteCarloValidator | Statistical Pi approximation |
| BlurFilter | BlurFilterValidator | 2D convolution reference |
| Reduction | ReductionAddFloatsValidator | Simple sum reference |
| Mandelbrot | MandelbrotValidator | Iteration count comparison |

---

## Benchmark Catalogue

### OpenCL Benchmarks

| # | Algorithm | Class | Entry Point | Grid | Metric | Default Size |
|---|-----------|-------|-------------|------|--------|-------------|
| 1 | Matrix-Vector Row | MatrixVectorRowMajorSingleKernelBenchmark | matrixVectorRowMajor | 1D | GFLOP/s | 8192x2048 |
| 2 | MatrixMul 2D Local | MatrixMul2DLocalMemorySingleKernelBenchmark | matrixMultiplication | 2D | GFLOP/s | 1024 |
| 3 | MatrixMul 1D | MatrixMultiplication1DSingleKernelBenchmark | matrixMultiplication | 2D | GFLOP/s | 1024 |
| 4 | NBody | NBodySingleKernelBenchmark | nBody | 1D | GFLOP/s | 16384 |
| 5 | BFS | BFSSingleKernelBenchmark | runBFS | 2D | MTEPS | 4096 |
| 6 | Mandelbrot | MandelbrotSingleKernelBenchmark | mandelbrotTornado | 2D | MPixels/s | 4096 |
| 7 | MonteCarlo | MonteCarloSingleKernelBenchmark | computeMontecarlo | 1D | MSamples/s | 16M |
| 8 | BlackScholes | BlackScholesSingleKernelBenchmark | blackScholesKernel | 1D | MOptions/s | 4M |
| 9 | BlurFilter | BlurFilterSingleKernelBenchmark | compute | 2D | MPixels/s | 2048 |
| 10 | Reduction | ReductionAddFloatsSingleKernelBenchmark | reductionAddFloats | 1D | GB/s | 16M |
| 11 | FlashAttention | FlashAttentionSingleKernelBenchmark | processHeadsFlashAttention | 1D | GFLOP/s | Llama-3 dims |

### PTX Benchmarks

| # | Algorithm | Class | Notes |
|---|-----------|-------|-------|
| 1 | Matrix-Vector Row | MatrixVectorRowMajorPTXBenchmark | 6 parameters including localWorkGroupSize |
| 2 | MatrixMul 2D Local | MatrixMul2DLocalMemoryPTXBenchmark | 16x16 tile size |
| 3 | MatrixMul 1D | MatrixMultiplication1DPTXBenchmark | 2D grid setup |
| 4 | NBody | NBodyPTXBenchmark | Long auto-generated entry point name |
| 5 | BFS | BFSPTXBenchmark | Graph traversal |
| 6 | PiComputation | PiComputationPTXBenchmark | 16M terms, reduction |
| 7 | MonteCarlo | MonteCarloPTXBenchmark | Random sampling |
| 8 | FlashAttention | FlashAttentionPTXBenchmark | Multi-head attention |

### Domain-Specific Metrics

Each benchmark reports a domain-meaningful metric rather than raw milliseconds:

```
GFLOP/s = (FLOPs per iteration) / (kernel_time_ns) * 1e-9
          For MatrixMul: 2 * N^3 FLOPs
          For NBody: 20 * N^2 FLOPs (per timestep)

MPixels/s = (width * height) / (kernel_time_ns) * 1e-3

MSamples/s = samples / (kernel_time_ns) * 1e-3

MTEPS = (edges_traversed) / (kernel_time_ns) * 1e-3
```

---

## Kernel Files Organisation

```
kernels/
├── matrix1d_generated.cl          # TornadoVM-generated (baseline)
├── matrix1d_custom.cl             # Hand-optimized version
├── matrix2d_generated.cl
├── matrix2d_custom.cl
├── matrixvectorrow_custom.cl
├── matrixvectorrow_optimized.cl
├── blackscholes_custom.cl
├── flashattention_custom.cl
├── reductionsandflaots_custom.cl
├── nbody_fast.cl                  # Fast variants
├── nbody_fast_ilp.cl              # ILP-optimized
├── nbody_optimized.cl             # Fully optimized
│
├── nbody/                         # Progressive optimization study
│   ├── nbody_original.cl          # Baseline
│   ├── nbody_opt1_fp32_rsqrt.cl   # FP32 + rsqrt
│   ├── nbody_opt2_restrict.cl     # + restrict
│   ├── nbody_opt3_workgroup.cl    # + explicit work-group
│   ├── nbody_opt4_unroll.cl       # + loop unrolling
│   ├── nbody_opt5_register_cache.cl # + register caching
│   ├── nbody_opt6_scalar_accum.cl # + scalar accumulators
│   └── nbody_opt7_local_memory.cl # + local memory tiling
│
├── matrixrowmajor/                # Matrix-vector optimizations
│   ├── cleaninnerloop.cl
│   ├── hoistedrowbase.cl
│   ├── optimized_32threads.cl
│   ├── optimized_rtx4090.cl       # RTX 4090-specific
│   ├── reductioncleanup.cl
│   └── typedpointers.cl
│
├── macbook/                       # Apple M4-specific kernels
│   ├── bfs_custom.cl
│   ├── bfs_generated.cl
│   ├── matrix2d_custom.cl
│   ├── matrix2d_generated.cl
│   ├── matrixrowmajor_custom.cl
│   └── matrixrowmajor_generated.cl
│
└── ptx/                           # NVIDIA PTX kernels
    ├── matrixvector_generated.ptx
    ├── matrix2d_generated.ptx
    ├── matrix1d_generated.ptx
    ├── nbody_generated.ptx
    ├── bfs_generated.ptx
    ├── picomputation_generated.ptx
    └── flashattention_generated.ptx
```

---

## Platform-Specific Considerations

### Apple M4 (MacBook)

- **Max work-group size:** 1024 (theoretical), but 32-256 optimal
- **SIMD width:** 32 threads
- **Unified memory:** No separate CPU/GPU memory, transfers are cheap
- **Limitation:** Only OpenCL supported (no PTX/CUDA)
- **Kernels in:** `kernels/macbook/`

### NVIDIA RTX 4090 (Server "storm")

- **Max work-group size:** 1024
- **Warp size:** 32
- **Shared memory:** 48KB+ per SM
- **Compute units:** 128 SMs
- **Supports:** Both OpenCL and PTX backends
- **Kernels in:** `kernels/` (root) and `kernels/ptx/`

### PTX-Specific Requirements

PTX kernels require header stripping before use with `prebuiltTask`:

```bash
# TornadoVM adds .version/.target/.address_size headers automatically
# Having them in the file causes CUDA error 218 (INVALID_PTX)
sed -i '/^\.version/d; /^\.target/d; /^\.address_size/d' kernel.ptx
```

PTX entry point names are auto-generated and much longer:
```
s0_t0_nbody_2048_arrays_floatarray_arrays_floatarray_0_005_500_0
```

---

## Build and Run

### Environment Setup

```bash
# On the server (storm):
export JAVA_HOME=$HOME/graalvm-jdk-21.0.9+7.1
export PATH=$JAVA_HOME/bin:$PATH
cd ~/TornadoVM-fork-2
source setvars.sh

# Verify:
echo $TORNADO_SDK
tornado --devices
```

### Building

```bash
make
# Compiles all benchmark classes to bin/examples/
```

### Running a Benchmark

```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMultiplication1DSingleKernelBenchmark \
  kernels/matrix1d_generated.cl 1024
```

### Running a Validator

```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMultiplication1DValidator \
  kernels/matrix1d_generated.cl kernels/matrix1d_custom.cl
```

### Standard Benchmark Workflow

```bash
# 1. Always validate FIRST
java ... Validator kernel_generated.cl kernel_custom.cl

# 2. Benchmark generated kernel (fresh JVM)
java ... SingleKernelBenchmark kernel_generated.cl [size]

# 3. Benchmark custom kernel (fresh JVM)
java ... SingleKernelBenchmark kernel_custom.cl [size]

# 4. Compare results (manual or script)
# speedup = generated_time / custom_time
```
