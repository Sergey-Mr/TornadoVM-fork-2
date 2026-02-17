# TornadoVM OpenCL Kernel Optimization Project

## Project Context

This repository contains a benchmarking infrastructure for **fairly comparing TornadoVM-generated OpenCL kernels against hand-optimized custom kernels**. The goal is to measure and improve GPU kernel performance while ensuring valid, reproducible results.

### What We're Working On

1. **Generating OpenCL kernels** from TornadoVM Java code using `--printKernel`
2. **Creating hand-optimized versions** with progressive optimizations
3. **Benchmarking with KERNEL_TIME only** - isolating GPU execution from data transfer
4. **Validating correctness** before trusting benchmark results
5. **Documenting optimization techniques** that work (or don't) on different hardware

### Key Insight: Separate Runs for Fair Comparison

**CRITICAL**: Generated and optimized kernels must be benchmarked in **separate JVM executions**:

```
# Run 1 (fresh JVM, cold GPU):
java ... BenchmarkClass kernels/matrix1d_generated.cl 1024
# Result: Avg 5.234 ms, 412.5 GFLOP/s

# Run 2 (fresh JVM, cold GPU):
java ... BenchmarkClass kernels/matrix1d_custom.cl 1024
# Result: Avg 4.123 ms, 523.8 GFLOP/s
```

**Why?** Traditional benchmarks that compare kernels in the same run suffer from **ordering bias** - the second kernel always runs faster (20-30%) due to:
- GPU being in boosted clock state
- Driver optimizations cached
- Memory controllers warmed up

Our methodology ensures both kernels start from identical cold GPU state.

---

## System Information

### Remote Server (Primary)
- **Host**: `serhii@storm`
- **Location**: `~/TornadoVM-fork-2`
- **GPU**: NVIDIA GeForce RTX 4090
- **Platform**: Linux (Ubuntu)
- **Java**: GraalVM JDK 21.0.9+7.1

### Local Development (MacBook)
- **GPU**: Apple M4 (max 32 threads/work-group)
- Kernels in `kernels/macbook/` for platform-specific testing

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

## File Structure

```
TornadoVM-fork-2/
├── CLAUDE.md                    # This file - AI context and instructions
├── docs/
│   └── BENCHMARKING.md          # Detailed methodology documentation
├── kernels/                     # OpenCL kernel files
│   ├── *_generated.cl           # Extracted from TornadoVM
│   ├── *_custom.cl              # Hand-optimized versions
│   ├── opt1_restrict.cl         # Matrix optimization series
│   ├── opt2_unroll.cl
│   ├── opt3_workgroup.cl
│   ├── opt4_local_memory.cl
│   ├── opt5_local_unroll.cl
│   ├── nbody/                   # NBody optimization series
│   │   ├── nbody_opt1_fp32_rsqrt.cl
│   │   ├── nbody_opt2_restrict.cl
│   │   └── ... (7 variants)
│   ├── matrixrowmajor/          # Matrix-vector optimizations
│   └── macbook/                 # Platform-specific kernels
└── tornado-examples/src/main/java/uk/ac/manchester/tornado/examples/compute/custom/
    ├── *SingleKernelBenchmark.java   # Benchmark classes
    └── *Validator.java               # Validation classes
```

---

## Available Benchmarks

| Algorithm | Benchmark Class | Validator Class | Metric |
|-----------|-----------------|-----------------|--------|
| Matrix-Vector Row | `MatrixVectorRowMajorSingleKernelBenchmark` | `KernelValidator` | GFLOP/s |
| MatrixMul 2D Local | `MatrixMul2DLocalMemorySingleKernelBenchmark` | `MatrixMul2DLocalMemoryValidator` | GFLOP/s |
| MatrixMul 1D | `MatrixMultiplication1DSingleKernelBenchmark` | `MatrixMultiplication1DValidator` | GFLOP/s |
| NBody | `NBodySingleKernelBenchmark` | `NBodyValidator` | GFLOP/s |
| BFS | `BFSSingleKernelBenchmark` | `BFSValidator` | MTEPS |
| Mandelbrot | `MandelbrotSingleKernelBenchmark` | `MandelbrotValidator` | MPixels/s |
| MonteCarlo | `MonteCarloSingleKernelBenchmark` | `MonteCarloValidator` | MSamples/s |
| BlackScholes | `BlackScholesSingleKernelBenchmark` | `BlackScholesValidator` | MOptions/s |
| BlurFilter | `BlurFilterSingleKernelBenchmark` | `BlurFilterValidator` | MPixels/s |
| ReductionAddFloats | `ReductionAddFloatsSingleKernelBenchmark` | `ReductionAddFloatsValidator` | GB/s |
| FlashAttention | `FlashAttentionSingleKernelBenchmark` | `FlashAttentionValidator` | GFLOP/s |

---

## How to Run Benchmarks

### General Command Format
```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.<BenchmarkClass> \
  <kernel.cl> [size_params]
```

### Example: Matrix Multiplication 1D
```bash
# Run benchmark
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMultiplication1DSingleKernelBenchmark \
  kernels/matrix1d_generated.cl 1024

# Validate first (always validate before trusting benchmark)
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMultiplication1DValidator \
  kernels/matrix1d_generated.cl kernels/matrix1d_custom.cl
```

### Example: NBody
```bash
# Benchmark
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.NBodySingleKernelBenchmark \
  kernels/nbody/nbody_opt7_local_memory.cl 16384

# Validate
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.NBodyValidator \
  kernels/nbody_optimized.cl --bodies=1024
```

---

## How to Generate Kernels from TornadoVM

Use `--printKernel` flag to see generated OpenCL:
```bash
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.<ExampleClass> 2>&1 | tee output.txt
```

Extract the `__kernel void ...` section and save to `kernels/<name>_generated.cl`.

### Kernel Entry Points

| Algorithm | Entry Point Function |
|-----------|---------------------|
| Matrix-Vector Row | `matrixVectorRowMajor` or `matrixVectorGeneric` |
| MatrixMul 2D Local | `matrixMultiplication` |
| MatrixMul 1D | `matrixMultiplication` |
| NBody | `nBody` |
| BFS | `runBFS` |
| Mandelbrot | `mandelbrotTornado` |
| FlashAttention | `processHeadsFlashAttention` |

---

## TornadoVM Kernel Signature

All kernels MUST follow this signature pattern (TornadoVM internal parameters):

```c
__kernel void myFunction(
    __global long *_kernel_context,      // TornadoVM internal - contains N
    __constant uchar *_constant_region,  // TornadoVM internal
    __local uchar *_local_region,        // TornadoVM internal
    __global int *_atomics,              // TornadoVM internal
    // User parameters start here:
    __global uchar *inputArray,          // Cast to actual type inside
    __global uchar *outputArray,
    __private int size
)
```

Access array data like this:
```c
const int N = (int)_kernel_context[0];
__global const float *a = ((__global const float *)inputArray) + 4;  // +4 offset!
```

---

## Optimization Techniques Tested

### Matrix Multiplication (Progressive)
1. **opt1_restrict.cl** - `restrict` keyword for pointer aliasing hints
2. **opt2_unroll.cl** - 4x loop unrolling with `#pragma unroll`
3. **opt3_workgroup.cl** - Explicit `__attribute__((reqd_work_group_size(...)))`
4. **opt4_local_memory.cl** - Local memory tiling (~22% improvement)
5. **opt5_local_unroll.cl** - Combines local memory + unrolling

### NBody (Progressive)
1. **nbody_opt1_fp32_rsqrt.cl** - FP32 `rsqrt()` instead of FP64
2. **nbody_opt2_restrict.cl** - `restrict` keyword
3. **nbody_opt3_workgroup.cl** - Explicit work-group size
4. **nbody_opt4_unroll.cl** - Loop unrolling
5. **nbody_opt5_register_cache.cl** - Caching in registers
6. **nbody_opt6_scalar_accum.cl** - Separate scalar accumulators
7. **nbody_opt7_local_memory.cl** - Local memory tiling

### Key Optimization Patterns
- **Local memory tiling** - Biggest impact on memory-bound kernels
- **Loop unrolling** - Reduces branch overhead, improves pipelining
- **Register caching** - Cache frequently accessed values
- **fma()** - Fused multiply-add for better accuracy and performance
- **native_* functions** - Hardware-accelerated math (platform-specific)

---

## Creating New Benchmarks

### Benchmark Class Template
```java
public class MyAlgorithmSingleKernelBenchmark {
    private static final int WARM_UP_ITERATIONS = 50;
    private static final int BENCHMARK_ITERATIONS = 100;
    private static final String ENTRY_POINT = "myKernelFunction";

    public static void main(String[] args) {
        // 1. Parse args, setup data
        // 2. Create AccessorParameters matching kernel signature
        // 3. Build TaskGraph with prebuiltTask()
        // 4. Warmup loop (no profiler)
        // 5. Measurement loop with ProfilerMode.SILENT
        //    - Collect getDeviceKernelTime() only
        // 6. Calculate statistics and domain-specific metrics
    }
}
```

### Validator Class Template
```java
public class MyAlgorithmValidator {
    private static final float TOLERANCE = 1e-4f;

    public static void main(String[] args) {
        // 1. Smaller problem size than benchmark
        // 2. Compute sequential reference on CPU
        // 3. Run kernel once
        // 4. Compare with tolerance
        // 5. Report pass/fail with error details
    }
}
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `could not open tornado-argfile` | Run `source setvars.sh` |
| `ClassNotFoundException` | Run `make` to rebuild |
| `clCreateKernel -> Returned: -46` | Entry point name mismatch |
| `file does not exist: kernels/...` | Check kernel file path |
| Large numerical differences | Check `+4` offset for float arrays |

---

## Quick Reference

```bash
# After SSH to server
cd ~/TornadoVM-fork-2
source setvars.sh

# Rebuild after Java changes
make

# Run any benchmark
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.<ClassName> \
  kernels/<kernel>.cl [params]

# Generate kernel from TornadoVM example
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.<Example>
```

---

## For AI Assistants

When helping with this project:

1. **Always validate first** - Run validator before trusting benchmark results
2. **One kernel per run** - Don't compare kernels in same execution
3. **Preserve TornadoVM signature** - Keep the 4 internal parameters
4. **Use +4 float offset** - Array data starts at index 4, not 0
5. **Match entry point names** - Benchmark ENTRY_POINT must match kernel function name
6. **Consider work-group limits** - Apple M4 max is 32; NVIDIA can do 256+
7. **Test optimizations progressively** - Isolate each optimization's impact
