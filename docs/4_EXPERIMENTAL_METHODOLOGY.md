# Experimental Methodology: From Manual Proof of Concept to Automated Optimisation

> How the optimisation library was built, validated, and populated — and how this manual process became the blueprint for the MCP server.  
> *This document serves as a foundation for the dissertation chapter on experimental methodology.*

## Table of Contents

1. [Overview: The Manual-First Approach](#overview-the-manual-first-approach)
2. [Starting Point: TornadoVM Example Kernels](#starting-point-tornadovm-example-kernels)
3. [Proof of Concept: Manual Optimisation with Claude](#proof-of-concept-manual-optimisation-with-claude)
4. [Building the Benchmarking Infrastructure](#building-the-benchmarking-infrastructure)
5. [How prebuiltTask Works: Two Independent Compilation Paths](#how-prebuilttask-works-two-independent-compilation-paths)
6. [Measuring Kernel Time: Why and How](#measuring-kernel-time-why-and-how)
7. [The Separate-Runs Methodology and Its Trade-offs](#the-separate-runs-methodology-and-its-trade-offs)
8. [Validation: Only Correct Kernels Are Accepted](#validation-only-correct-kernels-are-accepted)
9. [Cross-Platform Kernel Development](#cross-platform-kernel-development)
10. [Populating the Optimisation Library](#populating-the-optimisation-library)
11. [The Library Creator Application](#the-library-creator-application)
12. [From Manual Process to MCP Server](#from-manual-process-to-mcp-server)

---

## Overview: The Manual-First Approach

Before building any automated system, the entire optimisation pipeline was executed **by hand**. This was deliberate: the MCP server is a formalisation of a process that was first proven to work manually. Every component of the MCP server — the analysis step, the reference lookup, the planning, the code generation, the feedback loop — maps directly to something the author did manually during the proof-of-concept phase.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                THE MANUAL WORKFLOW (later automated by MCP)               │
│                                                                           │
│  1. Take a TornadoVM-generated kernel                                     │
│        ↓                                                                  │
│  2. Study the kernel, identify bottlenecks          (→ MCP Step 1)        │
│        ↓                                                                  │
│  3. Look at similar optimisations for reference     (→ MCP Step 2 / RAG)  │
│        ↓                                                                  │
│  4. Plan which optimisations to apply               (→ MCP Step 3)        │
│        ↓                                                                  │
│  5. Prompt Claude Opus 4.6 to generate              (→ MCP Step 4)        │
│     the optimised kernel                                                  │
│        ↓                                                                  │
│  6. Validate correctness against CPU reference      (→ TornadoVM validator)│
│        ↓                                                                  │
│  7. Benchmark in separate JVM runs                  (→ TornadoVM benchmark)│
│        ↓                                                                  │
│  8. If wrong or slower → analyse failure,           (→ MCP feedback loop)  │
│     re-prompt Claude with feedback                                        │
│        ↓                                                                  │
│  9. Upload successful pair to vector database       (→ Library Creator)    │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Starting Point: TornadoVM Example Kernels

TornadoVM ships with a set of example applications that demonstrate heterogeneous programming in Java. These examples provided the raw material for the optimisation study:

| Algorithm | TornadoVM Class | Characteristics |
|-----------|----------------|-----------------|
| Matrix Multiplication 1D | `MatrixMultiplication1D` | Flat 1D array, O(n^3), memory-bound |
| Matrix Multiplication 2D | `MatrixMul2DLocalMemory` | 2D tiling, local memory, compute-bound |
| Matrix-Vector Row Major | `MatrixVector` | 1D reduction per row, memory-bound |
| NBody Simulation | `NBody` | O(n^2) pairwise physics, compute-bound |
| BFS Graph Traversal | `BFS` | Irregular memory access, data-dependent |
| Mandelbrot Fractal | `Mandelbrot` | Per-pixel compute, embarrassingly parallel |
| MonteCarlo Pi | `MonteCarlo` | Random sampling, reduction |
| BlackScholes Pricing | `BlackScholes` | Element-wise math, compute-bound |
| Blur Filter | `BlurFilter` | 2D stencil convolution, memory-bound |
| Reduction (Sum) | `ReductionAddFloats` | Tree reduction, synchronisation-heavy |
| Flash Attention | *(custom)* | Multi-head attention (Llama-3 8B dimensions) |

### Extracting Generated Kernels

For each algorithm, the TornadoVM-generated kernel was extracted using the `--printKernel` flag:

```bash
tornado --printKernel -m tornado.examples/uk.ac.manchester.tornado.examples.compute.NBody \
  --params="2048 1" 2>&1 | tee nbody_output.txt
```

This produces the OpenCL (or PTX) kernel that TornadoVM's JIT compiler generates from the Java source. The kernel was saved as `kernels/<algorithm>_generated.cl`.

For PTX kernels on NVIDIA hardware, an additional step was required — stripping the `.version`, `.target`, and `.address_size` headers that TornadoVM's `prebuiltTask` adds automatically:

```bash
sed -i '/^\.version/d; /^\.target/d; /^\.address_size/d' kernels/ptx/nbody_generated.ptx
```

These generated kernels serve as the **baseline**: the unoptimised kernel against which all optimisations are measured.

---

## Proof of Concept: Manual Optimisation with Claude

Before building any infrastructure, the first question was: **can an LLM actually produce useful GPU kernel optimisations?**

### The Manual Process

Using Claude Opus 4.6 via the chat interface (not the API), the following workflow was repeated for each algorithm:

1. **Paste the generated kernel** into the Claude conversation
2. **Explain the context**: "This is a TornadoVM-generated OpenCL kernel for matrix multiplication. It runs on an RTX 4090. The kernel has a +4 offset for float arrays. Please optimise it for performance."
3. **Iterate with feedback**: When the first attempt had errors or was slower, provide the error message or benchmark results and ask Claude to try a different approach
4. **Extract the working kernel** once it passed validation

This manual phase was critical for several reasons:

- It **validated that LLM-based kernel optimisation was feasible** at all
- It revealed the **common failure modes** (signature changes, missing +4 offset, grid-stride incompatibility with tiling) that later became constraints in the system prompt
- It produced the **first set of optimisation pairs** that seeded the vector database
- It established the **domain knowledge** needed to design effective prompts

### What Worked and What Failed

During manual prompting, certain patterns emerged:

| Technique | Success Rate | Notes |
|-----------|-------------|-------|
| Local memory tiling | High | Biggest performance gains, but requires removing grid-stride loops |
| Loop unrolling | Moderate | Often helps, but aggressive unrolling causes register pressure |
| `restrict` keyword | High | Easy for LLM to add correctly, small benefit |
| FP32 instead of FP64 | High | Significant on consumer GPUs, but must preserve correctness |
| `fma()` / `native_rsqrt()` | High | Hardware-accelerated, easy substitution |
| Work-group size tuning | Mixed | LLM often picks incorrect sizes for the device |
| Signature changes | **Common failure** | LLM would add parameters or rename the function |
| Missing +4 offset | **Common failure** | LLM would forget the TornadoVM array header |
| Grid-stride + tiling | **Common failure** | LLM would try both simultaneously (incompatible) |

These observations directly informed the system prompt constraints and the feedback loop templates in the MCP server.

---

## Building the Benchmarking Infrastructure

### The Problem: How to Fairly Compare Two Kernels

To know whether an optimisation actually helps, two kernels must be compared under identical conditions. This requires:

1. **Isolating kernel execution time** from data transfer and JVM overhead
2. **Eliminating ordering bias** from GPU warmup effects
3. **Validating correctness** before trusting any performance numbers
4. **Repeatable results** with deterministic data initialisation

### The Solution: Custom Benchmark and Validator Classes

For each algorithm, two Java classes were written:

**`<Algorithm>SingleKernelBenchmark.java`** — Runs a single kernel file, measures kernel execution time only, reports statistics and domain-specific metrics.

**`<Algorithm>Validator.java`** — Runs one or more kernel files, compares GPU output against a sequential CPU reference implementation, reports pass/fail with mismatch details.

These classes rely heavily on TornadoVM's `prebuiltTask` mechanism and `ProfilerMode.SILENT` — understanding how these work is essential to evaluating the correctness of the benchmarking methodology.

---

## How prebuiltTask Works: Two Independent Compilation Paths

### The Mechanism

TornadoVM normally compiles Java code to GPU kernels at runtime through its JIT compiler. `prebuiltTask` **bypasses the JIT** and loads a pre-written kernel directly from a file:

```java
TaskGraph graph = new TaskGraph("s0")
    .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA, matrixB)
    .prebuiltTask("t0", "matrixMultiplication", "kernels/matrix2d_custom.cl", accessors)
    .transferToHost(DataTransferMode.EVERY_EXECUTION, matrixC);
```

### What Happens at Runtime

```
prebuiltTask("t0", "matrixMultiplication", "kernels/matrix2d_custom.cl", accessors)
        │
        v
┌─────────────────────────────────────────────────────────────┐
│  1. READ FILE                                                │
│     Files.readAllBytes("kernels/matrix2d_custom.cl")         │
│     → raw OpenCL source string                               │
│                                                              │
│  2. CREATE PROGRAM                                           │
│     clCreateProgramWithSource(context, source)               │
│     → OpenCL program object                                  │
│                                                              │
│  3. COMPILE                                                  │
│     clBuildProgram(program, compilerFlags)                    │
│     → compiled binary for target device                      │
│     (This is a FULL, INDEPENDENT compilation from scratch)   │
│                                                              │
│  4. CREATE KERNEL                                            │
│     clCreateKernel(program, "matrixMultiplication")           │
│     → kernel object ready for execution                      │
│                                                              │
│  5. CACHE                                                    │
│     Store in ConcurrentHashMap<String, OCLInstalledCode>     │
│     Key: "t0-matrixMultiplication"                            │
│     → All subsequent executions use cached binary            │
│                                                              │
│  For PTX: cuModuleLoadData(source) instead of clBuildProgram │
└─────────────────────────────────────────────────────────────┘
```

### Why This Enables Fair Comparison

When comparing a generated kernel against an optimised kernel, each gets its own **completely independent compilation path**:

```
Run 1 (JVM Process A):
  prebuiltTask("t0", ..., "matrix2d_generated.cl", ...)
  → clCreateProgramWithSource(generated_source)
  → clBuildProgram(generated_program)
  → Independent compilation, independent binary

Run 2 (JVM Process B):
  prebuiltTask("t0", ..., "matrix2d_custom.cl", ...)
  → clCreateProgramWithSource(custom_source)
  → clBuildProgram(custom_program)
  → Independent compilation, independent binary
```

There is **no shared state** between the two runs: no shared compilation cache, no shared GPU context, no shared driver optimisations. Each kernel is compiled and executed in isolation.

### Parameter Mapping via AccessorParameters

`AccessorParameters` defines how Java objects map to kernel parameters:

```java
AccessorParameters accessors = new AccessorParameters(4);
accessors.set(0, matrixA, Access.READ_ONLY);     // __global uchar *A in kernel
accessors.set(1, matrixB, Access.READ_ONLY);     // __global uchar *B in kernel
accessors.set(2, matrixC, Access.WRITE_ONLY);    // __global uchar *C in kernel
accessors.set(3, Integer.valueOf(size), Access.NONE);  // __private int size
```

The `Access` mode determines data transfer behaviour:
- `READ_ONLY` → data transferred to GPU before execution
- `WRITE_ONLY` → data transferred back from GPU after execution
- `READ_WRITE` → transferred both ways (e.g., NBody position/velocity arrays)
- `NONE` → scalar value, embedded directly in kernel arguments

### Grid Configuration via WorkerGrid

The grid configuration determines how many GPU threads are launched:

```java
// 2D grid for matrix operations
WorkerGrid2D worker = new WorkerGrid2D(size, size);     // Global: size × size threads
worker.setLocalWork(16, 16, 1);                          // Local: 16 × 16 = 256 per group
GridScheduler scheduler = new GridScheduler("s0.t0", worker);

// 1D grid for element-wise operations
WorkerGrid1D worker = new WorkerGrid1D(numElements);    // Global: numElements threads
worker.setLocalWork(256, 1, 1);                          // Local: 256 per group
```

At execution time, this maps directly to `clEnqueueNDRangeKernel(global_work_size, local_work_size)` for OpenCL, or `cuLaunchKernel()` for CUDA/PTX.

---

## Measuring Kernel Time: Why and How

### Why Kernel-Time-Only Measurement

Total execution time includes multiple components:

```
Total time = JVM overhead + Data Transfer (H→D) + Kernel Execution + Data Transfer (D→H)

               NOT relevant        NOT relevant      THE ONLY THING      NOT relevant
               to kernel           to kernel          THAT CHANGES        to kernel
               optimisation        optimisation       WITH OPTIMISATION   optimisation
```

When comparing two kernels that process the same data on the same device, data transfer time is identical for both. JVM overhead is also identical. The **only variable** is the kernel execution time itself. Measuring anything else adds noise that obscures the real difference.

### How getDeviceKernelTime() Works

TornadoVM's profiler uses **hardware-level event timing** provided by the GPU driver:

```java
// In the benchmark loop:
TornadoExecutionResult result = plan
    .withProfiler(ProfilerMode.SILENT)    // Enable profiling, suppress console output
    .execute();

TornadoProfilerResult profilerResult = result.getProfilerResult();
long kernelTime = profilerResult.getDeviceKernelTime();  // Nanoseconds
```

**For OpenCL:** The kernel time comes from OpenCL event profiling:
```c
cl_ulong start, end;
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, &start);
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, &end);
kernel_time_ns = end - start;
```

**For PTX/CUDA:** The kernel time comes from CUDA event timing:
```c
cuEventRecord(start); cuEventRecord(stop);
cuEventElapsedTime(&milliseconds, start, stop);
```

These are **hardware timer measurements** — they capture the exact GPU execution time with nanosecond precision, independent of any host-side overhead.

### `ProfilerMode.SILENT` vs Other Modes

`ProfilerMode.SILENT` enables the profiling instrumentation but suppresses console output. This gives access to `getDeviceKernelTime()` without flooding the benchmark output with per-iteration profiler dumps. The profiling overhead itself is minimal — it only inserts GPU event markers around kernel dispatch, which costs microseconds on modern GPUs.

---

## The Separate-Runs Methodology and Its Trade-offs

### The Core Approach

Each kernel is benchmarked in its own, fresh JVM execution:

```bash
# Run 1: Generated kernel (fresh JVM, cold GPU)
java ... MatrixMul2DLocalMemorySingleKernelBenchmark kernels/matrix2d_generated.cl 1024
# → Avg: 5.234 ms, 412.5 GFLOP/s

# Run 2: Custom kernel (fresh JVM, cold GPU)
java ... MatrixMul2DLocalMemorySingleKernelBenchmark kernels/matrix2d_custom.cl 1024
# → Avg: 4.123 ms, 523.8 GFLOP/s

# Comparison derived externally: 5.234 / 4.123 = 1.27x speedup
```

### Why Not Both Kernels in the Same JVM?

Running two kernels sequentially in the same process introduces **ordering bias**. The second kernel benefits from:

1. **GPU clock boost** — Modern GPUs dynamically increase clock frequency under sustained load. After 50+ warmup iterations of kernel A, the GPU is in a boosted clock state when kernel B starts.
2. **Driver-level caching** — OpenCL/CUDA drivers cache memory allocation metadata, page table entries, and scheduling decisions. The second kernel inherits these cached states.
3. **Memory controller warm-up** — GPU DRAM controllers and L2 caches are in active state after the first kernel's workload.
4. **JIT warm-up** — TornadoVM's JIT compiler and class loader are fully warmed after the first kernel.

Empirically, this ordering bias produces **20-30% faster times for whichever kernel runs second**, regardless of its actual quality. This is unacceptable for fair comparison.

### The Early Approach: CustomBenchmark Classes

The project's first iteration (`MatrixMultiplication2DCustomBenchmark`, `PiComputationCustomBenchmark`, etc.) ran both kernels in the same JVM and used wall-clock timing:

```java
// Early approach (later abandoned for fair comparison):
long startTime = System.nanoTime();
plan.execute();
long endTime = System.nanoTime();
long totalTime = endTime - startTime;  // Includes data transfer + JVM overhead
```

This approach had two problems:
1. **Ordering bias** (discussed above)
2. **Wall-clock timing** includes JVM overhead, data transfer, and synchronisation — not just kernel execution

These classes were retained in the codebase for quick validation and regression testing, but all reported benchmark results use the SingleKernelBenchmark methodology.

### Trade-offs of the Separate-Runs Approach

| Advantage | Disadvantage |
|-----------|-------------|
| No ordering bias — both kernels start from identical cold state | Cannot directly observe the comparison in one output — must run twice and compare externally |
| Measures pure kernel time via hardware event timing | Slightly higher total experiment time (two JVM startups) |
| Deterministic — same random seed produces same data | JVM startup variance could affect warmup phase (mitigated by 50 warmup iterations) |
| Reproducible — same command produces same measurement | Requires trusting TornadoVM's profiler implementation |

### Statistical Methodology

Each benchmark run collects 50-100 kernel time measurements after a warmup phase:

```java
// Warmup: stabilise GPU clocks, JIT, caches
for (int i = 0; i < 50; i++) {
    plan.execute();  // No profiling, just warm the system
}

// Measurement: collect individual kernel times
ArrayList<Long> kernelTimes = new ArrayList<>();
for (int i = 0; i < 100; i++) {
    TornadoExecutionResult result = plan.withProfiler(ProfilerMode.SILENT).execute();
    long kernelTime = result.getProfilerResult().getDeviceKernelTime();
    kernelTimes.add(kernelTime);
}

// Statistics
LongSummaryStatistics stats = kernelTimes.stream()
    .mapToLong(Long::longValue).summaryStatistics();
// Reports: average, min, max in milliseconds
// Reports: domain-specific metric (GFLOP/s, MPixels/s, etc.)
```

The warmup phase (50 iterations with no profiling) ensures:
- The GPU clock has reached boost frequency
- The driver has cached compilation artifacts
- TornadoVM's internal data structures are initialised
- Memory pages are faulted in and resident

After warmup, the 100 measurement iterations operate on a stable, steady-state system. The average across 100 iterations provides a reliable estimate of sustained kernel performance.

### Data Transfer Modes

All benchmarks use asymmetric data transfer:

```java
.transferToDevice(DataTransferMode.FIRST_EXECUTION, inputA, inputB)
.prebuiltTask("t0", entryPoint, kernelPath, accessors)
.transferToHost(DataTransferMode.EVERY_EXECUTION, output)
```

- **`FIRST_EXECUTION`** for inputs: Data is sent to the GPU **once** during the first warmup iteration and stays resident in device memory for all subsequent iterations. This eliminates redundant host-to-device copies.
- **`EVERY_EXECUTION`** for outputs: Results are copied back to host memory after every execution. This ensures the output array is always fresh for correctness checking.

This asymmetry mirrors real-world GPU workloads where input data is loaded once and computation runs repeatedly.

---

## Validation: Only Correct Kernels Are Accepted

### The Principle

**No performance number is meaningful if the kernel produces wrong results.** Every optimised kernel must pass validation before its benchmark results are reported or used.

### How Validation Works

Each validator class implements a **sequential CPU reference** of the same algorithm and compares it element-by-element against the GPU kernel's output:

```
┌──────────────────────┐         ┌──────────────────────┐
│  CPU Sequential       │         │  GPU Kernel           │
│  Reference            │         │  (under test)         │
│                      │         │                      │
│  Triple nested loop   │         │  prebuiltTask()       │
│  O(n^3) for matmul   │         │  Same input data      │
│  Same input data      │         │  Same random seed     │
│  Known-correct result │         │                      │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                │
           v                                v
     referenceC[]                      kernelC[]
           │                                │
           └────────────┬───────────────────┘
                        │
                        v
              ┌─────────────────────────┐
              │  Element-by-Element      │
              │  Comparison:             │
              │                         │
              │  for (int i = 0; i < N) │
              │    diff = |ref[i]-gpu[i]│
              │    if (diff > TOLERANCE) │
              │      MISMATCH            │
              │                         │
              │  Result: PASS or FAIL    │
              │  Max error: X at index Y │
              └─────────────────────────┘
```

### Algorithm-Specific Tolerances

Different algorithms accumulate floating-point error differently. The tolerance is calibrated per algorithm:

| Algorithm | Tolerance | Justification |
|-----------|----------|---------------|
| Matrix Multiplication | `0.01f` | Accumulated multiply-add errors across N inner-loop iterations |
| NBody | `0.1f` | Physics simulation accumulates FP error across pairwise interactions; position updates compound |
| BlackScholes | `1e-4f` | Numerical stability of `exp()` and `log()` operations in pricing formula |
| FlashAttention | `1e-3f` | Softmax normalisation amplifies small differences |
| Mandelbrot | `0` (exact) | Integer iteration counts must match exactly |
| Reduction | `1e-3f` | Parallel reduction reorders additions, changing FP rounding |

### Validation Size vs Benchmark Size

Validators use **smaller problem sizes** than benchmarks to keep the sequential CPU reference tractable:

| Algorithm | Benchmark Size | Validation Size | Reason |
|-----------|---------------|----------------|--------|
| MatrixMul 2D | 1024×1024 | 512×512 | CPU O(n^3) takes minutes at 1024 |
| NBody | 16,384 bodies | 1,024 bodies | CPU O(n^2) takes minutes at 16K |
| BlackScholes | 4,194,304 | 100,000 | CPU is fast per element, still practical |
| FlashAttention | 32 heads, 2048 ctx | Same | CPU reference is feasible at full size |

### Multiple Kernel Validation

Validators can test multiple kernels in a single run — each gets a fresh `TaskGraph`, `ExecutionPlan`, and output array:

```bash
java ... MatrixMul2DLocalMemoryValidator kernels/matrix2d_generated.cl kernels/matrix2d_custom.cl
# Output:
# Kernel 1 (matrix2d_generated.cl): PASS (max error: 0.0023 at [241, 508])
# Kernel 2 (matrix2d_custom.cl): PASS (max error: 0.0019 at [127, 89])
```

This is acceptable in the validator because we are not measuring performance — only correctness. There is no ordering bias concern because each kernel executes once with fresh state.

### NBody Validation: A Detailed Example

NBody validation is particularly interesting because it validates **both position and velocity** across three spatial dimensions:

```java
// For each body, check x, y, z of both position and velocity
for (int i = 0; i < numBodies; i++) {
    for (int k = 0; k < 3; k++) {  // x, y, z (skip mass at index 3)
        int idx = 4 * i + k;
        
        float posDiff = Math.abs(refPos.get(idx) - gpuPos.get(idx));
        float velDiff = Math.abs(refVel.get(idx) - gpuVel.get(idx));
        
        if (posDiff > 0.1f) mismatches++;
        if (velDiff > 0.1f) mismatches++;
    }
}
// Total comparisons: 2 × numBodies × 3 = 6,144 values for 1024 bodies
```

The higher tolerance (0.1f vs 0.01f for matmul) accounts for the cascading nature of physics simulation: small floating-point differences in force calculation propagate into position updates, which feed back into the next force calculation.

---

## Cross-Platform Kernel Development

Kernels were developed and tested on two platforms:

### MacBook (Apple M4)

- **Backend:** OpenCL only
- **Max work-group:** 1024 (but 32-256 optimal due to register pressure)
- **SIMD width:** 32
- **Memory:** Unified (no CPU/GPU transfer cost)
- **Kernel directory:** `kernels/macbook/`
- **Work-group sizes:** Tuned smaller (e.g., 4×8 = 32 for MatrixMul1D)

### Server "storm" (NVIDIA RTX 4090)

- **Backends:** OpenCL and PTX
- **Max work-group:** 1024
- **Warp size:** 32
- **Shared memory:** 48KB+ per SM
- **Compute units:** 128 SMs
- **Kernel directory:** `kernels/` (root) and `kernels/ptx/`
- **Work-group sizes:** Larger (e.g., 16×16 = 256 for MatrixMul2D)

### Development Workflow

```
MacBook (local development)              Server "storm" (benchmarking)
┌──────────────────────┐                 ┌──────────────────────┐
│ Edit kernel code      │                │ Pull from git         │
│ Test on Apple M4      │  ─ git push ─> │ Build with make       │
│ Quick validation      │                │ Validate on RTX 4090  │
│ Iterate with Claude   │  <─ git pull ─ │ Full benchmark suite  │
└──────────────────────┘                 └──────────────────────┘
```

---

## Populating the Optimisation Library

### The Complete Workflow Per Kernel

For each algorithm and platform, the following steps were performed manually:

```
Step 1: GENERATE
  tornado --printKernel -m tornado.examples/.../<Algorithm> 2>&1 | tee output.txt
  → Extract kernel, save as kernels/<algo>_generated.cl

Step 2: OPTIMISE
  Paste generated kernel into Claude Opus 4.6 chat
  Provide context: algorithm, device, TornadoVM constraints
  Iterate until kernel compiles and passes validation
  → Save as kernels/<algo>_custom.cl

Step 3: VALIDATE
  java ... <Algorithm>Validator kernels/<algo>_generated.cl kernels/<algo>_custom.cl
  → Must show PASS for both kernels

Step 4: BENCHMARK (separate JVM runs)
  java ... <Algorithm>SingleKernelBenchmark kernels/<algo>_generated.cl [size]
  → Record: Avg X.XXX ms
  
  java ... <Algorithm>SingleKernelBenchmark kernels/<algo>_custom.cl [size]
  → Record: Avg Y.YYY ms
  
  Speedup = X / Y

Step 5: UPLOAD TO VECTOR DATABASE
  Open Library Creator (http://localhost:8000)
  Paste original + optimised kernels
  Fill in: description, device, rationale, speedup
  Click "Create Embedding & Upload"
  
Step 6: VERIFY
  Paste the original kernel into the search box
  Confirm the correct optimised version is returned
  Check similarity score > 0.85
```

This workflow was repeated for each algorithm, on each platform, for both OpenCL and PTX backends. The NBody study additionally repeated this for 7 progressive optimisation stages.

---

## The Library Creator Application

A dedicated FastAPI web application (`TVM-MCP-Library/`) was built to manage the optimisation knowledge base. Rather than interacting with Pinecone and Supabase through raw API calls, the Library Creator provides:

**Upload:** Paste original + optimised kernel code, fill in metadata (description, device family, rationale, speedup), click submit. The app automatically generates a code embedding via Voyage AI and stores the entry in the correct Pinecone namespace.

**Search:** Paste kernel code, select namespace (openCL/ptx) and optional device filter, see ranked results with similarity scores. This is the same semantic search the MCP server uses, exposed through a human-friendly UI.

**Two upload paths:**
- `/upload` for OpenCL kernels — full code stored directly in Pinecone metadata
- `/upload-large` for PTX kernels — full code stored in Supabase, reference ID in Pinecone

**Utility scripts:**
- `check_pinecone.py` — detect and remove truncated or corrupt entries
- `view_entry.py` — inspect individual vector database entries

---

## From Manual Process to MCP Server

The MCP server is a **direct formalisation of the manual workflow**. Every component maps to a manual step:

| Manual Step | MCP Server Component |
|-------------|---------------------|
| Study the kernel, identify bottlenecks | Step 1: Analyse (LLM with profiling context) |
| Look at similar optimisations | Step 2: Review Reference (RAG from Pinecone) |
| Plan which techniques to apply | Step 3: Plan (LLM with device constraints) |
| Prompt Claude to generate optimised kernel | Step 4: Generate (LLM with Extended Thinking) |
| If failed: analyse error, re-prompt Claude | Feedback loop (Steps 5-6, 7-8...) |
| Validate against CPU reference | TornadoVM validator classes |
| Benchmark in separate JVM | TornadoVM benchmark classes |
| Upload successful pair | Library Creator / vector DB |

The key insight is that **the MCP server did not replace understanding — it encoded it**. The system prompt constraints, the feedback templates, the error taxonomy — all of these came from observations made during the manual phase. The MCP server is only as good as the domain knowledge embedded in its prompts, and that knowledge was earned through hundreds of manual optimisation attempts.

### Same Benchmarking Methodology in the MCP Pipeline

When the MCP server is used to optimise a kernel, the **same benchmarking and validation infrastructure** is used to evaluate the result. The MCP server does not introduce its own measurement system — it relies on the SingleKernelBenchmark and Validator classes described above.

The evaluation flow for an MCP-optimised kernel is:

```
1. MCP server returns optimised kernel + grid config
       │
       v
2. Save optimised kernel to file (kernels/<algo>_mcp_optimized.cl)
       │
       v
3. VALIDATE: Run the validator with both generated and MCP-optimised kernels
   java ... <Algorithm>Validator kernels/<algo>_generated.cl kernels/<algo>_mcp_optimized.cl
   → Must PASS (same tolerance, same CPU reference as manual optimisations)
   → If FAIL: collect error details, send to /explain-error, retry with feedback
       │
       v
4. BENCHMARK (separate JVM runs, kernel-time-only):
   Run 1: java ... <Algorithm>SingleKernelBenchmark kernels/<algo>_generated.cl [size]
   → Record kernel time (ProfilerMode.SILENT, getDeviceKernelTime())
   
   Run 2: java ... <Algorithm>SingleKernelBenchmark kernels/<algo>_mcp_optimized.cl [size]
   → Record kernel time
   
   Speedup = generated_time / optimized_time
```

This means every speedup number reported — whether from a manually-optimised kernel or an MCP-generated kernel — was measured using the **same methodology**: separate JVM runs, hardware-level kernel timing, deterministic data, and prior validation against a sequential CPU reference. There is no measurement bias between manual and automated optimisations.

### The Chicken-and-Egg Resolution

The project faced a circular dependency:

- To build an AI that optimises kernels, you need examples of optimised kernels.
- To have examples, you need to optimise kernels manually.
- To know which optimisations work, you need benchmarking infrastructure.
- To build benchmarking infrastructure, you need to understand TornadoVM internals.

The resolution was **staged bootstrapping**:

1. **Understand TornadoVM** → `prebuiltTask`, `ProfilerMode.SILENT`, `getDeviceKernelTime()`
2. **Build benchmarks and validators** → SingleKernelBenchmark pattern, separate-runs methodology
3. **Manually optimise kernels** → Using Claude Opus 4.6 via chat, iterating until correct and faster
4. **Build the Library Creator** → Upload validated optimisation pairs to Pinecone/Supabase
5. **Build the MCP Server** → Formalise the manual workflow into an automated pipeline
6. **MCP generates new optimisations** → Validated results can be added back to the library

The manual work was not wasted effort — it **was** the training data, the domain knowledge, and the test suite for the automated system.
