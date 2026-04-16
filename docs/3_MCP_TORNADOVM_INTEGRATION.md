# MCP Server Integration with TornadoVM

> How TornadoVM sends kernels and feedback to the MCP server, and how optimised kernels are executed.

## Table of Contents

1. [Integration Overview](#integration-overview)
2. [TornadoVM Core Modifications](#tornadovm-core-modifications)
3. [Data Flow: Complete Optimisation Cycle](#data-flow-complete-optimisation-cycle)
4. [HTTP Communication Protocol](#http-communication-protocol)
5. [Kernel Replacement API](#kernel-replacement-api)
6. [Feedback Loop Integration](#feedback-loop-integration)
7. [Grid Configuration Flow](#grid-configuration-flow)
8. [Dual-Protocol Support](#dual-protocol-support)
9. [Error Handling Chain](#error-handling-chain)

---

## Integration Overview

The TornadoVM-MCP integration connects two independent systems:

1. **TornadoVM** (Java) - A heterogeneous programming framework that compiles Java to GPU kernels
2. **MCP Server** (Python) - An AI-powered kernel optimization service

The integration was designed as a **loose coupling via HTTP REST API**, allowing both systems to evolve independently. TornadoVM sends kernel source code and profiling data to the MCP server, receives an optimised kernel back, and executes it on the GPU.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          TornadoVM (Java)                             │
│                                                                       │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│  │  TaskGraph       │    │  TornadoExecution │    │  MCPKernel      │  │
│  │                 │    │  Plan             │    │  Optimizer      │  │
│  │  .prebuiltTask()│    │  .execute()       │    │                 │  │
│  │  .task()        │───>│  .withProfiler()  │───>│  .optimize()    │  │
│  │                 │    │  .getGenerated     │    │  HTTP POST      │  │
│  │                 │    │   KernelSource()   │    │  /optimize      │  │
│  └─────────────────┘    │  .replaceKernel   │    └────────┬────────┘  │
│                         │   Source()         │             │           │
│                         └──────────────────┘             │           │
│                                                           │           │
└───────────────────────────────────────────────────────────┼───────────┘
                                                            │
                                              HTTP POST /optimize
                                              ┌─────────────┘
                                              │
                                              │  Request:
                                              │  - kernel_code
                                              │  - backend (opencl/ptx)
                                              │  - device_family
                                              │  - kernel_time_ns
                                              │  - previous_attempts[]
                                              │
                                              v
┌──────────────────────────────────────────────────────────────────────┐
│                       MCP Server (Python)                             │
│                                                                       │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ HTTP Handler │->│ Chat     │->│ Claude   │->│ RAG (Pinecone +  │  │
│  │             │  │ Optimizer │  │ API      │  │  Voyage AI)      │  │
│  └─────────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
│                                                                       │
│  Response:                                                            │
│  - optimized_kernel                                                   │
│  - grid_config                                                        │
│  - chain_of_thought                                                   │
│  - derived_insights                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## TornadoVM Core Modifications

### What Was Modified

The TornadoVM fork includes modifications at multiple levels:

```
TornadoVM Changes
├── tornado-api/
│   └── MCPKernelOptimizer.java          # NEW: HTTP client for MCP server
├── tornado-runtime/
│   ├── TornadoExecutionPlan.java        # MODIFIED: Added kernel source access + MCP hooks
│   ├── TornadoExecutor.java             # MODIFIED: Profiling data extraction
│   └── TaskGraph.java                   # MODIFIED: prebuiltTask enhancements
├── tornado-drivers/
│   ├── opencl/
│   │   ├── OCLCodeCache.java            # MODIFIED: Kernel source extraction
│   │   └── OCLDeviceContext.java        # MODIFIED: Device capability reporting
│   ├── ptx/
│   │   ├── PTXCodeCache.java            # MODIFIED: PTX source extraction
│   │   └── PTXDeviceContext.java        # MODIFIED: Device capability reporting
│   └── spirv/
│       └── SPIRVDeviceContext.java       # MODIFIED: Device capability reporting
└── tornado-examples/
    └── compute/custom/                   # NEW: All benchmark + validator classes
```

### Key API Additions

**1. `getGeneratedKernelSource(taskId)`**

Extracts the actual OpenCL/PTX source code from TornadoVM's code cache after execution:

```java
// After executing a task, get the kernel source that was actually compiled
String kernelSource = executionPlan.getGeneratedKernelSource("t0");
// Returns the full OpenCL/PTX source including pragmas and function body
```

This is critical for the MCP integration: TornadoVM compiles Java to GPU code at runtime, and we need to capture that code to send it for optimization.

**2. `replaceKernelSource(taskId, newSource)`**

Replaces the kernel in TornadoVM's code cache with a new version:

```java
boolean success = executionPlan.replaceKernelSource("t0", optimizedKernel);
// Next execute() will use the optimized kernel
```

This enables the inline optimization path where the MCP server's output is injected directly into the running TornadoVM instance.

**3. `MCPKernelOptimizer`**

A Java HTTP client that communicates with the MCP server:

```java
MCPKernelOptimizer optimizer = new MCPKernelOptimizer("http://localhost:8080");
String optimized = optimizer.optimize(
    kernelSource,           // Original kernel
    "opencl",               // Backend
    "nvidia_ada",           // Device family
    kernelTimeNs,           // Profiling data
    previousAttempts        // Feedback from prior attempts
);
```

---

## Data Flow: Complete Optimisation Cycle

### First-Time Optimisation (No Feedback)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  1. TornadoVM executes Java code, generating OpenCL/PTX kernel           │
│     ┌─────────────────┐                                                  │
│     │ @Parallel        │                                                  │
│     │ for (int i...)   │  ──JIT Compilation──>  __kernel void func(...)   │
│     └─────────────────┘                                                  │
│                                                                          │
│  2. Benchmark infrastructure runs kernel, collects profiling data        │
│     ┌──────────────────────────────────┐                                 │
│     │ Warmup: 50 iterations            │                                 │
│     │ Measure: 100 iterations          │                                 │
│     │ Collect: kernel_time_ns per iter │                                 │
│     │ Result: avg 5.234 ms             │                                 │
│     └──────────────────────────────────┘                                 │
│                                                                          │
│  3. Send to MCP server via HTTP POST /optimize                           │
│     ┌──────────────────────────────────┐                                 │
│     │ {                                │                                 │
│     │   "kernel_code": "...",          │                                 │
│     │   "backend": "opencl",           │                                 │
│     │   "device_family": "nvidia_ada", │                                 │
│     │   "kernel_time_ns": 5234000      │                                 │
│     │ }                                │                                 │
│     └──────────────────────────────────┘                                 │
│                          │                                               │
│                          v                                               │
│  4. MCP Server runs 4-step pipeline (30-60 seconds)                      │
│     Step 1: Analyze → Step 2: RAG → Step 3: Plan → Step 4: Generate     │
│                          │                                               │
│                          v                                               │
│  5. Receive optimized kernel + grid config                               │
│     ┌──────────────────────────────────┐                                 │
│     │ {                                │                                 │
│     │   "optimized_kernel": "...",     │                                 │
│     │   "grid_config": {               │                                 │
│     │     "dimensions": 2,             │                                 │
│     │     "global": ["size", "size"],  │                                 │
│     │     "local": [16, 16]            │                                 │
│     │   }                              │                                 │
│     │ }                                │                                 │
│     └──────────────────────────────────┘                                 │
│                                                                          │
│  6. Save kernel to file, validate, then benchmark                        │
│     ┌──────────────────────────────────┐                                 │
│     │ Validate: Compare GPU vs CPU ref │                                 │
│     │ Benchmark: 50 warmup + 100 iter  │                                 │
│     │ Result: avg 4.123 ms             │                                 │
│     │ Speedup: 1.27x                   │                                 │
│     └──────────────────────────────────┘                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### With Feedback Loop (Retry After Failure)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Attempt 1: MCP returns kernel → TornadoVM runs it → FAILS              │
│  (compilation error, wrong results, or performance regression)           │
│                                                                          │
│  ┌──────────────────────────────────────────────┐                        │
│  │  Collect failure information:                 │                        │
│  │  - compilation_error: "clBuildProgram: -11"  │                        │
│  │  - OR validation_error: "Mismatch at idx 42" │                        │
│  │  - OR optimized_time_ms: 7.891 (slower)      │                        │
│  └──────────────────────────────────────────────┘                        │
│                          │                                               │
│                          v                                               │
│  (Optional) Call /explain-error for AI diagnosis                         │
│  ┌──────────────────────────────────────────────┐                        │
│  │  Response:                                    │                        │
│  │  - explanation: "Error -11 = build failure"   │                        │
│  │  - likely_cause: "Undeclared 'tile' variable" │                        │
│  │  - suggested_fix: "Add __local float tile..." │                        │
│  └──────────────────────────────────────────────┘                        │
│                          │                                               │
│                          v                                               │
│  Attempt 2: Send to /optimize with previous_attempts                     │
│  ┌──────────────────────────────────────────────┐                        │
│  │ {                                             │                        │
│  │   "kernel_code": "<ORIGINAL kernel>",         │                       │
│  │   "previous_attempts": [{                     │                        │
│  │     "optimized_kernel": "<failed kernel>",    │                        │
│  │     "attempt_number": 1,                      │                        │
│  │     "compilation_error": "clBuildProgram: -11",│                       │
│  │     "error_explanation": "...",               │                        │
│  │     "error_likely_cause": "...",              │                        │
│  │     "error_suggested_fix": "..."              │                        │
│  │   }]                                          │                        │
│  │ }                                             │                        │
│  └──────────────────────────────────────────────┘                        │
│                          │                                               │
│                          v                                               │
│  MCP Server runs feedback pipeline:                                      │
│  Step 5: Analyze failure → Step 6: Generate new kernel                   │
│  (LLM has context of what went wrong and tries different approach)       │
│                          │                                               │
│                          v                                               │
│  Attempt 2 result: validate and benchmark                                │
│  If still fails → Attempt 3 (max_feedback_iterations = 3)               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## HTTP Communication Protocol

### Request Flow

```
TornadoVM (Java)                    MCP Server (Python)
      │                                    │
      │  POST /optimize                    │
      │  Content-Type: application/json    │
      │  ─────────────────────────────>    │
      │                                    │
      │    {                               │
      │      "kernel_code": "...",         │
      │      "backend": "opencl",          │
      │      "device_family": "nvidia_ada",│
      │      "kernel_time_ns": 5234000,    │
      │      "copy_in_time_ns": 120000,    │
      │      "copy_out_time_ns": 80000,    │
      │      "global_work_size": [1024,    │
      │                           1024],   │
      │      "local_work_size": [16, 16],  │
      │      "parameter_values": {         │
      │        "size": 1024                │
      │      },                            │
      │      "max_work_group_size": 1024,  │
      │      "compute_units": 128,         │
      │      "device_name": "RTX 4090",    │
      │      "session_id": "exp-1",        │
      │      "previous_attempts": []       │
      │    }                               │
      │                                    │
      │         (30-90 seconds)            │
      │                                    │
      │  200 OK                            │
      │  <─────────────────────────────    │
      │                                    │
      │    {                               │
      │      "optimized_kernel": "...",    │
      │      "grid_config": {...},         │
      │      "chain_of_thought": {...},    │
      │      "similar_examples_used": 1,   │
      │      "model": "claude-opus-4-6",   │
      │      "iterations": 1,             │
      │      "derived_insights": {...}     │
      │    }                               │
      │                                    │
```

### Timeout Handling

The MCP server calls Claude API which can take 30-90 seconds depending on:
- Model load (cold start vs warm)
- Extended thinking budget consumed
- Conversation length (tokens accumulate across steps)

**Java side timeout:** 180 seconds (3 minutes) for the HTTP call
**Python side:** No timeout on LLM calls (relies on Anthropic API timeout)

### Session Management

The `session_id` field groups multiple optimization attempts:

```
Session: "blackscholes-experiment-1"
├── Attempt 1: /optimize (initial) → compilation error
├── /explain-error → diagnosis
├── Attempt 2: /optimize (with previous_attempts[1]) → validation error
├── /explain-error → diagnosis
└── Attempt 3: /optimize (with previous_attempts[1,2]) → success, 1.15x speedup
```

Langfuse groups all traces by session_id for end-to-end visibility.

---

## Kernel Replacement API

### Two Modes of Operation

**Mode 1: File-Based (prebuiltTask)**

Used by the SingleKernelBenchmark infrastructure. The optimized kernel is saved to a file and loaded via `prebuiltTask`:

```java
// Save MCP output to file
writeFile("kernels/matrix1d_mcp_optimized.cl", optimizedKernel);

// Run benchmark with the file
prebuiltTask("t0", "matrixMultiplication", "kernels/matrix1d_mcp_optimized.cl", accessors);
```

**Advantage:** Clean separation, kernel is preserved on disk for inspection.

**Mode 2: In-Memory Replacement**

Used by the MCPHttpIntegrationTest. The kernel is replaced directly in TornadoVM's code cache:

```java
// Run original to collect profiling data
plan.execute();
String kernelSource = plan.getGeneratedKernelSource("t0");

// Get optimization from MCP
String optimized = callMCPServer(kernelSource, ...);

// Replace kernel in memory
plan.replaceKernelSource("t0", optimized);

// Re-execute with optimized kernel
plan.execute();  // Now uses the optimized version
```

**Advantage:** No file I/O, works within a single execution flow.  
**Disadvantage:** Not suitable for fair benchmarking (same JVM, warm GPU).

---

## Grid Configuration Flow

### The Problem

TornadoVM decides the grid configuration (global and local work sizes) when it compiles a kernel. When the MCP server returns an optimized kernel, it may need a **different** grid configuration (e.g., the optimized kernel uses 16x16 tiling, so local work size must be 16x16).

### How Grid Config Flows

```
MCP Server                          TornadoVM
    │                                    │
    │  "grid_config": {                  │
    │    "dimensions": 2,                │
    │    "global_work_size": ["size",    │
    │                         "size"],   │
    │    "local_work_size": [16, 16],    │
    │    "pattern": "tiled"              │
    │  }                                 │
    │                                    │
    │ ──────────────────────────────>    │
    │                                    │
    │                          Parse grid_config
    │                          Resolve "size" → 1024
    │                                    │
    │                          WorkerGrid2D(1024, 1024)
    │                          setLocalWork(16, 16, 1)
    │                                    │
    │                          Execute with optimized grid
```

### Parameter Resolution

The `global_work_size` uses parameter names (strings) that must be resolved to actual values:

| Expression | Context | Resolved Value |
|-----------|---------|---------------|
| `"size"` | MatrixMul with size=1024 | 1024 |
| `"size*size"` | BlackScholes with size=4096 | 16,777,216 |
| `"numBodies"` | NBody with numBodies=16384 | 16,384 |
| `"width"` | BlurFilter with width=2048 | 2048 |

---

## Dual-Protocol Support

### HTTP REST API (Primary for Benchmarking)

Used by the Java benchmark infrastructure:

```
Port 8080 (default)
POST /optimize     → Full optimization
POST /explain-error → Error diagnosis
GET  /health       → Health check
```

### MCP STDIO Protocol (For IDE Integration)

The server also implements the Model Context Protocol for integration with AI coding assistants:

```
stdin/stdout JSON-RPC messages
Tools: optimize_tornadovm_kernel, search_optimization_examples, etc.
```

This allows tools like Claude Code or VS Code extensions to call the optimization server as an MCP tool.

### Why Both?

| Protocol | Use Case | Caller |
|----------|----------|--------|
| HTTP | Automated benchmarking, Java integration | TornadoVM Java code |
| MCP STDIO | Interactive AI-assisted development | IDE extensions, Claude Code |

---

## Error Handling Chain

### Error Categories and Handling

```
┌───────────────────────────────────────────────────────────────┐
│                    ERROR HANDLING CHAIN                         │
│                                                                │
│  ┌─────────────────┐                                           │
│  │ MCP Server Error │  → HTTP 500, log error, retry with       │
│  │ (timeout, crash) │    different timeout                      │
│  └─────────────────┘                                           │
│                                                                │
│  ┌─────────────────┐                                           │
│  │ Compilation Error│  → Save error message                    │
│  │ (clBuildProgram  │    → Call /explain-error                 │
│  │  returns -11)    │    → Retry /optimize with                │
│  └─────────────────┘      previous_attempts + error diagnosis  │
│                                                                │
│  ┌─────────────────┐                                           │
│  │ Validation Error │  → Save mismatch details                 │
│  │ (wrong results)  │    → Retry /optimize with                │
│  └─────────────────┘      previous_attempts + validation error │
│                                                                │
│  ┌─────────────────┐                                           │
│  │ Performance      │  → Record both timings                   │
│  │ Regression       │    → Retry /optimize with                │
│  │ (slower)         │      previous_attempts + timing data     │
│  └─────────────────┘                                           │
│                                                                │
│  ┌─────────────────┐                                           │
│  │ Entry Point      │  → Check kernel function name            │
│  │ Mismatch         │    → Usually a signature change          │
│  │ (error -46)      │    → Retry with emphasis on signature    │
│  └─────────────────┘                                           │
│                                                                │
│  Max retries: 3 (configurable via max_feedback_iterations)     │
└───────────────────────────────────────────────────────────────┘
```

### Common Errors and Their Causes

| Error | Code | Typical Cause |
|-------|------|---------------|
| `clBuildProgram → -11` | CL_BUILD_PROGRAM_FAILURE | Syntax error, undeclared variable, wrong types |
| `clCreateKernel → -46` | CL_INVALID_KERNEL_NAME | Entry point name changed by optimization |
| `cuModuleLoadData → 218` | CUDA_ERROR_INVALID_PTX | PTX has duplicate headers (.version, .target) |
| Validation mismatch | N/A | Missing +4 offset, wrong bounds, barrier issues |
| Performance regression | N/A | Wrong grid size, excessive local memory overhead |
