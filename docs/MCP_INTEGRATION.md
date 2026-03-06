# TornadoVM MCP Kernel Optimization Integration

This document describes the integration between TornadoVM and the MCP (Model Context Protocol) server for AI-powered kernel optimization.

## Overview

The MCP integration allows TornadoVM to automatically optimize its generated OpenCL/PTX kernels using Claude AI with RAG (Retrieval-Augmented Generation) context from a knowledge base of optimization examples.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TornadoVM (Java)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Java Code → Graal JIT → Generated Kernel (OpenCL/PTX)                    │
│                                    ↓                                        │
│                         ┌─────────────────────┐                            │
│                         │  MCP Optimization?  │                            │
│                         └─────────┬───────────┘                            │
│                                   ↓                                        │
│                    MCPKernelOptimizer (JSON-RPC over stdio)                │
│                                   ↓                                        │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MCP Server (Python)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   optimize_tornadovm_kernel() Tool                                         │
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│   │  Pinecone   │    │  Claude AI  │    │  Langfuse   │                   │
│   │  (RAG/VDB)  │    │   (LLM)     │    │  (Tracing)  │                   │
│   └─────────────┘    └─────────────┘    └─────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Architecture Components

### 1. TornadoVM Side (Java)

| File | Purpose |
|------|---------|
| `TornadoOptions.java` | MCP configuration flags |
| `MCPKernelOptimizer.java` | Java MCP client (JSON-RPC over stdio) |
| `OCLCodeCache.java` | OpenCL integration point |
| `PTXCodeCache.java` | PTX integration point |

### 2. MCP Server Side (Python)

| File | Purpose |
|------|---------|
| `server.py` | FastMCP server with optimization tool |
| `optimizer.py` | 4-step chain-of-thought optimization |
| `rag.py` | Pinecone vector search |
| `prompts.py` | LLM prompt templates |
| `insights.py` | Profiling data analysis |
| `tracing.py` | Langfuse observability |

## Configuration

### Environment Variables

```bash
# Required: Path to MCP server
export TORNADOVM_MCP_PATH=/path/to/MCP-server
```

### JVM System Properties

| Property | Default | Description |
|----------|---------|-------------|
| `tornado.mcp.optimization` | `false` | Enable MCP kernel optimization |
| `tornado.mcp.server.path` | `$TORNADOVM_MCP_PATH` | Path to MCP server directory |

## Usage

### Basic Usage

```bash
# Enable MCP optimization
tornado --jvm="-Dtornado.mcp.optimization=true" \
        --printKernel \
        -m tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D
```

### Testing MCP Integration

```bash
# Run the MCP integration test
export TORNADOVM_MCP_PATH=/path/to/MCP-server
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MCPIntegrationTest
```

---

## Profiling Methodology

### Why Multiple Iterations?

When TornadoVM runs a kernel, you see multiple profiling outputs because:

1. **JIT Compilation** - First run includes Graal compilation time
2. **GPU Warmup** - GPU needs time to reach boost clocks
3. **Memory Allocation** - First run allocates device memory
4. **Driver Optimization** - OpenCL/CUDA drivers optimize on first runs

### Iteration Phases

```
Iteration 1-10:   Warmup (DISCARDED)
                  - JIT compilation
                  - Memory allocation
                  - GPU clock ramping

Iteration 11-50:  Stabilization
                  - Performance stabilizes
                  - Minor variations

Iteration 51+:    Steady State (MEASURED)
                  - Representative performance
                  - Used for benchmarking
```

### Which Metric to Use?

| Metric | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Average** | Performance comparison, GFLOP/s | Stable, representative | Affected by outliers |
| **Median** | Robust comparison | Ignores outliers | Slightly less common |
| **Minimum** | Best-case analysis | Shows achievable peak | May be unrealistic |
| **Maximum** | Worst-case analysis | Shows tail latency | Often noise/interference |

**Our benchmarks use Average** for GFLOP/s calculations because:
- It's the standard in HPC benchmarking
- It accounts for real-world variance
- It's reproducible across runs

### Benchmark Structure

```java
private static final int WARM_UP_ITERATIONS = 50;   // Discarded
private static final int BENCHMARK_ITERATIONS = 100; // Measured

// Phase 1: Warmup (no profiling)
for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
    plan.execute();
}

// Phase 2: Measurement (with profiling)
for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
    TornadoExecutionResult result = plan
            .withProfiler(ProfilerMode.SILENT)
            .execute();
    long kernelTime = result.getProfilerResult().getDeviceKernelTime();
    kernelTimes.add(kernelTime);
}

// Statistics
LongSummaryStatistics stats = kernelTimes.stream()
        .mapToLong(Long::longValue)
        .summaryStatistics();

double avgMs = stats.getAverage() / 1_000_000.0;
double minMs = stats.getMin() / 1_000_000.0;
double maxMs = stats.getMax() / 1_000_000.0;
```

### Profiling Data Fields

| Field | Unit | Description |
|-------|------|-------------|
| `TOTAL_KERNEL_TIME` | ns | GPU kernel execution time |
| `COPY_IN_TIME` | ns | Host → Device transfer time |
| `COPY_OUT_TIME` | ns | Device → Host transfer time |
| `TOTAL_COPY_IN_SIZE_BYTES` | bytes | Data transferred to device |
| `TOTAL_COPY_OUT_SIZE_BYTES` | bytes | Data transferred from device |
| `TOTAL_GRAAL_COMPILE_TIME` | ns | JIT compilation time (first run only) |
| `TOTAL_DRIVER_COMPILE_TIME` | ns | OpenCL/CUDA compilation time (first run only) |

### Example Profiling Analysis

From a Matrix2D run on Apple M4 Max:

```
First run:
  KERNEL_TIME: 39,651 ns (includes compilation overhead)
  GRAAL_COMPILE_TIME: 41,074,584 ns
  DRIVER_COMPILE_TIME: 44,935,583 ns

Warmup (iterations 2-10):
  KERNEL_TIME: ~30,000-35,000 ns

Steady state (iterations 11+):
  KERNEL_TIME: ~7,600-8,000 ns  ← Use this!
  COPY_IN_TIME: ~60 ns
  COPY_OUT_TIME: ~230 ns
```

**Conclusion**: Use steady-state values (~8,000 ns) for MCP optimization, not first-run values.

---

## MCP Optimization Pipeline

### 4-Step Chain-of-Thought

The MCP server uses a 4-step reasoning process:

```
Step 1: UNDERSTAND KERNEL
├── What algorithm does this implement?
├── What are the memory access patterns?
├── What is the computational intensity?
└── Analyze profiling data (if available)

Step 2: ANALYZE RAG EXAMPLES
├── Search Pinecone for similar kernels
├── What optimizations worked before?
├── What speedups were achieved?
└── Which techniques are applicable?

Step 3: CREATE OPTIMIZATION PLAN
├── Based on kernel analysis
├── Based on RAG examples
├── Consider device-specific optimizations
└── Prioritize by expected impact

Step 4: GENERATE OPTIMIZED KERNEL
├── Apply planned optimizations
├── Preserve correctness
├── Maintain TornadoVM signature
└── Add comments explaining changes
```

### Profiling Data for MCP

When calling MCP with profiling data, provide:

```json
{
  "kernel_code": "...",
  "backend": "opencl",
  "device_family": "apple_m4",
  "kernel_time_ns": 8000,
  "copy_in_time_ns": 60,
  "copy_out_time_ns": 230,
  "copy_in_bytes": 3145776,
  "copy_out_bytes": 1048592,
  "global_work_size": [512, 512],
  "local_work_size": [16, 16]
}
```

### Bottleneck Classification

The MCP server automatically classifies kernels:

| Classification | Criteria | Optimization Focus |
|----------------|----------|-------------------|
| **Compute-bound** | `transfer_time / kernel_time < 0.2` | Arithmetic optimizations, FMA, unrolling |
| **Balanced** | `0.2 ≤ ratio ≤ 0.5` | Both memory and compute |
| **Memory-bound** | `ratio > 0.5` | Coalescing, local memory, tiling |

---

## Current Limitation: Compile-Time Optimization

The current integration optimizes at **compile time** (before first execution), so profiling data is not yet available:

```
Current Flow:
  Compile kernel → Optimize (no profiling) → Execute

Ideal Flow:
  Compile kernel → Execute (collect profiling) → Re-optimize → Execute optimized
```

### Future Enhancement: Profile-Then-Optimize

A future enhancement could implement:

1. First execution with original kernel (collect profiling)
2. Send kernel + profiling data to MCP
3. Receive optimized kernel
4. Re-compile and use optimized kernel for subsequent runs

---

## Device Family Detection

The MCP client automatically detects device families:

| Vendor | Device | Family |
|--------|--------|--------|
| NVIDIA | RTX 4090/4080/4070 | `nvidia_ada` |
| NVIDIA | RTX 3090/3080/3070 | `nvidia_ampere` |
| NVIDIA | A100/A10 | `nvidia_ampere_datacenter` |
| NVIDIA | H100 | `nvidia_hopper` |
| Apple | M4 | `apple_m4` |
| Apple | M3 | `apple_m3` |
| Apple | M2 | `apple_m2` |
| Apple | M1 | `apple_m1` |
| AMD | Any | `amd_generic` |
| Intel | Any | `intel_generic` |

---

## Troubleshooting

### MCP Server Not Starting

```bash
# Check TORNADOVM_MCP_PATH
echo $TORNADOVM_MCP_PATH

# Test server manually
cd $TORNADOVM_MCP_PATH
source .venv/bin/activate
python -m tornadovm_mcp.server
```

### No Optimization Happening

1. Check flag is set: `-Dtornado.mcp.optimization=true`
2. Check server path is correct
3. Check server logs (stderr)

### Kernel Compilation Error After Optimization

If the optimized kernel fails to compile:
- The system automatically falls back to the original kernel
- Check MCP server logs for optimization details
- Report the issue with original and optimized kernels

---

## Files Reference

### TornadoVM Files

```
tornado-api/src/main/java/uk/ac/manchester/tornado/api/mcp/
└── MCPKernelOptimizer.java          # MCP client

tornado-runtime/src/main/java/uk/ac/manchester/tornado/runtime/common/
└── TornadoOptions.java              # MCP flags

tornado-drivers/opencl/src/main/java/uk/ac/manchester/tornado/drivers/opencl/
└── OCLCodeCache.java                # OpenCL integration

tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/
└── PTXCodeCache.java                # PTX integration
```

### MCP Server Files

```
MCP-server/src/tornadovm_mcp/
├── server.py                        # FastMCP server
├── optimizer.py                     # Chain-of-thought logic
├── prompts.py                       # LLM prompts
├── rag.py                           # Pinecone search
├── insights.py                      # Profiling analysis
├── tracing.py                       # Langfuse integration
└── config.py                        # Configuration
```
