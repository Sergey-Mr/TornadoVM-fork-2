# MCP Optimisation Results and Analysis

> Comprehensive analysis of kernel optimisation results across platforms, backends, and algorithms.  
> *This document serves as a foundation for the dissertation chapter on experimental results and evaluation.*

## Table of Contents

1. [Results Summary](#results-summary)
2. [NVIDIA RTX 4090 — OpenCL Results](#nvidia-rtx-4090--opencl-results)
3. [Apple M4 Max — OpenCL Results](#apple-m4-max--opencl-results)
4. [NVIDIA RTX 4090 — PTX Results](#nvidia-rtx-4090--ptx-results)
5. [Cross-Platform Comparison](#cross-platform-comparison)
6. [Per-Algorithm Deep Dive](#per-algorithm-deep-dive)
7. [Vector Database Entries and the Knowledge Base](#vector-database-entries-and-the-knowledge-base)
8. [Key Findings from the Optimisation Library](#key-findings-from-the-optimisation-library)
9. [MCP Pipeline Logs: What the LLM Does](#mcp-pipeline-logs-what-the-llm-does)
10. [Failure Modes and Lessons Learned](#failure-modes-and-lessons-learned)
11. [Limitations and Threats to Validity](#limitations-and-threats-to-validity)

---

## Results Summary

All speedup numbers were measured using the SingleKernelBenchmark methodology: separate JVM runs, hardware-level kernel timing via `getDeviceKernelTime()`, 50 warmup + 100 measurement iterations, with prior validation against a sequential CPU reference.

### NVIDIA RTX 4090 — OpenCL

| Algorithm | Speedup | Notes |
|-----------|---------|-------|
| BFS | **42.87x** | Largest speedup; original kernel highly suboptimal |
| MatrixVectorRowMajor | **4.06x** | Local memory + vectorised loads (float4) |
| MatrixMul2d | **1.26x** | Local memory 16×16 tiling |
| MatrixMul1d | **1.26x** | Local memory tiling |
| BlackScholes | **1.01x** | Already compute-efficient; minimal room for improvement |
| MonteCarlo | *[TO BE RE-RUN]* | Previous result (15.20x) used FP64→FP32 precision reduction — invalidated |
| NBody | *[TO BE RE-RUN]* | Previous result (1.92x) used FP64→FP32 rsqrt — invalidated |

### Apple M4 Max — OpenCL

| Algorithm | Speedup | Notes |
|-----------|---------|-------|
| MatrixVectorRowMajor | **2.46x** | Adapted tiling for unified memory |
| MatrixMul2d | **1.29x** | Slightly better than RTX 4090 (unified memory advantage) |
| MatrixMul1d | **1.29x** | Slightly better than RTX 4090 |
| BFS | **1.66x** | Less dramatic than RTX 4090 (different baseline) |
| ReductionAddFloats | **1.17x** | Parallel reduction tree |
| MonteCarlo | N/A | INT64 atomics not supported on Apple M4 |
| NBody | N/A | FP64 not supported on Apple M4 |

### NVIDIA RTX 4090 — PTX

| Algorithm | Speedup | Notes |
|-----------|---------|-------|
| MatrixMul1d | **2.16x** | PTX shared memory tiling outperforms OpenCL |
| MatrixMul2d | **1.61x** | Shared memory + FMA |
| BFS | **1.13x** | Modest gain in PTX |
| BlackScholes | **1.01x** | Same as OpenCL — minimal headroom |
| MatrixVectorRowMajor | **1.0x** | No improvement |
| NBody | **1.0x** | No improvement |
| ReductionAddFloats | **1.0x** | No improvement |
| MonteCarlo | *[TO BE RE-RUN]* | Previous result (12.83x) used FP64→FP32 precision reduction — invalidated |

---

## NVIDIA RTX 4090 — OpenCL Results

### BFS: 42.87x Speedup

The largest speedup in the entire study. The TornadoVM-generated BFS kernel had significant inefficiencies:

- **Original:** Naive adjacency matrix traversal with no local memory, no early exit, unnecessary global memory reads
- **Optimised:** Hoisted invariants, `vload4` vectorised loads, early exit when frontier is empty, local memory for partial results

The 42.87x number reflects the large gap between a compiler-generated BFS and a hand-tuned one. BFS is inherently irregular (data-dependent memory access), which makes automatic optimisation difficult for JIT compilers.

### MonteCarlo: *[TO BE RE-RUN]*

The previous result (15.20x OpenCL, 12.83x PTX) was achieved through **FP64→FP32 precision reduction**, which has since been identified as an invalid optimisation. The MCP server's system prompt now explicitly forbids precision changes (see Document 1, "Precision Preservation Rule"). This benchmark must be re-run with the updated prompt to measure the genuine structural optimisation potential.

### NBody: *[TO BE RE-RUN]*

The previous result (1.92x OpenCL) was dominated by **FP64 `rsqrt` → FP32 `native_rsqrt`** conversion. While additional structural optimisations were applied (register caching, scalar accumulators, local memory tiling), the precision change accounted for the majority of the speedup. This benchmark must be re-run with the updated prompt to isolate the structural contribution.

> **Note on FP64 performance on RTX 4090:** The RTX 4090 has 82.6 TFLOP/s FP32 vs 1.29 TFLOP/s FP64 (64x ratio). Kernels that use FP64 operations are severely compute-bottlenecked on this hardware, and the only way to achieve large speedups is precision reduction — which is a user decision, not an automated optimisation.

### MatrixVectorRowMajor: 4.06x Speedup

The matrix-vector multiplication saw significant improvement through:
- **Typed pointer casting** with hoisted row base
- **`vload4` vectorised loads** (4 floats per load instruction)
- **Parallel reduction tree** in local memory (log₂ reduction)
- **Loop unrolling** (4x)
- **`restrict` keyword** for pointer aliasing

### NBody: *[TO BE RE-RUN]* (see above)

### MatrixMul 2D and 1D: 1.26x Speedup

Both matrix multiplication variants benefited from:
- 16×16 local memory tiling
- Barrier-synchronised cooperative loading
- Removal of grid-stride loops (replaced with 1:1 thread mapping)

The bottleneck analysis classified these as **memory-bound**, where local memory tiling provides +20-25% improvement by reducing global memory bandwidth demands.

### BlackScholes: 1.01x (No Meaningful Improvement)

BlackScholes is already **compute-efficient** in its generated form — each thread computes one option independently with no memory-sharing opportunities. The kernel is dominated by `exp()`, `log()`, and `sqrt()` operations which are hardware-accelerated. There is minimal room for algorithmic improvement.

---

## Apple M4 Max — OpenCL Results

### Platform Differences

The Apple M4 results show a different optimisation landscape:

1. **Unified memory** — No CPU↔GPU transfer cost, meaning the "transfer overhead" concern that motivates kernel-time-only measurement is less relevant
2. **No FP64** — Many algorithms (NBody, MonteCarlo) could not run due to lack of FP64 support
3. **Smaller optimal work-groups** — 32-256 threads vs 256-1024 on NVIDIA
4. **Comparable matrix speedups** — MatrixMul achieved slightly better speedup (1.29x vs 1.26x) on M4, possibly because the unified memory architecture benefits more from local memory tiling (which reduces redundant loads from the unified address space)

### Notable Results

**MatrixVectorRowMajor: 2.46x** — While lower than the RTX 4090's 4.06x, this still represents a substantial improvement. The M4-specific kernel used smaller tiles and adapted the reduction pattern for the SIMD group width of 32.

**BFS: 1.66x** — Much lower than RTX 4090's 42.87x because the M4's generated kernel was already better optimised (different baseline).

**ReductionAddFloats: 1.17x** — A meaningful improvement from a parallel reduction tree with warp-level optimisation, but limited by the inherently sequential nature of the final reduction stages.

---

## NVIDIA RTX 4090 — PTX Results

### PTX vs OpenCL: Why the Differences?

PTX (Parallel Thread Execution) is NVIDIA's low-level assembly-like IR. Optimisations at the PTX level offer finer control but are more fragile:

| Algorithm | OpenCL Speedup | PTX Speedup | Analysis |
|-----------|---------------|-------------|----------|
| MatrixMul1d | 1.26x | **2.16x** | PTX shared memory + fully unrolled FMA chain outperforms OpenCL tiling |
| MatrixMul2d | 1.26x | **1.61x** | PTX shared memory with explicit register allocation |
| BFS | 42.87x | 1.13x | PTX baseline was already better optimised by the CUDA compiler |
| MatrixVectorRowMajor | 4.06x | 1.0x | PTX optimisation failed to improve |
| MonteCarlo | *[TO BE RE-RUN]* | *[TO BE RE-RUN]* | Previous results used precision reduction — invalidated |
| NBody | *[TO BE RE-RUN]* | 1.0x | OpenCL result used precision reduction; PTX failed to improve |

**Key observation:** PTX optimisation works best for structured, tiling-amenable algorithms (matrix multiplication) where explicit shared memory and register management provides measurable benefit. For irregular algorithms (BFS, NBody), the CUDA compiler's baseline PTX is already reasonably well-optimised.

### Why Some PTX Optimisations Showed 1.0x

The 1.0x results for NBody, MatrixVectorRowMajor, and ReductionAddFloats in PTX suggest that:

1. The LLM's PTX modifications did not produce valid improvements (possible compilation issues or grid misconfigurations)
2. NVIDIA's PTX compiler (`ptxas`) already performs aggressive optimisation on the generated PTX
3. PTX-level optimisation requires different techniques than OpenCL (explicit register allocation, instruction scheduling) that the LLM may not handle as effectively

---

## Cross-Platform Comparison

```
                        OpenCL RTX4090    OpenCL M4     PTX RTX4090
                        ──────────────    ──────────    ───────────
BFS                        42.87x           1.66x         1.13x
MatrixVectorRowMajor        4.06x           2.46x         1.0x
MatrixMul1d                 1.26x           1.29x         2.16x
MatrixMul2d                 1.26x           1.29x         1.61x
ReductionAddFloats           —              1.17x         1.0x
BlackScholes                1.01x            —            1.01x
MonteCarlo              [TO BE RE-RUN]       N/A       [TO BE RE-RUN]
NBody                   [TO BE RE-RUN]       N/A          1.0x
```

### Patterns (Precision-Preserving Results Only)

1. **Memory-bound algorithms benefit consistently** from local memory tiling (MatrixMul ~1.3x across platforms)
2. **PTX outperforms OpenCL for structured computations** (MatrixMul1d: 2.16x PTX vs 1.26x OpenCL) but underperforms for irregular ones
3. **Already-efficient algorithms resist optimisation** (BlackScholes 1.01x everywhere)
4. **BFS result reflects a poor baseline** (42.87x), not exceptional optimisation — the generated kernel was highly suboptimal
5. **Cross-platform portability** is limited — optimisations must be tuned per device
6. **FP64-bottlenecked algorithms** (NBody, MonteCarlo) require re-evaluation with the precision preservation constraint now enforced

---

## Per-Algorithm Deep Dive

### Matrix Multiplication: The Canonical Optimisation

Matrix multiplication was studied most thoroughly because it is the canonical GPU optimisation target:

**Progressive optimisation results (from vector database documentation):**

| Optimisation | Technique | Kernel Time | GFLOP/s | Cumulative Effect |
|-------------|-----------|-------------|---------|-------------------|
| opt1_restrict | `restrict` keyword | 0.431 ms | 4987 | Baseline + aliasing hints |
| opt2_unroll | `#pragma unroll 4` | 0.438 ms | 4900 | No improvement (compiler already unrolls) |
| opt3_workgroup | `reqd_work_group_size(16,16,1)` | TBD | TBD | Small register allocation benefit |
| opt4_local_memory | 16×16 tiles + barriers | TBD | TBD | +20-25% (the big win) |
| opt5_local_unroll | local memory + unrolling | TBD | TBD | Combined benefit |

**Bottleneck classification:** MEMORY-BOUND  
**Impact breakdown (from knowledge base):**
- Local memory tiling: **+20-25%** (dominant)
- Coalesced access: +5-8%
- Work-group size: +2-3%
- `restrict`: +1-2%
- Unrolling: +0-2%

**Size-dependent speedup (from knowledge base):**
- 256×256: 1.02-1.05x (fits L2 cache, tiling adds overhead)
- 512×512: 1.08-1.12x (starting to benefit)
- 1024×1024: 1.20-1.30x (sweet spot for tiling)
- 2048×2048: 1.30-1.40x (maximum benefit from tiling)

### NBody: The Precision Guardrail Discovery

The NBody progressive optimisation study (7 stages, documented in the knowledge base) revealed the most important finding of the entire project — not a performance result, but a **correctness constraint**.

During manual experiments, converting NBody's `double rsqrt` to `float rsqrt` produced a 6.6x speedup (0.971ms → 0.147ms). This is because the RTX 4090's FP32 throughput (82.6 TFLOP/s) is 64x higher than its FP64 throughput (1.29 TFLOP/s). The dominance of this single change meant that all other structural optimisations (restrict, unrolling, register caching, local memory) contributed negligibly while the FP64 bottleneck existed.

**The critical implication:** An LLM optimising this kernel will *naturally* want to convert FP64→FP32 because it is by far the most impactful change. Without an explicit constraint, the MCP server **did** apply this conversion in early runs. This led to the **Precision Preservation Rule** (Hard Rule #2 in the system prompt): the system must never change numeric precision automatically.

The NBody and MonteCarlo MCP results are now marked *[TO BE RE-RUN]* with the precision constraint enforced, to measure the genuine structural optimisation potential when precision is preserved.

### FlashAttention: Modest Gains (~1.08x)

FlashAttention (Llama-3 8B dimensions: 32 heads, 128 head size, 2048 context) showed limited improvement:

**Progressive optimisation attempts (from knowledge base):**

| Variant | Kernel Time | GFLOP/s | Notes |
|---------|-------------|---------|-------|
| flash_loop_init.cl | 15.807 ms | 101.89 | Baseline with loop initialisation optimisation |
| flash_vectorized_load.cl | 14.629 ms | 110.10 | Vectorised memory loads (+8%) |
| flash_vectorized_dot.cl | 15.812 ms | — | Vectorised dot product (no improvement — regression?) |
| flash_divToMul.cl | TBD | TBD | Division-to-multiplication conversion |

The limited improvement reflects the nature of attention: it is already dominated by the softmax computation (which is inherently sequential within each head) and the KV cache memory access pattern (which is determined by the sequence length, not the kernel structure).

### BFS: Platform-Dependent Results

BFS results varied dramatically by platform:

| Platform | Speedup | Analysis |
|----------|---------|----------|
| RTX 4090 OpenCL | 42.87x | Original kernel was highly suboptimal |
| Apple M4 OpenCL | 1.66x | M4's generated kernel was already better |
| RTX 4090 PTX | 1.13x | PTX baseline was already well-optimised |

This highlights an important limitation: **speedup is relative to the baseline**. A high speedup may indicate a poor original kernel rather than an exceptionally good optimisation.

The optimised BFS kernel applied:
- **Early exit** when no frontier changes detected
- **Hoisted loop invariants** (node offsets, adjacency pointers)
- **`vload4` vectorised loads** for adjacency data
- **Local memory** for partial frontier data

---

## Vector Database Entries and the Knowledge Base

The vector database (Pinecone) was populated with optimisation pairs documented in the Notion export. Each entry consists of:

1. **Original kernel** — TornadoVM-generated source
2. **Optimised kernel** — Hand-tuned or MCP-generated source
3. **Optimisation guidance** — Detailed explanation of what was changed and why
4. **Benchmark results** — Speedup number validated through the benchmarking infrastructure

### Entries by Algorithm

| Algorithm | Entries | Devices | Notes |
|-----------|---------|---------|-------|
| MatrixVectorRowMajor | 3+ | RTX 4090, M4 | Multiple optimisation variants (typed pointers, vectorised, reduction) |
| MatrixMul 2D | 2+ | RTX 4090, M4 | Local memory tiling series |
| MatrixMul 1D | 2+ | RTX 4090, M4 | Tiling + FMA |
| NBody | 3+ | RTX 4090 | 2048 bodies, 8192 bodies, full progressive series |
| BFS | 2+ | RTX 4090 | With optimisation guidance |
| FlashAttention | 1+ | RTX 4090 | With progressive optimisation results |
| BlackScholes | 1 | RTX 4090 | Minimal optimisation (documented as "no speedup") |
| MonteCarlo | 1+ | RTX 4090 | FP64→FP32 conversion |
| PiComputation | 1 | RTX 4090 | Reduction optimisation |

### Optimisation Guidance Documents

In addition to kernel pairs, the knowledge base contains **structured optimisation guidance documents** that catalogue patterns for each algorithm type:

- **Guidance 1 (MatrixVectorRowMajor):** 6 patterns — vectorised loads, multiple accumulators, loop unrolling, unrolled parallel reduction, restrict, TornadoVM header offset
- **Guidance 2 (General):** Pattern catalogue — pointer casting/base hoisting, vectorised loads, alignment via scalar prologue, loop unrolling, constant invariants
- **Guidance 5 (BFS):** BFS-specific patterns
- **Guidance 6 (Mandelbrot):** 7 patterns — pointer casting, constant invariants, 2x unrolling, early exit, remove temporaries, FMA
- **Guidance 7 (PiComputation):** 7 patterns — pointer casting, replace `pow()` with bitwise, hoist Leibniz term, local reduction, branch reduction

These guidance documents serve a dual purpose: they are stored in the vector database for RAG retrieval, AND they document the domain knowledge that was manually discovered during the proof-of-concept phase.

---

## Key Findings from the Optimisation Library

### Finding 1: Bottleneck Classification Determines Strategy

The most impactful observation is that **the correct optimisation strategy depends entirely on whether the kernel is memory-bound or compute-bound**:

```
MEMORY-BOUND kernels (MatrixMul, MatrixVector, BFS):
  → Local memory tiling is the primary optimisation (+20-35%)
  → Vectorised loads (vload4) provide additional benefit (+5-8%)
  → Loop unrolling has minimal effect (+0-2%)

COMPUTE-BOUND kernels (NBody, MonteCarlo with FP64):
  → FP64→FP32 conversion is the primary optimisation (5-60x)
  → Tiling HURTS performance (adds overhead without reducing the bottleneck)
  → Register caching and scalar accumulators help (+5-15%)

ALREADY-EFFICIENT kernels (BlackScholes):
  → No significant optimisation available
  → Hardware-accelerated math functions dominate execution time
```

This was formalised into a "Kernel Bottleneck Detection Quick Reference" in the knowledge base, which became the basis for Step 1 (Analyse) in the MCP pipeline.

### Finding 2: Precision Reduction Must Be Explicitly Forbidden

Early experiments showed that FP64→FP32 conversion produced the largest speedup numbers (NBody 6.6x, MonteCarlo 15.2x). However, these are **not valid structural optimisations** — they change the numerical output of the kernel. On consumer NVIDIA GPUs (RTX 4090: 64x FP64/FP32 throughput ratio), precision reduction dominates any structural improvement.

**Action taken:** The system prompt was updated to explicitly forbid precision changes as Hard Rule #2 (see Document 1). The NBody and MonteCarlo results are marked *[TO BE RE-RUN]* with the updated constraint. This finding is important for the dissertation: it demonstrates that an LLM-based optimisation system requires explicit guardrails to prevent semantics-violating transformations that inflate benchmark numbers.

### Finding 3: PTX Can Outperform OpenCL for Structured Computations

For algorithms with regular, tiling-amenable access patterns (matrix multiplication), PTX-level optimisation achieved better results than OpenCL:

- MatrixMul1d: 2.16x (PTX) vs 1.26x (OpenCL)
- MatrixMul2d: 1.61x (PTX) vs 1.26x (OpenCL)

This is because PTX allows explicit control over shared memory allocation, register usage, and instruction scheduling that the OpenCL compiler abstracts away.

### Finding 4: Size-Dependent Optimisation Effectiveness

Local memory tiling effectiveness scales with problem size:

| Size | Expected Speedup | Reason |
|------|-----------------|--------|
| 256×256 | 1.02-1.05x | Data fits in L2 cache; tiling adds overhead |
| 512×512 | 1.08-1.12x | Starting to exceed L2 cache |
| 1024×1024 | 1.20-1.30x | Main memory bandwidth becomes bottleneck |
| 2048×2048 | 1.30-1.40x | Maximum tiling benefit |

This finding informed the MCP server's analysis step: the LLM considers problem size when recommending optimisations.

---

## MCP Pipeline Logs: What the LLM Does

The Notion export contains detailed logs from MCP pipeline runs. Each log captures the full multi-step conversation between the system and Claude Opus 4.6.

### Anatomy of a Typical Log

Using MatrixMul2d (OpenCL, RTX 4090) as an example:

```
LOG: MatrixMul2d Optimization
Backend: OpenCL / Device: nvidia_ada
Original kernel time: 4.358 ms

STEP 1 — ANALYSIS:
  LLM identified: 2D matrix multiplication, global memory accesses,
  grid-stride loop, no local memory usage.
  Classification: MEMORY-BOUND
  Key observation: "Every thread reads an entire row of A and column of B
  from global memory — massive bandwidth waste"

STEP 2 — RAG REFERENCE:
  Pinecone query: namespace=openCL, top_k=1
  Result: 95.4% similarity match
  Reference: Matrix multiplication with 1.26x speedup
  Reference used local memory tiling with 16×16 tiles
  LLM decision: "Score ≥ 85% — follow reference EXACTLY"

STEP 3 — PLAN:
  Selected strategy: Local memory tiling (16×16)
  Work-group size: 16×16 = 256 threads
  Grid config: 2D, global=[size, size], local=[16, 16]
  Decision to remove grid-stride loop for 1:1 thread mapping
  Dimension source: _kernel_context[0] (N from TornadoVM context)

STEP 4 — GENERATE (with Extended Thinking):
  Generated MCP_GRID_CONFIG and optimised kernel
  Key changes:
  - Added __local float tileA[16][16], tileB[16][16]
  - Cooperative tile loading with barrier(CLK_LOCAL_MEM_FENCE)
  - Removed grid-stride loop
  - Preserved +4 float offset
  - Preserved function signature
```

### RAG Similarity Scores Observed

| Algorithm | Backend | RAG Score | Effect |
|-----------|---------|-----------|--------|
| MatrixMul2d | OpenCL/nvidia | 95.4% | Followed reference exactly (≥85% threshold) |
| MatrixMul1d | OpenCL/nvidia | 89.8% | Followed reference exactly |
| MatrixMul1d | OpenCL/M4 | 89.6% | Followed reference exactly |
| NBody | OpenCL/nvidia | — | Used progressive optimisation guidance |
| BFS | OpenCL/nvidia | — | Multiple attempts needed |

### Multi-Attempt Logs

Some optimisations required multiple attempts through the feedback loop:

**BFS (OpenCL, RTX 4090):** 3 attempts
- Attempt 1: Generated optimised kernel
- Attempt 2: -0.2% regression (essentially no change) — feedback identified insufficient optimisation
- Attempt 3: Applied local memory tiling for frontier data → success

**BlackScholes (PTX, RTX 4090):** Extended log (27K+ tokens)
- Multiple attempts to optimise an already-efficient kernel
- Final result: 1.01x — confirming that some kernels simply cannot be improved

---

## Failure Modes and Lessons Learned

### Failure Mode 1: Signature Changes

The most common failure in early runs was the LLM changing the kernel function signature — adding parameters, renaming the function, or reordering arguments. This causes TornadoVM error -46 (`CL_INVALID_KERNEL_NAME`).

**Resolution:** The system prompt was updated with explicit "PRESERVE EXACT FUNCTION SIGNATURE" constraints, repeated at multiple points in the prompt chain.

### Failure Mode 2: Missing +4 Array Offset

The LLM would generate correct-looking code but forget the TornadoVM array header, causing all array accesses to be off by 4 elements. This passes compilation but fails validation.

**Resolution:** The system prompt includes explicit examples of the +4 offset pattern, and the feedback loop specifically checks for this when validation fails.

### Failure Mode 3: Grid-Stride + Tiling Incompatibility

The LLM would add local memory tiling while keeping the grid-stride loop from the original kernel. This combination is incorrect because grid-stride loops assume multiple iterations per thread, while tiling assumes 1:1 thread-to-output mapping.

**Resolution:** The system prompt explicitly states "Grid-stride loops are INCOMPATIBLE with local memory tiling" and the feedback template checks for this pattern.

### Failure Mode 4: Incorrect Global Work Size

For element-wise kernels (like BlackScholes), the LLM would output a grid configuration using a dimension parameter (`size`) instead of the total element count (`size*size`), causing massive GPU underutilisation.

**Resolution:** The MCP_GRID_CONFIG prompt was updated with explicit examples for different kernel types.

---

## Limitations and Threats to Validity

### Measurement Limitations

1. **Single GPU per platform** — Results on one RTX 4090 may not generalise to other NVIDIA GPUs (different SM counts, memory bandwidth). Similarly, Apple M4 results may differ on M1/M2/M3.

2. **Speedup is relative to baseline** — High speedup numbers (e.g., BFS 42.87x) may indicate a poor TornadoVM-generated baseline rather than an exceptionally good optimisation.

3. **Problem size sensitivity** — All benchmarks used fixed sizes. Different problem sizes may show different speedup characteristics (as documented for MatrixMul: 1.05x at 256×256 vs 1.30x at 2048×2048).

4. **JVM variability** — Despite warmup iterations, JVM garbage collection and JIT compilation can introduce variance between runs.

### LLM Limitations

1. **Non-deterministic** — Claude Opus 4.6 may produce different optimisations on repeated runs with the same input. Results are representative, not deterministic.

2. **RAG dependency** — The quality of MCP-generated optimisations depends heavily on the vector database content. If no similar kernel exists in the database, the LLM falls back to general knowledge.

3. **PTX capability gap** — The LLM is less effective at PTX-level optimisation than OpenCL, as evidenced by the 1.0x results for several PTX benchmarks.

4. **Cost per run** — At ~$0.50-$1.40 per optimisation attempt, systematic exploration of the optimisation space is expensive. The experiments reported here represent a targeted selection, not an exhaustive search.

### Generalisability

The results apply specifically to:
- TornadoVM-generated kernels (which have specific patterns like grid-stride loops and array headers)
- The algorithms in the benchmark suite (11 algorithms)
- Two specific hardware platforms (RTX 4090 and Apple M4)
- Claude Opus 4.6 as the LLM (other models may produce different results)

Extending to other GPU programming frameworks (CUDA, SYCL, Vulkan Compute), other algorithms, or other LLMs would require additional validation.
