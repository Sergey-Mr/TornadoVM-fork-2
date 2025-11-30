#!/usr/bin/env bash
set -euo pipefail

# BFS Custom Benchmark Runner
# Compares TornadoVM-generated BFS kernel vs custom optimized kernel

NUM_NODES="${1:-1000}"
GENERATED_KERNEL="${2:-kernels/bfs_generated.cl}"
CUSTOM_KERNEL="${3:-kernels/bfs_custom.cl}"

echo "======================================"
echo "BFS Custom Kernel Benchmark"
echo "======================================"
echo "Number of nodes: ${NUM_NODES}"
echo "Generated kernel: ${GENERATED_KERNEL}"
echo "Custom kernel: ${CUSTOM_KERNEL}"
echo ""

# Check if kernels exist
if [[ ! -f "${GENERATED_KERNEL}" ]]; then
  echo "Error: Generated kernel not found: ${GENERATED_KERNEL}" >&2
  exit 1
fi

if [[ ! -f "${CUSTOM_KERNEL}" ]]; then
  echo "Error: Custom kernel not found: ${CUSTOM_KERNEL}" >&2
  exit 1
fi

# Run the benchmark
java --enable-preview @tornado-argfile \
  -cp "/tmp/tornado-custom-classes:bin/sdk/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.BFSCustomBenchmark \
  "${NUM_NODES}" \
  "${GENERATED_KERNEL}" \
  "${CUSTOM_KERNEL}"
