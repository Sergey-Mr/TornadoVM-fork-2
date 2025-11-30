#!/usr/bin/env bash
set -euo pipefail

# Pi Computation Custom Benchmark Runner
# Compares TornadoVM-generated Pi reduction kernel vs custom optimized kernel

SIZE="${1:-8192}"
GENERATED_KERNEL="${2:-kernels/picomputation_generated.cl}"
CUSTOM_KERNEL="${3:-kernels/picomputation_custom.cl}"

echo "======================================"
echo "Pi Computation Custom Kernel Benchmark"
echo "======================================"
echo "Array size: ${SIZE}"
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
  uk.ac.manchester.tornado.examples.compute.custom.PiComputationCustomBenchmark \
  "${SIZE}" \
  "${GENERATED_KERNEL}" \
  "${CUSTOM_KERNEL}"
