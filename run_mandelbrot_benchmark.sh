#!/usr/bin/env bash
set -euo pipefail

# Mandelbrot Custom Benchmark Runner
# Compares TornadoVM-generated Mandelbrot kernel vs custom optimized kernel

SIZE="${1:-1024}"
GENERATED_KERNEL="${2:-kernels/mandelbrot_generated.cl}"
CUSTOM_KERNEL="${3:-kernels/mandelbrot_custom.cl}"

echo "======================================"
echo "Mandelbrot Custom Kernel Benchmark"
echo "======================================"
echo "Image size: ${SIZE}x${SIZE}"
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
  uk.ac.manchester.tornado.examples.compute.custom.MandelbrotCustomBenchmark \
  "${SIZE}" \
  "${GENERATED_KERNEL}" \
  "${CUSTOM_KERNEL}"
