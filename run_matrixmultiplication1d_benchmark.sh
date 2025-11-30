#!/usr/bin/env bash
set -euo pipefail

# Matrix Multiplication 1D Custom Benchmark Runner
# Compares TornadoVM-generated kernels vs user-provided OpenCL kernels

SIZE="${1:-512}"
GENERATED_KERNEL="${2:-kernels/matrixmultiplication1d_generated.cl}"
CUSTOM_KERNEL="${3:-kernels/matrixmultiplication1d_custom.cl}"

printf '%s\n' "======================================" \
  "Matrix Multiplication 1D Benchmark" \
  "======================================"
printf 'Matrix size      : %sx%s\n' "${SIZE}" "${SIZE}"
printf 'Generated kernel : %s\n' "${GENERATED_KERNEL}"
printf 'Custom kernel    : %s\n\n' "${CUSTOM_KERNEL}"

if [[ ! -f "${GENERATED_KERNEL}" ]]; then
  echo "Error: Generated kernel not found: ${GENERATED_KERNEL}" >&2
  exit 1
fi

if [[ ! -f "${CUSTOM_KERNEL}" ]]; then
  echo "Error: Custom kernel not found: ${CUSTOM_KERNEL}" >&2
  exit 1
fi

java --enable-preview @tornado-argfile \
  -cp "/tmp/tornado-custom-classes:bin/sdk/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.MatrixMultiplication1DCustomBenchmark \
  "${SIZE}" \
  "${GENERATED_KERNEL}" \
  "${CUSTOM_KERNEL}"
