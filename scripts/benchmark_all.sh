#!/bin/bash
# ============================================================================
# Comprehensive MCP Benchmark Suite
# ============================================================================
# Runs all kernel types and generates a summary report.
#
# Usage:
#   ./scripts/benchmark_all.sh              # Run all benchmarks
#   ./scripts/benchmark_all.sh --quick      # Quick subset (matrix2d, nbody)
#
# Output:
#   - Individual logs in mcp_test_results/
#   - Summary report: mcp_test_results/summary_TIMESTAMP.txt
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/mcp_test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$RESULTS_DIR/summary_${TIMESTAMP}.txt"

cd "$PROJECT_DIR"

# Source TornadoVM environment
if [ -z "$TORNADO_SDK" ]; then
    source setvars.sh
fi

mkdir -p "$RESULTS_DIR"

# ============================================================================
# Configuration
# ============================================================================

# Define benchmarks: name|class|args
if [ "$1" == "--quick" ]; then
    BENCHMARKS=(
        "matrix2d|uk.ac.manchester.tornado.examples.compute.MatrixMultiplication2D|"
        "nbody|uk.ac.manchester.tornado.examples.compute.NBody|4096 1"
    )
else
    BENCHMARKS=(
        "matrix2d|uk.ac.manchester.tornado.examples.compute.MatrixMultiplication2D|"
        "matrix2d_local|uk.ac.manchester.tornado.examples.kernelcontext.matrices.MatrixMul2DLocalMemory|"
        "nbody|uk.ac.manchester.tornado.examples.compute.NBody|4096 1"
        "mandelbrot|uk.ac.manchester.tornado.examples.compute.Mandelbrot|"
        "montecarlo|uk.ac.manchester.tornado.examples.compute.MonteCarlo|"
        "blackscholes|uk.ac.manchester.tornado.examples.compute.BlackScholes|"
    )
fi

# ============================================================================
# Helper Functions
# ============================================================================

check_server() {
    if ! curl -s http://localhost:8090/health > /dev/null 2>&1; then
        echo "ERROR: MCP server not running at http://localhost:8090"
        echo "Start it with: ./scripts/start_mcp_server.sh"
        exit 1
    fi
}

run_benchmark() {
    local name=$1
    local class=$2
    local args=$3
    local log_file="$RESULTS_DIR/${name}_${TIMESTAMP}.log"

    echo "----------------------------------------"
    echo "Running: $name"
    echo "Class: $class"
    echo "Args: $args"
    echo "Log: $log_file"
    echo ""

    local start_time=$(date +%s)

    # Run and capture output
    java --enable-preview @${TORNADO_SDK}/tornado-argfile \
        -Dtornado.mcp.optimization=true \
        -Dtornado.mcp.server.url=http://localhost:8090/optimize \
        -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
        $class $args 2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Extract results
    local original_time=$(grep -oP "Original.*?Avg:\s*\K[\d.]+" "$log_file" 2>/dev/null || echo "N/A")
    local optimized_time=$(grep -oP "MCP-Optimized.*?Avg:\s*\K[\d.]+" "$log_file" 2>/dev/null || echo "N/A")
    local speedup=$(grep -oP "Speedup:\s*\K[\d.]+x" "$log_file" 2>/dev/null || echo "N/A")
    local status="UNKNOWN"

    if grep -q "FASTER" "$log_file"; then
        status="SUCCESS"
    elif grep -q "SLOWER" "$log_file"; then
        status="SLOWER"
    elif grep -q "TIMEOUT" "$log_file"; then
        status="TIMEOUT"
    elif [ $exit_code -ne 0 ]; then
        status="FAILED"
    fi

    # Append to summary
    printf "%-20s %-10s %-12s %-12s %-10s %ds\n" \
        "$name" "$status" "$original_time" "$optimized_time" "$speedup" "$duration" >> "$SUMMARY_FILE"

    echo ""
    echo "Result: $status (${duration}s)"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

echo "============================================================"
echo "  MCP Benchmark Suite"
echo "  Timestamp: $TIMESTAMP"
echo "  Device: $(uname -m) / $(uname -s)"
echo "============================================================"
echo ""

check_server

# Initialize summary file
cat > "$SUMMARY_FILE" << EOF
============================================================
  MCP Kernel Optimization Benchmark Results
  Timestamp: $TIMESTAMP
  Device: $(uname -m) / $(uname -s)
============================================================

$(printf "%-20s %-10s %-12s %-12s %-10s %s\n" "Benchmark" "Status" "Original" "Optimized" "Speedup" "Duration")
$(printf "%-20s %-10s %-12s %-12s %-10s %s\n" "---------" "------" "--------" "---------" "-------" "--------")
EOF

# Run all benchmarks
total_start=$(date +%s)
passed=0
failed=0

for benchmark in "${BENCHMARKS[@]}"; do
    IFS='|' read -r name class args <<< "$benchmark"
    run_benchmark "$name" "$class" "$args"

    # Count results
    if grep -q "SUCCESS" <<< "$(tail -1 "$SUMMARY_FILE")"; then
        ((passed++))
    else
        ((failed++))
    fi
done

total_end=$(date +%s)
total_duration=$((total_end - total_start))

# Finalize summary
cat >> "$SUMMARY_FILE" << EOF

============================================================
  Summary
============================================================
  Total benchmarks: ${#BENCHMARKS[@]}
  Passed (speedup): $passed
  Failed/Slower: $failed
  Total time: ${total_duration}s
============================================================
EOF

echo ""
echo "============================================================"
echo "  Benchmark Complete!"
echo "============================================================"
echo "  Results: $SUMMARY_FILE"
echo "  Passed: $passed / ${#BENCHMARKS[@]}"
echo "  Total time: ${total_duration}s"
echo "============================================================"
echo ""
cat "$SUMMARY_FILE"
