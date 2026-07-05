#!/bin/bash
# ============================================================================
# MCP Kernel Optimization Test Suite
# ============================================================================
# Usage:
#   ./scripts/mcp_test.sh                    # Run all tests
#   ./scripts/mcp_test.sh matrix2d           # Run specific test
#   ./scripts/mcp_test.sh --list             # List available tests
#
# Prerequisites:
#   1. MCP server running: cd ../TornadoVM_MCP/MCP-server && ./run.sh
#   2. TornadoVM built: make
#   3. Environment sourced: source setvars.sh
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MCP_SERVER_URL="http://localhost:8090"
RESULTS_DIR="$PROJECT_DIR/mcp_test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Ensure we're in project directory
cd "$PROJECT_DIR"

# Source TornadoVM environment if not already done
if [ -z "$TORNADO_SDK" ]; then
    if [ -f "setvars.sh" ]; then
        source setvars.sh
    else
        echo -e "${RED}Error: setvars.sh not found. Run from TornadoVM root directory.${NC}"
        exit 1
    fi
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_mcp_server() {
    echo -n "Checking MCP server at $MCP_SERVER_URL... "
    if curl -s "$MCP_SERVER_URL/health" > /dev/null 2>&1; then
        print_success "Running"
        return 0
    else
        print_error "Not running"
        echo ""
        echo "Please start the MCP server first:"
        echo "  cd ../TornadoVM_MCP/MCP-server"
        echo "  ./run.sh"
        echo ""
        return 1
    fi
}

# Base Java command for TornadoVM with MCP
get_java_cmd() {
    echo "java --enable-preview @\${TORNADO_SDK}/tornado-argfile \
        -Dtornado.mcp.optimization=true \
        -Dtornado.mcp.server.url=http://localhost:8090/optimize \
        -cp \"bin/examples:\${TORNADO_SDK}/share/java/tornado/*\""
}

run_test() {
    local test_name=$1
    local class_name=$2
    local args=$3
    local log_file="$RESULTS_DIR/${test_name}_${TIMESTAMP}.log"

    print_header "Running: $test_name"
    echo "Class: $class_name"
    echo "Args: $args"
    echo "Log: $log_file"
    echo ""

    # Run the test and capture output
    local start_time=$(date +%s)

    java --enable-preview @${TORNADO_SDK}/tornado-argfile \
        -Dtornado.mcp.optimization=true \
        -Dtornado.mcp.server.url=http://localhost:8090/optimize \
        -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
        $class_name $args 2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    echo "Duration: ${duration}s"

    if [ $exit_code -eq 0 ]; then
        # Check for speedup in log
        if grep -q "FASTER" "$log_file"; then
            print_success "Test completed with SPEEDUP"
        elif grep -q "SLOWER" "$log_file"; then
            print_warning "Test completed but kernel was SLOWER"
        else
            print_warning "Test completed (check results manually)"
        fi
    else
        print_error "Test failed with exit code $exit_code"
    fi

    echo ""
    return $exit_code
}

# ============================================================================
# Test Definitions
# ============================================================================

test_matrix2d() {
    run_test "matrix2d" \
        "uk.ac.manchester.tornado.examples.compute.MatrixMultiplication2D" \
        ""
}

test_matrix2d_local() {
    run_test "matrix2d_local" \
        "uk.ac.manchester.tornado.examples.kernelcontext.matrices.MatrixMul2DLocalMemory" \
        ""
}

test_matrix1d() {
    run_test "matrix1d" \
        "uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D" \
        ""
}

test_nbody() {
    run_test "nbody" \
        "uk.ac.manchester.tornado.examples.compute.NBody" \
        "4096 1"  # 4096 bodies, 1 iteration for testing
}

test_mandelbrot() {
    run_test "mandelbrot" \
        "uk.ac.manchester.tornado.examples.compute.Mandelbrot" \
        ""
}

test_montecarlo() {
    run_test "montecarlo" \
        "uk.ac.manchester.tornado.examples.compute.MonteCarlo" \
        ""
}

test_blackscholes() {
    run_test "blackscholes" \
        "uk.ac.manchester.tornado.examples.compute.BlackScholes" \
        ""
}

test_blurfilter() {
    run_test "blurfilter" \
        "uk.ac.manchester.tornado.examples.compute.BlurFilter" \
        ""
}

test_reduction() {
    run_test "reduction" \
        "uk.ac.manchester.tornado.examples.reductions.ReductionAddFloats" \
        ""
}

# Test mode - uses known-good kernel (bypasses LLM)
test_matrix2d_test_mode() {
    print_header "Running: matrix2d (TEST MODE - no LLM)"
    echo "This tests benchmarking infrastructure with a known-good kernel"
    echo ""

    local log_file="$RESULTS_DIR/matrix2d_test_mode_${TIMESTAMP}.log"

    java --enable-preview @${TORNADO_SDK}/tornado-argfile \
        -Dtornado.mcp.optimization=true \
        -Dtornado.mcp.test=true \
        -Dtornado.mcp.server.url=http://localhost:8090/optimize-test \
        -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
        uk.ac.manchester.tornado.examples.compute.MatrixMultiplication2D 2>&1 | tee "$log_file"
}

# ============================================================================
# Main
# ============================================================================

list_tests() {
    echo "Available tests:"
    echo "  matrix2d        - 2D Matrix Multiplication"
    echo "  matrix2d_local  - 2D Matrix Multiplication with Local Memory (KernelContext)"
    echo "  matrix1d        - 1D Matrix Multiplication"
    echo "  nbody           - N-Body Simulation"
    echo "  mandelbrot      - Mandelbrot Fractal"
    echo "  montecarlo      - Monte Carlo Pi Estimation"
    echo "  blackscholes    - Black-Scholes Option Pricing"
    echo "  blurfilter      - Image Blur Filter"
    echo "  reduction       - Float Array Reduction"
    echo ""
    echo "Special tests:"
    echo "  test_mode       - Matrix2D with known-good kernel (no LLM call)"
    echo "  all             - Run all tests"
    echo "  quick           - Run quick smoke test (matrix2d only)"
}

main() {
    print_header "MCP Kernel Optimization Test Suite"
    echo "Timestamp: $TIMESTAMP"
    echo "Results directory: $RESULTS_DIR"
    echo ""

    # Check prerequisites
    check_mcp_server || exit 1

    case "${1:-all}" in
        --list|-l)
            list_tests
            ;;
        matrix2d)
            test_matrix2d
            ;;
        matrix2d_local)
            test_matrix2d_local
            ;;
        matrix1d)
            test_matrix1d
            ;;
        nbody)
            test_nbody
            ;;
        mandelbrot)
            test_mandelbrot
            ;;
        montecarlo)
            test_montecarlo
            ;;
        blackscholes)
            test_blackscholes
            ;;
        blurfilter)
            test_blurfilter
            ;;
        reduction)
            test_reduction
            ;;
        test_mode)
            test_matrix2d_test_mode
            ;;
        quick)
            test_matrix2d
            ;;
        all)
            test_matrix2d
            test_nbody
            test_mandelbrot
            ;;
        *)
            echo "Unknown test: $1"
            list_tests
            exit 1
            ;;
    esac

    print_header "Test Complete"
    echo "Results saved to: $RESULTS_DIR"
}

main "$@"
