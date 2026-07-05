#!/bin/bash
# ============================================================================
# Quick MCP Test - Single kernel optimization test
# ============================================================================
# This is the fastest way to verify the MCP integration is working.
# Uses test mode (known-good kernel) to bypass LLM and test benchmarking only.
#
# Usage:
#   ./scripts/quick_test.sh           # Test mode (no LLM, ~10s)
#   ./scripts/quick_test.sh --full    # Full LLM optimization (~60s)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Source TornadoVM environment
if [ -z "$TORNADO_SDK" ]; then
    source setvars.sh
fi

# Check MCP server
echo "Checking MCP server..."
if ! curl -s http://localhost:8090/health > /dev/null 2>&1; then
    echo "MCP server not running. Starting in background..."
    echo "(You may need to run ./scripts/start_mcp_server.sh in another terminal)"
    exit 1
fi
echo "MCP server: OK"
echo ""

if [ "$1" == "--full" ]; then
    echo "=== FULL LLM OPTIMIZATION TEST ==="
    echo "This will call the LLM and may take ~60 seconds"
    echo ""

    java --enable-preview @${TORNADO_SDK}/tornado-argfile \
        -Dtornado.mcp.optimization=true \
        -Dtornado.mcp.server.url=http://localhost:8090/optimize \
        -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
        uk.ac.manchester.tornado.examples.compute.MatrixMultiplication2D
else
    echo "=== TEST MODE (Known-Good Kernel, No LLM) ==="
    echo "This tests the benchmarking infrastructure without LLM calls"
    echo "Expected: ~1.2x speedup with tiled local memory kernel"
    echo ""

    java --enable-preview @${TORNADO_SDK}/tornado-argfile \
        -Dtornado.mcp.optimization=true \
        -Dtornado.mcp.test=true \
        -Dtornado.mcp.server.url=http://localhost:8090/optimize-test \
        -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
        uk.ac.manchester.tornado.examples.compute.MatrixMultiplication2D
fi

echo ""
echo "Test complete!"
