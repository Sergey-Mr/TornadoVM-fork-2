#!/bin/bash
# ============================================================================
# Start MCP Server for TornadoVM Kernel Optimization
# ============================================================================
#
# Prerequisites:
#   - Python 3.10+
#   - API keys in MCP-server/.env:
#       ANTHROPIC_API_KEY=sk-ant-...
#       VOYAGE_API_KEY=pa-...
#       PINECONE_API_KEY=...
#
# Usage:
#   ./scripts/start_mcp_server.sh         # Start on port 8090
#   ./scripts/start_mcp_server.sh 8080    # Start on custom port
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_SERVER_DIR="/Users/serhiitupikin/Documents/Coding/TornadoVM_MCP/MCP-server"

echo "============================================"
echo "  MCP Server for TornadoVM"
echo "============================================"
echo "Directory: $MCP_SERVER_DIR"
echo ""

cd "$MCP_SERVER_DIR"

# Check for .env file
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Create $MCP_SERVER_DIR/.env with:"
    echo "  ANTHROPIC_API_KEY=sk-ant-..."
    echo "  VOYAGE_API_KEY=pa-..."
    echo "  PINECONE_API_KEY=..."
    echo ""
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -e .
else
    source .venv/bin/activate
fi

# Set port (default 8090 for TornadoVM integration)
PORT=${1:-8090}

echo ""
echo "Starting server on port $PORT..."
echo ""
echo "Endpoints:"
echo "  POST http://localhost:$PORT/optimize      - Full optimization (LLM + RAG)"
echo "  POST http://localhost:$PORT/optimize-test - Test mode (known-good kernel)"
echo "  GET  http://localhost:$PORT/health        - Health check"
echo ""
echo "Test with:"
echo "  curl http://localhost:$PORT/health"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

# Start the server
python -m tornadovm_mcp.api $PORT
