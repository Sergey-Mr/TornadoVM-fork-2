# How to Run TornadoVM on NVIDIA RTX 4090 Server

## Server Details
- **Host**: `serhii@storm`
- **GPU**: NVIDIA GeForce RTX 4090
- **Location**: `~/TornadoVM-fork-2`

---

## Initial Setup (After Each SSH Login)

Every time you SSH into the server, run these commands in order:

```bash
# 1. Navigate to project
cd ~/TornadoVM-fork-2

# 2. Set Java Home (GraalVM JDK 21)
export JAVA_HOME=$HOME/graalvm-jdk-21.0.9+7.1
export PATH=$JAVA_HOME/bin:$PATH

# 3. Set TornadoVM SDK path
export TORNADO_SDK=$HOME/TornadoVM-fork-2/dist/tornadovm-1.1.2-dev-opencl-linux-amd64/tornadovm-1.1.2-dev-opencl

# 4. Verify setup
echo "JAVA_HOME: $JAVA_HOME"
echo "TORNADO_SDK: $TORNADO_SDK"
java -version
ls $TORNADO_SDK/tornado-argfile
```

Expected output:
- Java version 21.0.9 (GraalVM)
- tornado-argfile exists

---

## Building TornadoVM

Only needed if you modified Java source code:

```bash
cd ~/TornadoVM-fork-2
export JAVA_HOME=$HOME/graalvm-jdk-21.0.9+7.1
export PATH=$JAVA_HOME/bin:$PATH
make
```

After build completes, update TORNADO_SDK:
```bash
export TORNADO_SDK=$HOME/TornadoVM-fork-2/dist/tornadovm-1.1.2-dev-opencl-linux-amd64/tornadovm-1.1.2-dev-opencl
```

---

## Running Benchmarks WITHOUT MCP Optimization

Basic benchmark (no LLM optimization):

```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.<BenchmarkClass> \
  kernels/<kernel>.cl <size>
```

### Examples:

**NBody (16K bodies):**
```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" uk.ac.manchester.tornado.examples.compute.custom.NBodySingleKernelBenchmark kernels/nbody/nbody_generated.cl 16384
```

**BlackScholes (1M options):**
```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" uk.ac.manchester.tornado.examples.compute.custom.BlackScholesSingleKernelBenchmark kernels/blackscholes_generated.cl 1048576
```

---

## Running Benchmarks WITH MCP Optimization

MCP optimization uses an LLM to automatically optimize kernels at runtime.

### Step 1: Start MCP Server (Separate Terminal)

Open a **new terminal**, SSH into storm, and run:

```bash
cd ~/TornadoVM-fork-2
export JAVA_HOME=$HOME/graalvm-jdk-21.0.9+7.1
export TORNADO_SDK=$HOME/TornadoVM-fork-2/dist/tornadovm-1.1.2-dev-opencl-linux-amd64/tornadovm-1.1.2-dev-opencl
python -m tornadovm_mcp.api.http_server 8090
```

Keep this terminal open. You should see:
```
 * Running on http://0.0.0.0:8090
```

### Step 2: Run Benchmark with MCP Flags (Main Terminal)

Add these JVM flags to enable MCP optimization:
- `-Dtornado.mcp.optimization=true`
- `-Dtornado.mcp.server.url=http://localhost:8090/optimize`

```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile \
  -Dtornado.mcp.optimization=true \
  -Dtornado.mcp.server.url=http://localhost:8090/optimize \
  -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" \
  uk.ac.manchester.tornado.examples.compute.custom.<BenchmarkClass> \
  kernels/<kernel>.cl <size>
```

---

## Available Benchmarks

| Algorithm | Benchmark Class | Kernel File | Size Param |
|-----------|-----------------|-------------|------------|
| NBody | `NBodySingleKernelBenchmark` | `kernels/nbody/nbody_generated.cl` | bodies (16384) |
| BlackScholes | `BlackScholesSingleKernelBenchmark` | `kernels/blackscholes_generated.cl` | options (4194304) |
| MatrixMul 1D | `MatrixMultiplication1DSingleKernelBenchmark` | `kernels/matrix1d_generated.cl` | matrix size (1024) |
| MonteCarlo | `MonteCarloSingleKernelBenchmark` | `kernels/montecarlo_generated.cl` | samples (4194304) |
| Mandelbrot | `MandelbrotSingleKernelBenchmark` | `kernels/mandelbrot_generated.cl` | image size (4096) |
| BlurFilter | `BlurFilterSingleKernelBenchmark` | `kernels/blurfilter_generated.cl` | image size (4096) |
| Reduction | `ReductionAddFloatsSingleKernelBenchmark` | `kernels/reduction_generated.cl` | elements (16777216) |

---

## Complete Examples with MCP Optimization

### NBody (16K bodies)
```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile -Dtornado.mcp.optimization=true -Dtornado.mcp.server.url=http://localhost:8090/optimize -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" uk.ac.manchester.tornado.examples.compute.custom.NBodySingleKernelBenchmark kernels/nbody/nbody_generated.cl 16384
```

### BlackScholes (4M options)
```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile -Dtornado.mcp.optimization=true -Dtornado.mcp.server.url=http://localhost:8090/optimize -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" uk.ac.manchester.tornado.examples.compute.custom.BlackScholesSingleKernelBenchmark kernels/blackscholes_generated.cl 4194304
```

### MatrixMul 1D (1024x1024)
```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile -Dtornado.mcp.optimization=true -Dtornado.mcp.server.url=http://localhost:8090/optimize -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" uk.ac.manchester.tornado.examples.compute.custom.MatrixMultiplication1DSingleKernelBenchmark kernels/matrix1d_generated.cl 1024
```

### MonteCarlo (4M samples)
```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile -Dtornado.mcp.optimization=true -Dtornado.mcp.server.url=http://localhost:8090/optimize -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" uk.ac.manchester.tornado.examples.compute.custom.MonteCarloSingleKernelBenchmark kernels/montecarlo_generated.cl 4194304
```

### Mandelbrot (4096x4096)
```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile -Dtornado.mcp.optimization=true -Dtornado.mcp.server.url=http://localhost:8090/optimize -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" uk.ac.manchester.tornado.examples.compute.custom.MandelbrotSingleKernelBenchmark kernels/mandelbrot_generated.cl 4096
```

### BlurFilter (4096x4096)
```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile -Dtornado.mcp.optimization=true -Dtornado.mcp.server.url=http://localhost:8090/optimize -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" uk.ac.manchester.tornado.examples.compute.custom.BlurFilterSingleKernelBenchmark kernels/blurfilter_generated.cl 4096
```

### Reduction (16M elements)
```bash
java --enable-preview @${TORNADO_SDK}/tornado-argfile -Dtornado.mcp.optimization=true -Dtornado.mcp.server.url=http://localhost:8090/optimize -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" uk.ac.manchester.tornado.examples.compute.custom.ReductionAddFloatsSingleKernelBenchmark kernels/reduction_generated.cl 16777216
```

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `JAVA_HOME env variable not defined` | JAVA_HOME not set | `export JAVA_HOME=$HOME/graalvm-jdk-21.0.9+7.1` |
| `could not open tornado-argfile` | TORNADO_SDK wrong or not set | Set correct TORNADO_SDK path |
| `ClassNotFoundException` | Code not compiled | Run `make` |
| `Connection refused` on port 8090 | MCP server not running | Start MCP server in separate terminal |
| `clCreateKernel -> Returned: -46` | Kernel entry point mismatch | Check kernel function name matches benchmark |
| `file does not exist: kernels/...` | Missing kernel file | Generate kernel with `tornado --printKernel` |

---

## MCP Optimization Logs

Optimization logs are saved to:
```
~/TornadoVM-fork-2/mcp_opt_<timestamp>.log
```

View the latest log:
```bash
ls -lt ~/TornadoVM-fork-2/mcp_opt_*.log | head -1
cat $(ls -t ~/TornadoVM-fork-2/mcp_opt_*.log | head -1)
```

---

## Quick Reference

```bash
# Setup (run after each SSH login)
cd ~/TornadoVM-fork-2
export JAVA_HOME=$HOME/graalvm-jdk-21.0.9+7.1
export PATH=$JAVA_HOME/bin:$PATH
export TORNADO_SDK=$HOME/TornadoVM-fork-2/dist/tornadovm-1.1.2-dev-opencl-linux-amd64/tornadovm-1.1.2-dev-opencl

# Start MCP server (separate terminal)
python -m tornadovm_mcp.api.http_server 8090

# Run benchmark with MCP optimization
java --enable-preview @${TORNADO_SDK}/tornado-argfile -Dtornado.mcp.optimization=true -Dtornado.mcp.server.url=http://localhost:8090/optimize -cp "bin/examples:${TORNADO_SDK}/share/java/tornado/*" uk.ac.manchester.tornado.examples.compute.custom.<BenchmarkClass> kernels/<kernel>.cl <size>
```

---

## Results Achieved

| Benchmark | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| NBody (16K) | - | - | 1.49x |
| BlackScholes (1M) | 0.012 ms | 0.004 ms | 3.20x |
