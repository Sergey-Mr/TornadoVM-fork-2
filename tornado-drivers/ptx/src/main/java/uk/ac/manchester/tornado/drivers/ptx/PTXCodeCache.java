/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */
package uk.ac.manchester.tornado.drivers.ptx;

import java.util.concurrent.ConcurrentHashMap;

import uk.ac.manchester.tornado.api.exceptions.TornadoBailoutRuntimeException;
import uk.ac.manchester.tornado.api.mcp.MCPKernelOptimizer;
import uk.ac.manchester.tornado.drivers.ptx.graal.PTXInstalledCode;
import uk.ac.manchester.tornado.runtime.common.RuntimeUtilities;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;

public class PTXCodeCache {

    private final PTXDeviceContext deviceContext;
    private final ConcurrentHashMap<String, PTXInstalledCode> cache;

    // MCP Kernel Optimizer (lazy initialized, shared across instances)
    private static MCPKernelOptimizer mcpOptimizer;
    private static boolean mcpInitialized = false;
    private static final Object mcpLock = new Object();

    PTXCodeCache(PTXDeviceContext deviceContext) {
        this.deviceContext = deviceContext;
        cache = new ConcurrentHashMap<>();
    }

    /**
     * Get or initialize the MCP optimizer (thread-safe, lazy initialization).
     */
    private static MCPKernelOptimizer getMCPOptimizer() {
        if (!mcpInitialized) {
            synchronized (mcpLock) {
                if (!mcpInitialized) {
                    if (TornadoOptions.MCP_OPTIMIZATION_ENABLED && TornadoOptions.MCP_SERVER_PATH != null) {
                        try {
                            mcpOptimizer = new MCPKernelOptimizer();
                            mcpOptimizer.start();
                            System.out.println("[TornadoVM-MCP] PTX Kernel optimizer initialized");
                        } catch (Exception e) {
                            System.err.println("[TornadoVM-MCP] Failed to initialize PTX optimizer: " + e.getMessage());
                            mcpOptimizer = null;
                        }
                    }
                    mcpInitialized = true;
                }
            }
        }
        return mcpOptimizer;
    }

    /**
     * Attempt to optimize PTX kernel source using MCP server.
     */
    private byte[] optimizeKernelWithMCP(byte[] source, String entryPoint) {
        if (!TornadoOptions.MCP_OPTIMIZATION_ENABLED) {
            return source;
        }

        MCPKernelOptimizer optimizer = getMCPOptimizer();
        if (optimizer == null) {
            return source;
        }

        try {
            String kernelCode = new String(source, java.nio.charset.StandardCharsets.UTF_8);
            String deviceFamily = detectDeviceFamily();

            // Initial optimization without profiling data
            String optimizedCode = optimizer.optimizeKernel(
                kernelCode,
                "ptx",
                deviceFamily,
                0,  // kernel_time_ns - unknown at compile time
                0,  // copy_in_time_ns
                0,  // copy_out_time_ns
                0,  // copy_in_bytes
                0,  // copy_out_bytes
                null,  // global_work_size
                null   // local_work_size
            );

            if (optimizedCode != null && !optimizedCode.equals(kernelCode)) {
                System.out.println("[TornadoVM-MCP] PTX Kernel " + entryPoint + " optimized successfully");
                return optimizedCode.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            }
        } catch (Exception e) {
            System.err.println("[TornadoVM-MCP] PTX optimization failed for " + entryPoint + ": " + e.getMessage());
        }

        return source;
    }

    /**
     * Detect NVIDIA device family based on device context.
     */
    private String detectDeviceFamily() {
        String deviceName = deviceContext.getDevice().getDeviceName().toLowerCase();

        if (deviceName.contains("4090") || deviceName.contains("4080") || deviceName.contains("4070")) {
            return "nvidia_ada";
        } else if (deviceName.contains("3090") || deviceName.contains("3080") || deviceName.contains("3070")) {
            return "nvidia_ampere";
        } else if (deviceName.contains("a100") || deviceName.contains("a10")) {
            return "nvidia_ampere_datacenter";
        } else if (deviceName.contains("h100")) {
            return "nvidia_hopper";
        }
        return "nvidia_generic";
    }

    public PTXInstalledCode installSource(String name, byte[] targetCode, String resolvedMethodName, boolean debugKernel) {

        if (!cache.containsKey(name)) {
            // Apply MCP optimization if enabled
            targetCode = optimizeKernelWithMCP(targetCode, resolvedMethodName);

            if (debugKernel) {
                RuntimeUtilities.dumpKernel(targetCode);
            }

            PTXModule module = new PTXModule(resolvedMethodName, targetCode, name);

            if (module.isPTXJITSuccess()) {
                PTXInstalledCode code = new PTXInstalledCode(name, module, deviceContext);
                cache.put(name, code);
                return code;
            } else {
                throw new TornadoBailoutRuntimeException("PTX JIT compilation failed!");
            }
        }

        return cache.get(name);
    }

    PTXInstalledCode getCachedCode(String name) {
        return cache.get(name);
    }

    boolean isCached(String name) {
        return cache.containsKey(name);
    }

    void reset() {
        for (PTXInstalledCode code : cache.values()) {
            code.invalidate();
        }
        cache.clear();
    }
}
