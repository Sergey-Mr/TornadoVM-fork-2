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
import uk.ac.manchester.tornado.drivers.ptx.graal.PTXInstalledCode;
import uk.ac.manchester.tornado.runtime.common.RuntimeUtilities;

public class PTXCodeCache {

    private final PTXDeviceContext deviceContext;
    private final ConcurrentHashMap<String, PTXInstalledCode> cache;

    PTXCodeCache(PTXDeviceContext deviceContext) {
        this.deviceContext = deviceContext;
        cache = new ConcurrentHashMap<>();
    }

    public PTXInstalledCode installSource(String name, byte[] targetCode, String resolvedMethodName, boolean debugKernel) {

        if (!cache.containsKey(name)) {
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

    /**
     * Get the kernel source code for a cached kernel.
     * This is used for MCP kernel comparison - to extract the generated kernel
     * after first execution.
     *
     * @param name The kernel name/id
     * @return The kernel source code as a string, or null if not found
     */
    public String getKernelSource(String name) {
        System.out.println("[MCP DEBUG] Looking for kernel with name: " + name);
        System.out.println("[MCP DEBUG] Cache keys: " + cache.keySet());

        // Try exact match first
        PTXInstalledCode installedCode = cache.get(name);
        if (installedCode != null) {
            String source = installedCode.getGeneratedSourceCode();
            System.out.println("[MCP DEBUG] Found kernel (exact match), source length: " + (source != null ? source.length() : 0));
            return source;
        }

        // PTX cache uses fully mangled names (e.g., "s0_t0_matrixmultiplication_arrays_floatarray_...")
        // but the lookup uses the simple task ID (e.g., "s0.t0").
        // Fall back to prefix matching: sanitize the task ID (dots → underscores) and find a key that starts with it.
        String sanitized = name.replace(".", "_").toLowerCase();
        for (var entry : cache.entrySet()) {
            if (entry.getKey().startsWith(sanitized)) {
                String source = entry.getValue().getGeneratedSourceCode();
                System.out.println("[MCP DEBUG] Found kernel (prefix match on '" + sanitized + "'), key: " + entry.getKey()
                        + ", source length: " + (source != null ? source.length() : 0));
                return source;
            }
        }

        System.out.println("[MCP DEBUG] Kernel not found in cache");
        return null;
    }

    /**
     * Invalidate a cached kernel to force recompilation.
     * This is used for MCP kernel comparison - to replace the kernel with an optimized version.
     *
     * @param name The kernel name/id
     */
    public void invalidateKernel(String name) {
        PTXInstalledCode installedCode = cache.get(name);
        if (installedCode != null) {
            installedCode.invalidate();
            cache.remove(name);
            return;
        }
        // Prefix match fallback (same as getKernelSource)
        String sanitized = name.replace(".", "_").toLowerCase();
        String matchedKey = null;
        for (var entry : cache.entrySet()) {
            if (entry.getKey().startsWith(sanitized)) {
                matchedKey = entry.getKey();
                break;
            }
        }
        if (matchedKey != null) {
            cache.get(matchedKey).invalidate();
            cache.remove(matchedKey);
        }
    }

    /**
     * Replace a cached kernel with new source code.
     * This is used for MCP kernel comparison - run the optimized kernel in the same conditions.
     *
     * @param name The kernel name/id
     * @param resolvedMethodName The method/entry point name
     * @param newSource New kernel source code
     * @param debugKernel Whether to print debug info
     * @return The new installed code, or null if installation failed
     */
    public PTXInstalledCode replaceKernelSource(String name, String resolvedMethodName, String newSource, boolean debugKernel) {
        // Resolve the actual cache key (may be mangled)
        String actualKey = name;
        if (!cache.containsKey(name)) {
            String sanitized = name.replace(".", "_").toLowerCase();
            for (String key : cache.keySet()) {
                if (key.startsWith(sanitized)) {
                    actualKey = key;
                    break;
                }
            }
        }

        // Invalidate the existing kernel
        invalidateKernel(actualKey);

        // Install the new source under the resolved key
        byte[] sourceBytes = newSource.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        return installSource(actualKey, sourceBytes, resolvedMethodName, debugKernel);
    }
}
