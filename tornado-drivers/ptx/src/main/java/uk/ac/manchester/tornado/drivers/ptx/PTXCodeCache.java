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
import java.util.Set;

import uk.ac.manchester.tornado.api.exceptions.TornadoBailoutRuntimeException;
import uk.ac.manchester.tornado.drivers.ptx.graal.PTXInstalledCode;
import uk.ac.manchester.tornado.runtime.common.RuntimeUtilities;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.runtime.tasks.meta.TaskDataContext;

public class PTXCodeCache {

    private final PTXDeviceContext deviceContext;
    private final ConcurrentHashMap<String, PTXInstalledCode> cache;
    
    private static final Set<String> SUPPORTED_PTX_JIT_FLAGS =
            Set.of(
                    CUjitOption.CU_JIT_OPTIMIZATION_LEVEL.name(),
                    CUjitOption.CU_JIT_MAX_REGISTERS.name(),
                    CUjitOption.CU_JIT_CACHE_MODE.name(),
                    CUjitOption.CU_JIT_GENERATE_DEBUG_INFO.name(),
                    CUjitOption.CU_JIT_LOG_VERBOSE.name(),
                    CUjitOption.CU_JIT_GENERATE_LINE_INFO.name(),
                    CUjitOption.CU_JIT_TARGET.name()
            );

    PTXCodeCache(PTXDeviceContext deviceContext) {
        this.deviceContext = deviceContext;
        cache = new ConcurrentHashMap<>();
    }

    public PTXInstalledCode installSource(TaskDataContext taskMeta, String name, byte[] targetCode, String resolvedMethodName, boolean debugKernel) {

        if (!cache.containsKey(name)) {
            if (debugKernel) {
                RuntimeUtilities.dumpKernel(targetCode);
            }

            String compilerFlags = taskMeta.getCompilerFlags(TornadoVMBackendType.PTX);
            String[] parts = compilerFlags.trim().split("\\s+");

            if (parts.length % 2 != 0) {
                throw new TornadoBailoutRuntimeException(
                        "Malformed compilerFlags string: expected pairs of <flag> <value>. Got: " + compilerFlags
                );
            }

            int[] jitOptions = new int[parts.length / 2];
            long[] jitValues = new long[parts.length / 2];

            for (int i = 0; i < parts.length; i += 2) {
                String flagName = parts[i];

                if (!SUPPORTED_PTX_JIT_FLAGS.contains(flagName)) {
                    throw new TornadoBailoutRuntimeException(
                            "Unsupported PTX JIT compiler flag: " + flagName +
                                    ". Supported flags are: " + SUPPORTED_PTX_JIT_FLAGS
                    );
                }

                CUjitOption option;
                try {
                    option = CUjitOption.valueOf(flagName);
                } catch (IllegalArgumentException e) {
                    throw new TornadoBailoutRuntimeException(
                            "Invalid PTX JIT flag name: " + flagName, e
                    );
                }

                jitOptions[i / 2] = option.getValue();

                try {
                    jitValues[i / 2] = Long.parseLong(parts[i + 1]);
                } catch (NumberFormatException e) {
                    throw new TornadoBailoutRuntimeException(
                            "Invalid flag value (must be integer): '" + parts[i + 1] + "'", e
                    );
                }
            }

            PTXModule module = new PTXModule(resolvedMethodName, targetCode, name, jitOptions, jitValues);

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
