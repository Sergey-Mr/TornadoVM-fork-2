/*
 * Copyright (c) 2013-2023, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package uk.ac.manchester.tornado.api;

import java.util.Set;

import uk.ac.manchester.tornado.api.common.SchedulableTask;
import uk.ac.manchester.tornado.api.memory.TornadoMemoryProvider;

public interface TornadoDeviceContext {

    TornadoTargetDevice getDevice();

    TornadoMemoryProvider getMemoryManager();

    boolean wasReset();

    void reset(long executionPlanId);

    void setResetToFalse();

    boolean isPlatformFPGA();

    boolean isPlatformXilinxFPGA();

    boolean isFP64Supported();

    boolean isCached(long executionPlanId, String methodName, SchedulableTask task);

    int getDeviceIndex();

    int getDevicePlatform();

    String getDeviceName();

    int getDriverIndex();

    Set<Long> getRegisteredPlanIds();

    // =========================================================================
    // MCP Kernel Comparison Support
    // =========================================================================

    /**
     * Get the kernel source code for a cached kernel.
     *
     * @param executionPlanId The execution plan ID
     * @param taskId The task ID
     * @param entryPoint The kernel entry point name
     * @return The kernel source code, or null if not found
     */
    String getKernelSource(long executionPlanId, String taskId, String entryPoint);

    /**
     * Replace a cached kernel with new source code.
     *
     * @param executionPlanId The execution plan ID
     * @param taskId The task ID
     * @param entryPoint The kernel entry point name
     * @param newKernelSource The new kernel source code
     * @param meta Task metadata (can be null)
     * @return true if replacement was successful
     */
    boolean replaceKernelSource(long executionPlanId, String taskId, String entryPoint, String newKernelSource, Object meta);
}
