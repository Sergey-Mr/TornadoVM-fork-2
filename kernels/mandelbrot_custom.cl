#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void mandelbrotTornado(__global long *_kernel_context,
                                __constant uchar *_constant_region,
                                __local uchar *_local_region,
                                __global int *_atomics,
                                __private int size,
                                __global uchar *output)
{
    // --- Typed pointer ---
    __global short *out = (__global short *)output;

    // --- Geometry ---
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int sx = get_global_size(0);
    const int sy = get_global_size(1);

    // --- Constants ---
    const float invSize = 1.0f / (float)size;
    const float scaleX  = 3.0f;            // real range (-1.5..1.5)
    const float scaleY  = 2.0f;            // imag range (-1..1)
    const int   maxIter = 10000;

    // --- Outer 2D loops (same stepping as generated kernel) ---
    for (int y = gy; y < size; y += sy) {
        const float cy = (y * invSize - 1.0f) * scaleY; // imaginary part
        const int rowBase = y * size + 12;             // preserve +12 offset

        for (int x = gx; x < size; x += sx) {
            const float cx = (x * invSize - 1.5f) * scaleX / 3.0f; // real part

            float zx = 0.0f;
            float zy = 0.0f;
            int iter = 0;

            // --- Main iteration loop (unrolled 2× for ILP) ---
            for (int i = 0; i < maxIter; i += 2) {
                // 1st iteration
                float zx2 = zx * zx;
                float zy2 = zy * zy;
                if (zx2 + zy2 > 4.0f) break;
                float tmp = fma(2.0f * zx, zy, cy);
                zx = zx2 - zy2 + cx;
                zy = tmp;

                // 2nd iteration
                zx2 = zx * zx;
                zy2 = zy * zy;
                if (zx2 + zy2 > 4.0f) { iter = i + 1; break; }
                tmp = fma(2.0f * zx, zy, cy);
                zx = zx2 - zy2 + cx;
                zy = tmp;

                iter = i + 2;
            }

            // --- Map iteration count to grayscale short (same semantics) ---
            const int idx = rowBase + x;
            const int val = ((iter << 8) - iter) / maxIter;
            out[idx] = (short)val;
        }
    }
}
