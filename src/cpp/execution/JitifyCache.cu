#include <execution/JitifyCache.cuh>

namespace KernelColonel
{
    jitify::JitCache &globalJitifyCache()
    {
        static jitify::JitCache kernel_cache;
        return kernel_cache;
    }
} // namespace KernelColonel
