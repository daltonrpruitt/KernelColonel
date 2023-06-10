#include <config.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <execution/kc_jitify.hpp>

namespace KernelColonel
{
    jitify::JitCache &globalJitifyCache();
} // namespace KernelColonel
