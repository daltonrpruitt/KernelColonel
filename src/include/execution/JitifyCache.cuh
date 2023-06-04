#include <config.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef LINUX // Only supported by gcc on Linux (defined in Makefile)
#define JITIFY_ENABLE_EMBEDDED_FILES 1
#endif

#define VERBOSE
#ifdef VERBOSE
#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE 1
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_PTX 1
#define JITIFY_PRINT_LINKER_LOG 1
#define JITIFY_PRINT_LAUNCH 1
#endif // VERBOSE
#include "jitify.hpp"

namespace KernelColonel
{
    jitify::JitCache &globalJitifyCache();
} // namespace KernelColonel
