#include <execution/JitifyCache.cuh>
#include <unordered_map>


namespace KernelColonel
{
    jitify::JitCache &globalJitifyCache()
    {
        static jitify::JitCache kernel_cache;
        return kernel_cache;
    }

    // jitify::Program &getProgramFromJitifyCache(const std::string &program_name, const std::string &program_source){
    //     static std::unordered_map<std::string, jitify::Program> program_map;

    //     if(program_map.count(program_name) > 0 ) return program_map.at(program_name);

    //     auto cache = globalJitifyCache();
    //     auto program = cache.program(program_source);
    // }

} // namespace KernelColonel
