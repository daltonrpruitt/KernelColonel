# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.20)


project(microbenchmarks CUDA CXX)

include(FindCUDAToolkit)

add_subdirectory(sandbox)

set(${PROJECT_NAME}_CUDA_ARCHS  52 ) # 52-Quadro M2000 or GTX 980M (personal laptop)
# set(${PROJECT_NAME}_CUDA_ARCHS 80) #80-A100
# set(${PROJECT_NAME}_CUDA_ARCHS 35 52 72 80) #35-K20m, 52-Quadro M2000, 72-V100, 80-A100  


include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/utilities/matrices)


add_executable(main src/main.cu)
set_property(TARGET main PROPERTY CXX_STANDARD 17)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES  ${PROJECT_NAME}_CUDA_ARCHS )
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep -Xptxas -dlcm=cg")
if (MSVC)
    target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler "/bigobj" >)
    # list(APPEND CUDA_NVCC_FLAGS  "--compiler-options;/bigobj")
# else ()
#   target_compile_options(main PRIVATE -Wa,-mbig-obj)
endif ()
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --keep 
                        -Xptxas -dlcm=cg
                        -Xcudafe "--diag_suppress=unsigned_compare_with_zero"
>)

if (NOT MSVC)
add_executable(spmv src/spmv.cu)
set_property(TARGET spmv PROPERTY CXX_STANDARD 17)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES ${PROJECT_NAME}_CUDA_ARCHS)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 80) #80-A100
# set_property(TARGET spmv PROPERTY CUDA_ARCHITECTURES 35 52 72 80) #35-K20m, 52-Quadro M2000, 72-V100, 80-A100  
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --maxrregcount=255") #  -dlcm=cg  --expt-relaxed-constexpr
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --expt-relaxed-constexpr
                        --maxrregcount=255
                        --keep
                        -Xptxas -dlcm=cg
                        -Xcudafe "--diag_suppress=set_but_not_used --diag_suppress=declared_but_not_referenced"
>)

if (MSVC)
    target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler "/bigobj" >)
# else ()
#   target_compile_options(spmv PRIVATE -Wa,-mbig-obj)
endif ()
         
endif ()

add_subdirectory(tests)
