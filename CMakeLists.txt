# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.20)


project(microbenchmarks LANGUAGES CUDA CXX)

include(FindCUDAToolkit)

add_subdirectory(sandbox)

set(${PROJECT_NAME}_CUDA_ARCHS  native )


add_library(kernel_colonel INTERFACE )
target_include_directories(kernel_colonel INTERFACE ${CMAKE_SOURCE_DIR}/src/include)
# set_target_properties(kernel_colonel PROPERTIES LINKER_LANGUAGE CUDA)

set_property(TARGET kernel_colonel PROPERTY CXX_STANDARD 17)
set_property(TARGET kernel_colonel PROPERTY CUDA_ARCHITECTURES  ${${PROJECT_NAME}_CUDA_ARCHS})
# target_link_libraries(kernel_colonel PRIVATE ${CUDA_LIBRARIES})

if(false)
add_executable(main src/main.cu)
set_property(TARGET main PROPERTY CXX_STANDARD 17)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES  ${${PROJECT_NAME}_CUDA_ARCHS} )
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --keep 
                        -Xptxas -dlcm=cg
                        -Xcudafe "--diag_suppress=unsigned_compare_with_zero"
>)

add_executable(spmv src/spmv.cu)
set_property(TARGET spmv PROPERTY CXX_STANDARD 17)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES  ${${PROJECT_NAME}_CUDA_ARCHS})
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

endif ()

add_subdirectory(tests)
