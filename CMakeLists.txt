# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    SET(CMAKE_CUDA_ARCHITECTURES native)
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

project(microbenchmarks LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE "Debug")

include(FetchContent REQUIRED)

FetchContent_Declare(json
    # GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE  
    GIT_TAG bc889afb4c5bf1c0d8ee29ef35eaaf4c8bef8a5d # v3.11.2
)
FetchContent_MakeAvailable(json)

set(nlohmann_json_VERSION v3.11.2)
FetchContent_Declare(json_schema_validator
    GIT_REPOSITORY https://github.com/pboettch/json-schema-validator
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE  
    GIT_TAG 349cba9f7e3cb423bbc1811bdd9f6770f520b468 # 2.3.0
)
FetchContent_MakeAvailable(json_schema_validator)

FetchContent_Declare(
  jitify
  GIT_REPOSITORY https://github.com/NVIDIA/jitify
  GIT_TAG        0d0954efc5b3efa5224589ab91c53e1de9512201
#   CONFIGURE_COMMAND ""
#   BUILD_COMMAND ""
)
FetchContent_MakeAvailable(jitify)
message(STATUS "jitify_SOURCE_DIR=${jitify_SOURCE_DIR}")
FetchContent_GetProperties(jitify)
if(NOT jitify_POPULATED)
  FetchContent_Populate(jitify)
endif()

add_library(jitify INTERFACE)
target_include_directories(jitify INTERFACE ${jitify_SOURCE_DIR})
if(MSVC)
    target_compile_definitions(jitify INTERFACE _MSC_VER=${MSVC_VERSION})
    message(STATUS "MSVC_VERSION=${MSVC_VERSION}")
    target_link_libraries(jitify INTERFACE dbghelp.dll)
endif()

include(FindCUDAToolkit)
include(FindCUDA)

set(CMAKE_CXX_STANDARD 20)

set(kernel_colonel_SOURCE_BASE_DIR src/cpp)
set(kernel_colonel_SOURCES
   ${kernel_colonel_SOURCE_BASE_DIR}/config.cpp
   ${kernel_colonel_SOURCE_BASE_DIR}/execution/JitifyCache.cu
)
message(STATUS "kernel_colonel_SOURCES=${kernel_colonel_SOURCES}")

add_library(kernel_colonel STATIC ${kernel_colonel_SOURCES})
target_include_directories(kernel_colonel PUBLIC ${CMAKE_SOURCE_DIR}/src/include)
target_link_libraries(kernel_colonel INTERFACE jitify)
target_include_directories(kernel_colonel PRIVATE ${jitify_SOURCE_DIR} ${CUDA_LIBRARIES})
target_link_libraries(kernel_colonel PUBLIC nlohmann_json::nlohmann_json nlohmann_json_schema_validator nvrtc cuda)
set_target_properties(kernel_colonel PROPERTIES LINKER_LANGUAGE CUDA)

# set_property(TARGET kernel_colonel PROPERTY CXX_STANDARD 17)
# target_link_libraries(kernel_colonel PRIVATE ${CUDA_LIBRARIES})

add_executable(kernel_colonel_exec src/cpp/main.cpp)
target_link_libraries(kernel_colonel_exec PRIVATE kernel_colonel)


if(false)
add_executable(main src/main.cu)
# set_property(TARGET main PROPERTY CXX_STANDARD 17)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --keep 
                        -Xptxas -dlcm=cg
                        -Xcudafe "--diag_suppress=unsigned_compare_with_zero"
>)

add_executable(spmv src/spmv.cu)
# set_property(TARGET spmv PROPERTY CXX_STANDARD 17)
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

add_custom_target (run_sample_config
  DEPENDS kernel_colonel_exec
  COMMAND $<TARGET_FILE:kernel_colonel_exec> ${CMAKE_CURRENT_SOURCE_DIR}/examples/example_config.json
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

if(FALSE)
function (_get_all_cmake_targets out_var current_dir)
    get_property(targets DIRECTORY ${current_dir} PROPERTY BUILDSYSTEM_TARGETS)
    get_property(subdirs DIRECTORY ${current_dir} PROPERTY SUBDIRECTORIES)

    foreach(subdir ${subdirs})
        _get_all_cmake_targets(subdir_targets ${subdir})
        list(APPEND targets ${subdir_targets})
    endforeach()

    set(${out_var} ${targets} PARENT_SCOPE)
endfunction()

# Run at end of top-level CMakeLists
_get_all_cmake_targets(all_targets ${CMAKE_CURRENT_LIST_DIR})
message(STATUS "ALL TARGETS :")
foreach(target ${all_targets})
    message(STATUS "    ${target}")
endforeach()
endif()
