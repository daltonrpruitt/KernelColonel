# For gtest tests
cmake_minimum_required(VERSION 3.20)

include(CTest REQUIRED)

include(FetchContent REQUIRED)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # For Windows: Prevent overriding the parent project's compiler/linker settings
FetchContent_MakeAvailable(googletest)

include(FindCUDA)

set(TEST_BINARY tests)

set(TEST_SOURCES_files 
    main.cpp
    JITCompilationTests.cu
    IKernelDataTests.cu
    SimpleKernelExecutionTests.cu
    SimpleKernelFileExecutionTests.cu
    SimpleCopyTests.cu
    KernelRunSetTests.cu
)

if(NOT MSVC)
    list(APPEND TypeNameTests.cpp)  # type_name not working in msvc (2023-06-10)
endif()

set(TEST_SOURCES "") 
foreach(src_file ${TEST_SOURCES_files})
    list(APPEND TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cpp/${src_file})
endforeach()

message(STATUS "TEST_SOURCES = ${TEST_SOURCES}")

add_executable(${TEST_BINARY} ${TEST_SOURCES})
# set_property(TARGET ${TEST_BINARY} PROPERTY CXX_STANDARD 17)
set(CUDA_COMPILE_OPTIONS "-Xptxas;-dlcm=cg")
if(NOT MSVC)
    list(APPEND CUDA_COMPILE_OPTIONS "--keep")  # error with keeping generated source on msvc (2023-07-20)
endif()
target_compile_options(${TEST_BINARY}  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    ${CUDA_COMPILE_OPTIONS}
>)

add_test(NAME ${TEST_BINARY} COMMAND ${TEST_BINARY})

target_link_libraries(${TEST_BINARY} PUBLIC gtest_main gmock_main kernel_colonel nvrtc cuda)
target_include_directories(${TEST_BINARY} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(${TEST_BINARY} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../src")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR/../src = ${CMAKE_CURRENT_SOURCE_DIR}/../src")

add_custom_target (${TEST_BINARY}.tgt
  DEPENDS ${TEST_BINARY}
  COMMAND ${TEST_BINARY} --gtest_output=xml:test_output/${TEST_BINARY}.xml
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
