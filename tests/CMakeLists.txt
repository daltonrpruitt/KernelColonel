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

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.c*)

message("TEST_SOURCES = ${TEST_SOURCES}")

add_executable(${TEST_BINARY} ${TEST_SOURCES})
set_property(TARGET ${TEST_BINARY} PROPERTY CUDA_ARCHITECTURES ${${PROJECT_NAME}_CUDA_ARCHS})
set_property(TARGET ${TEST_BINARY} PROPERTY CXX_STANDARD 17)

add_test(NAME ${TEST_BINARY} COMMAND ${TEST_BINARY})

target_link_libraries(${TEST_BINARY} PUBLIC gtest_main gmock_main kernel_colonel nvrtc cuda)
target_include_directories(${TEST_BINARY} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_custom_target (${TEST_BINARY}.tgt
  DEPENDS ${TEST_BINARY}
  COMMAND ${TEST_BINARY} --gtest_output=xml:test_output/${TEST_BINARY}.xml
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
