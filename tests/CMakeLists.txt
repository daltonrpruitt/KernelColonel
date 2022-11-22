# For gtest tests
cmake_minimum_required(VERSION 3.20)


include(CTest REQUIRED)

include(FetchContent REQUIRED)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(FindCUDA)

set(TEST_BINARY tests)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cuh *.cpp *.cc *.cu)

set(SOURCES ${TEST_SOURCES})
message("SOURCES = ${SOURCES}")

add_executable(${TEST_BINARY} ${TEST_SOURCES})
set_property(TARGET ${TEST_BINARY} PROPERTY CUDA_ARCHITECTURES ${${PROJECT_NAME}_CUDA_ARCHS})

add_test(NAME ${TEST_BINARY} COMMAND ${TEST_BINARY})

target_link_libraries(${TEST_BINARY} PUBLIC gtest_main gmock_main ${CUDA_LIBRARIES} kernel_colonel) # ${CMAKE_PROJECT_NAME}_lib gtest)
if(MSVC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")
endif()

add_custom_target (${TEST_BINARY}.tgt
  DEPENDS ${TEST_BINARY}
  COMMAND ${TEST_BINARY} --gtest_output=xml:test_output/${TEST_BINARY}.xml
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
