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

set(BINARY test)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp *.cc)

set(SOURCES ${TEST_SOURCES})
message("SOURCES = ${SOURCES}")

add_executable(${BINARY} ${TEST_SOURCES})
set_property(TARGET ${BINARY} PROPERTY CUDA_ARCHITECTURES ${PROJECT_NAME}_CUDA_ARCHS)

add_test(NAME ${BINARY} COMMAND ${BINARY})


target_link_libraries(${BINARY} PUBLIC gtest ${CUDA_LIBRARIES}) # ${CMAKE_PROJECT_NAME}_lib gtest)
if(MSVC)
  set(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")
endif()

# add_test(BINARY

# add_executable(
#   driver_test
#   driver_test.cc
# )

# target_link_libraries(
#   driver_test
#   GTest::gtest_main
# )

# file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)

# include(GoogleTest)
# gtest_discover_tests(driver_test)

# Testing
# enable_testing()

# add_test(driver_test driver_test)
