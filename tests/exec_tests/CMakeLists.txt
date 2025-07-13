# For gtest tests
cmake_minimum_required(VERSION 3.20)

include(CTest REQUIRED)

add_test(NAME NoInput COMMAND ${CMAKE_BINARY_DIR}/kernel_colonel_exec  )
set_tests_properties(NoInput PROPERTIES WILL_FAIL true)


file(WRITE config.1.json "\
{
    \"test-configuration\": {
        \"
    }
}
")


add_test(NAME NoInput COMMAND ${CMAKE_BINARY_DIR}/kernel_colonel_exec  )
set_tests_properties(NoInput PROPERTIES WILL_FAIL true)
