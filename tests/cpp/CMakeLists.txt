# Find libtorch
find_package(Torch REQUIRED)

# Specify the executable
set(SOURCE_FILE nlp_solver_test.cpp)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_executable(nlp_solver_test ${SOURCE_FILE})

# Include GTest
find_package(GTest REQUIRED)
include_directories(src/cpp ${GTEST_INCLUDE_DIRS})

# Link GTest
target_link_libraries(nlp_solver_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

# Link libtorch
target_link_libraries(nlp_solver_test "${TORCH_LIBRARIES}")


set_property(TARGET nlp_solver_test PROPERTY CXX_STANDARD 17)

add_test(NAME NLPTest COMMAND ${CMAKE_BINARY_DIR}/bin/nlp_solver_test)

# Ensure the test executable is built when running 'make run_tests'
add_custom_target(all_tests
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target nlp_solver_test
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
)


