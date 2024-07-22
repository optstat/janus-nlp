cmake_minimum_required(VERSION 3.10)
project(JanusNLPProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "Suppress developer warnings")

# Find Python3 and NumPy
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)


# Set debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
# Add DEBUG definition
#add_definitions(-DDEBUG)

# Define a custom build type for profiling
set(CMAKE_BUILD_TYPE_PROFILE "Profile")
set(CMAKE_CXX_FLAGS_PROFILE "-pg")
set(CMAKE_EXE_LINKER_FLAGS_PROFILE "-pg")



enable_testing()
# Find libtorch
find_package(Torch REQUIRED)

# Add pthread
find_package(Threads REQUIRED)

# Define the source directory for headers
set(HEADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/cpp")
set(HEADER_DEST_DIR "/usr/local/include/janus")

# Custom command to copy headers
add_custom_target(copy_headers ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HEADER_DEST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${HEADER_SOURCE_DIR} ${HEADER_DEST_DIR}
    COMMENT "Copying header files to ${HEADER_DEST_DIR}"
)

# Add subdirectory for tests
add_subdirectory(tests/cpp)
add_subdirectory(examples/cpp)

# Install rules for header files
install(DIRECTORY ${HEADER_SOURCE_DIR}/ DESTINATION include/janus FILES_MATCHING PATTERN "*.hpp")
