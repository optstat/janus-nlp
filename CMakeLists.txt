cmake_minimum_required(VERSION 3.10)
project(janus_nlp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "Suppress developer warnings")
set(Torch_DIR "/home/panos/Applications/libtorch/share/cmake/Torch")

# Find Python3 and NumPy
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
# Find Pybind11
find_package(pybind11 CONFIG REQUIRED)


include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src/cpp)
include_directories(${CMAKE_SOURCE_DIR}/examples/cpp)
include_directories(${TORCH_INCLUDE_DIRS})  # Include Torch headers


# Set debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
# Add DEBUG definition
#add_definitions(-DDEBUG)

# Define a custom build type for profiling
set(CMAKE_BUILD_TYPE_PROFILE "Profile")
set(CMAKE_CXX_FLAGS_PROFILE "-pg")
set(CMAKE_EXE_LINKER_FLAGS_PROFILE "-pg")

#Change this to 

# Find libtorch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# Add pthread
find_package(Threads REQUIRED)

# Add the shared library
add_library(janus_nlp MODULE src/cpp/bindings.cpp)


target_link_libraries(janus_nlp PRIVATE pybind11::module ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

# Ensure the extension is named correctly for Python import
set_target_properties(janus_nlp PROPERTIES PREFIX "" SUFFIX "${PYTHON_MODULE_EXTENSION}")



# Define the source directory for headers
set(HEADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/cpp")
set(HEADER_DEST_DIR "/usr/local/include/janus")
# Add an option to control the installation of headers
option(INSTALL_HEADERS "Install header files to ${HEADER_DEST_DIR}" OFF)

        

# Add subdirectory for tests
add_subdirectory(tests/cpp)


# Install rules for header files
install(DIRECTORY ${HEADER_SOURCE_DIR}/ DESTINATION include/janus FILES_MATCHING PATTERN "*.hpp")
# Conditionally install headers during make install
if(INSTALL_HEADERS)
    install(DIRECTORY ${HEADER_SOURCE_DIR}/ DESTINATION ${HEADER_DEST_DIR} FILES_MATCHING PATTERN "*.hpp")
endif()