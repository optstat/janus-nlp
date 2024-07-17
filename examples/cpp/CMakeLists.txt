cmake_minimum_required(VERSION 3.10)

project(VanDerPolExample)

# Include directories from parent
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/examples/cpp)

# Find libtorch
find_package(Torch REQUIRED)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")


# Enable AddressSanitizer
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")

# Optionally enable AddressSanitizer
option(ENABLE_ASAN "Enable AddressSanitizer" ON)

if(ENABLE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
endif()

# Specify the executable
set(SOURCE_FILE1 newt_vdp_example.cpp)

add_executable(vdpnewt ${SOURCE_FILE1})

# Set the output directory for the executable
set_target_properties(vdpnewt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Link libraries
target_link_libraries(vdpnewt "${TORCH_LIBRARIES}")
target_link_libraries(vdpnewt Python3::Python Python3::NumPy)



# Ensure the executable is built when running 'make run_vdphamexample'
add_custom_target(vdpnewtexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdpnewt
)

