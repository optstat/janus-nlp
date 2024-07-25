cmake_minimum_required(VERSION 3.10)

project(VanDerPolExample)

# Include directories from parent
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/examples/cpp)

# Find libtorch
find_package(Torch REQUIRED)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


# Define a custom build type for profiling
set(CMAKE_BUILD_TYPE_PROFILE "Profile")
set(CMAKE_CXX_FLAGS_PROFILE "-pg")
set(CMAKE_EXE_LINKER_FLAGS_PROFILE "-pg")


set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")




# Enable AddressSanitizer
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")

# Optionally enable AddressSanitizer
#option(ENABLE_ASAN "Enable AddressSanitizer" ON)

#if(ENABLE_ASAN)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
#    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
#endif()

# Specify the executable
set(SOURCE_FILE1 newted_vdp_example.cpp)
set(SOURCE_FILE2 newt_dubins_example.cpp)
set(SOURCE_FILE3 newted_control_example.cpp)

add_executable(vdpnewtexample ${SOURCE_FILE1})
add_executable(dubinsnewtexample ${SOURCE_FILE2})
add_executable(controldnewtexample ${SOURCE_FILE3})


# Set the output directory for the executable
set_target_properties(vdpnewtexample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
# Set the output directory for the executable
set_target_properties(dubinsnewtexample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Set the output directory for the executable
set_target_properties(controldnewtexample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)


# Link libraries
target_link_libraries(vdpnewtexample "${TORCH_LIBRARIES}")
target_link_libraries(vdpnewtexample Python3::Python Python3::NumPy)


target_link_libraries(dubinsnewtexample "${TORCH_LIBRARIES}")
target_link_libraries(dubinsnewtexample Python3::Python Python3::NumPy)

target_link_libraries(controldnewtexample "${TORCH_LIBRARIES}")
target_link_libraries(controldnewtexample Python3::Python Python3::NumPy)

# Ensure the executable is built when running 'make run_vdphamexample'
add_custom_target(create_vdpnewtexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target vdpnewtexample
)

add_custom_target(create_dubinsnewtexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target dubinsnewtexample
)

add_custom_target(create_controldnewtexample
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target constroldnewtexample
)
