cmake_minimum_required(VERSION 3.31)
project(MetalTesting)

set(CMAKE_CXX_STANDARD 20)
add_executable(MetalTesting main.cpp
)

add_subdirectory(metal-cmake)

# Shader file paths
set(SHADER_FILE ${CMAKE_SOURCE_DIR}/shader.metal)
set(AIR_FILE ${CMAKE_BINARY_DIR}/shader.air)
set(METALLIB_FILE ${CMAKE_BINARY_DIR}/default.metallib)

# Add custom command to compile .metal to .air
add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND xcrun -sdk macosx metal -c ${SHADER_FILE} -o ${AIR_FILE}
        DEPENDS ${SHADER_FILE}
        VERBATIM
        COMMENT "Compiling shader.metal to shader.air"
)

# Add custom command to compile .air to .metallib
add_custom_command(
        OUTPUT ${METALLIB_FILE}
        COMMAND xcrun -sdk macosx metallib ${AIR_FILE} -o ${METALLIB_FILE}
        DEPENDS ${AIR_FILE}
        VERBATIM
        COMMENT "Linking shader.air to default.metallib"
)
add_custom_target(ShaderLib ALL DEPENDS ${METALLIB_FILE})
add_dependencies(MetalTesting ShaderLib)

target_link_libraries(MetalTesting METAL_CPP)
