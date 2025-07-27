cmake_minimum_required(VERSION 3.31)
project(AppleMetal)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
        LibDataTypes
        GIT_REPOSITORY https://github.com/ThePhoneBeLinging/LibDataTypes
        GIT_TAG main
)
FetchContent_MakeAvailable(LibDataTypes)

add_subdirectory(metal-cmake)
add_subdirectory(src)

