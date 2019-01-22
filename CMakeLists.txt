cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(SparseConv LANGUAGES CXX CUDA VERSION 1.0)

option(SPCONV_BuildTests "Build the unit tests when BUILD_TESTING is enabled." ON)
set(CMAKE_CXX_EXTENSIONS OFF) # avoid gnu++11 be added to CXX flags

set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}")
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
if(WIN32) # true if windows (32 and 64 bit)
    set(CUDA_LIB_PATH_HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
    add_compile_definitions(TV_WINDOWS)
else()
    set(CUDA_LIB_PATH_HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
endif()

find_library(CUDA_CUDART NAMES cudart HINTS ${CUDA_LIB_PATH_HINTS})
find_library(CUDA_CUBLAS NAMES cublas HINTS ${CUDA_LIB_PATH_HINTS})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(TV_DEBUG)
endif()

find_package(Torch REQUIRED)

# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_compile_definitions(SPCONV_CUDA)
add_subdirectory(third_party/pybind11)

set(ALL_LIBS ${CUDA_CUDART} ${CUDA_CUBLAS} ${TORCH_LIBRARIES}) 

set(ALL_INCLUDE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
    ${PROJECT_SOURCE_DIR}/include)


add_subdirectory(src/spconv)
add_subdirectory(src/utils)

if (SPCONV_BuildTests)
    include(CTest) #adds option BUILD_TESTING (default ON)
    if(BUILD_TESTING)
        enable_testing()
        add_subdirectory(test)
    endif()
endif()
