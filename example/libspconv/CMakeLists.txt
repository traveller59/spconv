
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(SpconvExample LANGUAGES CXX CUDA VERSION 0.1)

set(CUMM_DISABLE_CMAKE_INSTALL ON CACHE BOOL "enable X functionality" FORCE)

add_subdirectory(cumm)
add_subdirectory(spconv)

add_executable(main main.cu)
# SPCONV2_INCLUDE_PATH come from spconv/CMakeLists.txt
target_include_directories(main PRIVATE ${SPCONV2_INCLUDE_PATH})
target_link_libraries(main spconv cumm::cumm)
