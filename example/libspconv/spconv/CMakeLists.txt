cmake_minimum_required(VERSION 3.20)

project(spconv LANGUAGES CXX CUDA)
include_directories(include)

add_subdirectory(src)
# tell parent spconv2 include path.
set(SPCONV2_INCLUDE_PATH ${${PROJECT_NAME}_SOURCE_DIR}/include PARENT_SCOPE)

install (TARGETS spconv
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION bin)



