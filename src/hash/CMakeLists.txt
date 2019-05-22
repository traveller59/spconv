add_library(cudahash SHARED hash_functions.cu hash_table.cpp hash_table.cu 
                            mt19937ar.cpp)

target_include_directories(cudahash PRIVATE ${ALL_INCLUDE} )
set_property(TARGET cudahash PROPERTY CUDA_STANDARD 14)
set_property(TARGET cudahash PROPERTY CXX_STANDARD 14)
set_target_properties(cudahash PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudahash PRIVATE ${ALL_LIBS})
install (TARGETS cudahash DESTINATION lib)

add_executable(cudahash_test main.cc)

target_include_directories(cudahash_test PRIVATE ${ALL_INCLUDE} )
set_property(TARGET cudahash_test PROPERTY CUDA_STANDARD 14)
set_property(TARGET cudahash_test PROPERTY CXX_STANDARD 14)
set_target_properties(cudahash_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudahash_test PRIVATE ${ALL_LIBS} cudahash)
install (TARGETS cudahash_test DESTINATION bin)
