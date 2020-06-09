set(ALL_FILES ${ALL_FILES} gemm.cu)

add_library(spgemm SHARED ${ALL_FILES})

target_include_directories(spgemm PRIVATE ${ALL_INCLUDE} ${MP11_INCLUDE} ${CUTLASS_INCLUDE} )
set_property(TARGET spgemm PROPERTY CUDA_STANDARD 14)
set_property(TARGET spgemm PROPERTY CXX_STANDARD 14)
set_target_properties(spgemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(spgemm PRIVATE ${ALL_LIBS})
install (TARGETS spgemm DESTINATION lib)
