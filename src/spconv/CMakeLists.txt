add_library(spconv SHARED all.cc indice.cc indice.cu 
            reordering.cc reordering.cu maxpool.cc maxpool.cu)

target_include_directories(spconv PRIVATE ${ALL_INCLUDE} )
set_property(TARGET spconv PROPERTY CUDA_STANDARD 14)
set_property(TARGET spconv PROPERTY CXX_STANDARD 14)
set_target_properties(spconv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(spconv PRIVATE ${ALL_LIBS})
install (TARGETS spconv DESTINATION lib)
