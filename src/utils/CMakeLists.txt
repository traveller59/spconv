add_library(spconv_nms SHARED nms.cu)
set_target_properties(spconv_nms PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(spconv_nms PROPERTIES SOVERSION 1)
target_include_directories(spconv_nms PRIVATE ${ALL_INCLUDE})
set_property(TARGET spconv_nms PROPERTY CXX_STANDARD 14)
set_property(TARGET spconv_nms PROPERTY CUDA_STANDARD 14)
target_link_libraries(spconv_nms ${CUDA_CUDART})

install (TARGETS spconv_nms DESTINATION lib)

add_library(spconv_utils SHARED all.cc)
set_target_properties(spconv_utils PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(spconv_utils PROPERTIES SOVERSION 1)
target_include_directories(spconv_utils PRIVATE ${ALL_INCLUDE}
                    ${PROJECT_SOURCE_DIR}/third_party/pybind11/include)
set_property(TARGET spconv_utils PROPERTY CXX_STANDARD 14)
set_property(TARGET spconv_utils PROPERTY CUDA_STANDARD 14)
set_target_properties(spconv_utils PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
target_link_libraries(spconv_utils ${CUDA_CUDART} pybind11::module -Wl,--no-as-needed spconv_nms)

install (TARGETS spconv_utils DESTINATION lib)
