set(CATCH_HEADER ${PROJECT_SOURCE_DIR}/third_party/catch2)
add_library(catch_main OBJECT src/catch_main.cpp)
# target_compile_features(catch_main PUBLIC cxx_std_2a)
set_property(TARGET catch_main PROPERTY CXX_STANDARD 14)
target_include_directories(catch_main PRIVATE ${CATCH_HEADER})

file(GLOB files "src/test_*.cpp")
foreach(file ${files})
    get_filename_component(file_basename ${file} NAME_WE)
    string(REGEX REPLACE "test_([^$]+)" "test-\\1" testcase ${file_basename})

    add_executable(${testcase} ${file} $<TARGET_OBJECTS:catch_main>)
    set_property(TARGET ${testcase} PROPERTY CXX_STANDARD 14)
    # set_target_properties(${testcase} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # set_property(TARGET ${testcase} PROPERTY CUDA_STANDARD 14)

    target_compile_definitions(${testcase} PRIVATE
        CATCH_CONFIG_FAST_COMPILE
    )
    target_include_directories(${testcase} PRIVATE
        ${CATCH_HEADER} ${ALL_INCLUDE}
    )
    target_link_libraries(${testcase} ${ALL_LIBS} pybind11::embed -Wl,--no-as-needed spconv)
    add_test(NAME "${testcase}" 
        COMMAND ${testcase} 
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()