cmake_minimum_required(VERSION 3.14.3)
set(module_name cpp_example)
find_package(OpenCV REQUIRED)

include_directories(../../include)
file(GLOB src ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CppExample/*.cpp)
file(GLOB header ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CppExample/*.hpp)
add_executable(${module_name} ${src} ${header})

target_link_libraries(${module_name} primitives excalibur ${OpenCV_LIBS})
add_dependencies(${module_name} primitives excalibur)
