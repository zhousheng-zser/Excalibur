include_directories(../../include)
add_executable(speedbenchmark ../../src/SpeedBenchmark/speedbenchmark.cpp)

target_link_libraries(speedbenchmark Primitives Excalibur)
add_dependencies(speedbenchmark Primitives Excalibur)
