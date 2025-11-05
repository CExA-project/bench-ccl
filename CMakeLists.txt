cmake_minimum_required(VERSION 3.23)
project(benchmark-ccls LANGUAGES CXX)

# Add cmake helpers
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)

add_subdirectory(tpls/kokkos)
add_subdirectory(tpls/googletest)
add_subdirectory(tpls/benchmark)

find_package(oneCCL REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(oneccl-benchmark BenchmarkMain.cpp BenchmarkAll2All.cpp)

target_link_libraries(oneccl-benchmark PUBLIC ${oneCCL_LIBRARIES} Kokkos::kokkos MPI::MPI_CXX benchmark::benchmark)
target_include_directories(oneccl-benchmark PUBLIC ${oneCCL_INCLUDE_DIRS})
