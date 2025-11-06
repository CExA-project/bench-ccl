cmake_minimum_required(VERSION 3.23)
project(benchmark-ccls LANGUAGES CXX)

# Add cmake helpers
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)

add_subdirectory(tpls/kokkos)
add_subdirectory(tpls/googletest)
add_subdirectory(tpls/benchmark)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(benchmark-ccls BenchmarkMain.cpp BenchmarkAll2All.cpp)

if(Kokkos_ENABLE_CUDA)
  find_package(NCCL REQUIRED)
  target_link_libraries(benchmark-ccls PUBLIC NCCL::NCCL Kokkos::kokkos MPI::MPI_CXX benchmark::benchmark)
elseif(Kokkos_ENABLE_HIP)
  find_package(NCCL REQUIRED)
  target_link_libraries(benchmark-ccls PUBLIC roc::rccl Kokkos::kokkos MPI::MPI_CXX benchmark::benchmark)
elseif(Kokkos_ENABLE_SYCL)
  find_package(oneCCL REQUIRED)
  target_link_libraries(benchmark-ccls PUBLIC ${oneCCL_LIBRARIES} Kokkos::kokkos MPI::MPI_CXX benchmark::benchmark)
  target_include_directories(benchmark-ccls PUBLIC ${oneCCL_INCLUDE_DIRS})
endif()
