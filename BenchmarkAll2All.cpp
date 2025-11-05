#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include "oneccl_wrapper.hpp"

namespace {

 struct CommTag {
   struct MPI_Tag {};
   struct oneCCL_Tag {};
 };

template <typename T, typename LayoutType, typename ArgComm>
void benchmark_all2all(benchmark::State& state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_all2all needs at least 2 ranks");
  }

  const int n = state.range(0);
  if (n % size != 0) {
    state.SkipWithError("Input size must be divisible by number of ranks");
  }

  using View3DType =
      Kokkos::View<T***, LayoutType, Kokkos::DefaultExecutionSpace>;

  int n0_buffer = 0, n1_buffer = 0, n2_buffer = 0;
  if constexpr (std::is_same<LayoutType, Kokkos::LayoutLeft>::value) {
    n0_buffer = n / size;
    n1_buffer = n;
    n2_buffer = size;
  } else {
    n0_buffer = size;
    n1_buffer = n;
    n2_buffer = n / size;
  }
  View3DType send("send", n0_buffer, n1_buffer, n2_buffer),
      recv("recv", n0_buffer, n1_buffer, n2_buffer);

  if constexpr (std::is_same_v<ArgComm, CommTag::oneCCL_Tag>) {
    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
      kvs = ccl::create_main_kvs();
      main_addr = kvs->get_address();
      MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
      MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
      kvs = ccl::create_kvs(main_addr);
    }

    Kokkos::DefaultExecutionSpace exec_space;
    sycl::queue q = exec_space.sycl_queue();
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    for (auto _ : state) {
      Kokkos::Timer timer;

      alltoall(send, recv, comm, stream);
      exec_space.fence();
      report_results(state, send, recv, MPI_COMM_WORLD, timer.seconds());
    }
  } else {
    for (auto _ : state) {
      Kokkos::Timer timer;

      alltoall(send, recv, MPI_COMM_WORLD);
      report_results(state, send, recv, MPI_COMM_WORLD, timer.seconds());
    }
  }
}

#define BENCHMARK_All2All(type, layout, tag, start, stop) \
  BENCHMARK(benchmark_all2all<type, Kokkos::layout, tag>) \
      ->UseManualTime()                              \
      ->Unit(benchmark::kMillisecond)                \
      ->ArgName("N")                                 \
      ->RangeMultiplier(2)                           \
      ->Range(start, stop)

//BENCHMARK_All2All(float, LayoutLeft, CommTag::MPI_Tag, 256, 4096);
//BENCHMARK_All2All(float, LayoutRight, CommTag::MPI_Tag, 256, 4096);
BENCHMARK_All2All(double, LayoutLeft, CommTag::MPI_Tag, 256, 4096);
BENCHMARK_All2All(double, LayoutRight, CommTag::MPI_Tag, 256, 4096);

//BENCHMARK_All2All(float, LayoutLeft, CommTag::oneCCL_Tag, 256, 4096);
//BENCHMARK_All2All(float, LayoutRight, CommTag::oneCCL_Tag, 256, 4096);
BENCHMARK_All2All(double, LayoutLeft, CommTag::oneCCL_Tag, 256, 4096);
BENCHMARK_All2All(double, LayoutRight, CommTag::oneCCL_Tag, 256, 4096);

#undef BENCHMARK_All2All

}  // anonymous namespace