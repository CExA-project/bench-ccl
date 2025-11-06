#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include "mpi_wrapper.hpp"

#if defined(KOKKOS_ENABLE_SYCL)
#include "oneccl_wrapper.hpp"
#else
#include "nccl_wrapper.hpp"
#endif

namespace {

struct CommTag {
  struct MPI_Tag {};
  struct CCL_Tag {};
};

template <typename T, typename LayoutType, typename ArgComm>
void benchmark_all2all(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_all2all needs at least 2 ranks");
  }

  const int n = state.range(0);
  using View3DType =
      Kokkos::View<T ***, LayoutType, Kokkos::DefaultExecutionSpace>;

  int n0_buffer = 0, n1_buffer = 0, n2_buffer = 0;
  if constexpr (std::is_same<LayoutType, Kokkos::LayoutLeft>::value) {
    n0_buffer = (n - 1) / size + 1;
    n1_buffer = n;
    n2_buffer = size;
  } else {
    n0_buffer = size;
    n1_buffer = n;
    n2_buffer = (n - 1) / size + 1;
  }
  View3DType send("send", n0_buffer, n1_buffer, n2_buffer),
      recv("recv", n0_buffer, n1_buffer, n2_buffer);

  if constexpr (std::is_same_v<ArgComm, CommTag::CCL_Tag>) {
#if defined(KOKKOS_ENABLE_SYCL)
    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
      kvs       = ccl::create_main_kvs();
      main_addr = kvs->get_address();
      MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
                MPI_COMM_WORLD);
    } else {
      MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
                MPI_COMM_WORLD);
      kvs = ccl::create_kvs(main_addr);
    }

    Kokkos::DefaultExecutionSpace exec_space;
    sycl::queue q = exec_space.sycl_queue();
    auto dev      = ccl::create_device(q.get_device());
    auto ctx      = ccl::create_context(q.get_context());
    auto comm     = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    for (auto _ : state) {
      Kokkos::Timer timer;

      alltoall(send, recv, comm, stream);
      exec_space.fence();
      report_results(state, send, recv, MPI_COMM_WORLD, timer.seconds());
    }
#else
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    ::MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ::MPI_Barrier(MPI_COMM_WORLD);

    ncclComm_t comm;
    ncclCommInitRank(&comm, size, id, rank);

    Kokkos::DefaultExecutionSpace exec_space;

#if defined(KOKKOS_ENABLE_CUDA)
    auto stream = exec_space.cuda_stream();
#elif defined(KOKKOS_ENABLE_HIP)
    auto stream = exec_space.hip_stream();
#else
    static_assert(false,
                  "You need to enable CUDA (HIP) backend to use NCCL (RCCL).");
#endif

    for (auto _ : state) {
      Kokkos::Timer timer;

      alltoall(send, recv, comm, stream, size);
      exec_space.fence();
      report_results(state, send, recv, MPI_COMM_WORLD, timer.seconds());
    }
    ncclCommDestroy(comm);
#endif
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
      ->UseManualTime()                                   \
      ->Unit(benchmark::kMillisecond)                     \
      ->ArgName("N")                                      \
      ->RangeMultiplier(2)                                \
      ->Range(start, stop)

// BENCHMARK_All2All(float, LayoutLeft, CommTag::MPI_Tag, 256, 8192);
// BENCHMARK_All2All(float, LayoutRight, CommTag::MPI_Tag, 256, 8192);
BENCHMARK_All2All(double, LayoutLeft, CommTag::MPI_Tag, 256, 8192);
BENCHMARK_All2All(double, LayoutRight, CommTag::MPI_Tag, 256, 8192);

// BENCHMARK_All2All(float, LayoutLeft, CommTag::oneCCL_Tag, 256, 8192);
// BENCHMARK_All2All(float, LayoutRight, CommTag::oneCCL_Tag, 256, 8192);
BENCHMARK_All2All(double, LayoutLeft, CommTag::CCL_Tag, 256, 8192);
BENCHMARK_All2All(double, LayoutRight, CommTag::CCL_Tag, 256, 8192);

#undef BENCHMARK_All2All

}  // anonymous namespace
