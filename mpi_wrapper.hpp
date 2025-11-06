#ifndef MPI_WRAPPER_HPP_
#define MPI_WRAPPER_HPP_

#include <mpi.h>
#include <Kokkos_Core.hpp>

template <typename ValueType>
struct MPIDataType {};

template <>
struct MPIDataType<int> {
  static inline MPI_Datatype type() noexcept { return MPI_INT32_T; }
};

template <>
struct MPIDataType<std::size_t> {
  static inline MPI_Datatype type() noexcept { return MPI_UINT64_T; }
};

template <>
struct MPIDataType<float> {
  static inline MPI_Datatype type() noexcept { return MPI_FLOAT; }
};

template <>
struct MPIDataType<double> {
  static inline MPI_Datatype type() noexcept { return MPI_DOUBLE; }
};

template <>
struct MPIDataType<Kokkos::complex<float>> {
  static inline MPI_Datatype type() noexcept { return MPI_CXX_FLOAT_COMPLEX; }
};

template <>
struct MPIDataType<Kokkos::complex<double>> {
  static inline MPI_Datatype type() noexcept { return MPI_CXX_DOUBLE_COMPLEX; }
};

template <typename ViewType>
void alltoall(const ViewType& send, const ViewType& recv,
              const MPI_Comm& comm) {
  using value_type = typename ViewType::non_const_value_type;
  using LayoutType = typename ViewType::array_layout;
  int size_send    = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                         ? send.extent_int(ViewType::rank() - 1)
                         : send.extent_int(0);
  int size_recv    = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                         ? recv.extent_int(ViewType::rank() - 1)
                         : recv.extent_int(0);

  int count = static_cast<int>(send.size()) / size_send;
  auto type = MPIDataType<value_type>::type();
  ::MPI_Alltoall(send.data(), count, type, recv.data(), count, type, comm);
}

template <typename InViewType, typename OutViewType>
void report_results(benchmark::State& state, const InViewType& in,
                    const OutViewType& out, const MPI_Comm& comm, double time) {
  // data processed in megabytes
  const double in_data_processed =
      static_cast<double>(in.size() * sizeof(typename InViewType::value_type)) /
      1.0e6;
  const double out_data_processed =
      static_cast<double>(out.size() *
                          sizeof(typename OutViewType::value_type)) /
      1.0e6;

  double max_time = 0.0;
  MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  state.SetIterationTime(max_time);
  state.counters["In (MB)"]  = benchmark::Counter(in_data_processed);
  state.counters["Out (MB)"] = benchmark::Counter(out_data_processed);
}

#endif
