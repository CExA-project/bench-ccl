#ifndef ONECCL_WRAPPER_HPP_
#define ONECCL_WRAPPER_HPP_

#include <mpi.h>
#include <cstdint>
#include <ccl.hpp>
#include <Kokkos_Core.hpp>

template <typename ValueType>
struct oneCCLDataType {};

template <>
struct oneCCLDataType<int> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::int32; }
};

template <>
struct oneCCLDataType<std::uint32_t> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::uint32; }
};

template <>
struct oneCCLDataType<std::int64_t> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::int64; }
};

template <>
struct oneCCLDataType<std::uint64_t> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::uint64; }
};

template <>
struct oneCCLDataType<float> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::float32; }
};

template <>
struct oneCCLDataType<double> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::float64; }
};

template <typename ViewType>
void alltoall(const ViewType& send, const ViewType& recv,
              const ccl::communicator& comm, const ccl::stream& stream) {
  using value_type = typename ViewType::non_const_value_type;
  using LayoutType = typename ViewType::array_layout;
  int size_send    = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                         ? send.extent_int(ViewType::rank() - 1)
                         : send.extent_int(0);
  int size_recv    = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                         ? recv.extent_int(ViewType::rank() - 1)
                         : recv.extent_int(0);

  int count = static_cast<int>(send.size()) / size_send;
  auto type = oneCCLDataType<value_type>::type();
  ccl::alltoall(send.data(), recv.data(), count, type, comm, stream).wait();
}

#endif
