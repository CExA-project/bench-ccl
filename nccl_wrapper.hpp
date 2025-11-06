#ifndef NCCL_WRAPPER_HPP_
#define NCCL_WRAPPER_HPP_

#include <mpi.h>
#include <cstdint>
#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <nccl.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <rccl/rccl.h>
#else
static_assert(false,
              "You need to enable CUDA (HIP) backend to use NCCL (RCCL).");
#endif

template <typename ValueType>
struct NCCLDataType {};

template <>
struct NCCLDataType<int> {
  static inline ncclDataType_t type() noexcept { return ncclInt; }
};

template <>
struct NCCLDataType<std::uint32_t> {
  static inline ncclDataType_t type() noexcept { return ncclUint32; }
};

template <>
struct NCCLDataType<std::int64_t> {
  static inline ncclDataType_t type() noexcept { return ncclInt64; }
};

template <>
struct NCCLDataType<std::uint64_t> {
  static inline ncclDataType_t type() noexcept { return ncclUint64; }
};

template <>
struct NCCLDataType<float> {
  static inline ncclDataType_t type() noexcept { return ncclFloat; }
};

template <>
struct NCCLDataType<double> {
  static inline ncclDataType_t type() noexcept { return ncclDouble; }
};

template <typename ViewType, typename StreamType>
void alltoall(const ViewType& send, const ViewType& recv,
              const ncclComm_t& comm, const StreamType& stream,
              [[maybe_unused]] int size) {
  using value_type = typename ViewType::non_const_value_type;
  using LayoutType = typename ViewType::array_layout;
  int size_send    = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                         ? send.extent_int(ViewType::rank() - 1)
                         : send.extent_int(0);
  int size_recv    = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                         ? recv.extent_int(ViewType::rank() - 1)
                         : recv.extent_int(0);

  int count       = static_cast<int>(send.size()) / size_send;
  auto type       = NCCLDataType<value_type>::type();
  auto* send_data = send.data();
  auto* recv_data = recv.data();

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  ncclAlltoAll(send_data, recv_data, count, type, comm, stream);
#else
  ncclGroupStart();
  for (int r = 0; r < size; ++r) {
    ncclSend(send_data + r * count, count, type, r, comm, stream);
    ncclRecv(recv_data + r * count, count, type, r, comm, stream);
  }
  ncclGroupEnd();
#endif
}

#endif
