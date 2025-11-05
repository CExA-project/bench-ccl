# bench-ccl

# Compile and Run
## Installation
First of all, you need to clone this repo.
```bash
git clone --recursive https://github.com/CExA-project/bench-ccl.git
```

## GPU (SYCL backend on Aurora)

```bash
export CCL_PROCESS_LAUNCHER=pmix  
export CCL_ATL_TRANSPORT=mpi
export CCL_ALLREDUCE=topo
export CCL_ALLREDUCE_SCALEOUT=rabenseifner 

export CCL_KVS_MODE=mpi
export CCL_CONFIGURATION_PATH=""
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600

module load cmake frameworks
cmake -B build \
      -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_SYCL=ON \
      -DKokkos_ARCH_INTEL_PVC=ON

cmake --build build -j 8
cd build
mpiexec -n 4 --ppn 4 --depth=1 --cpu-bind depth ./oneccl-benchmark
```

## Results

```
NUM_OF_NODES= 1 TOTAL_NUM_RANKS= 4 RANKS_PER_NODE= 4 THREADS_PER_RANK= 1
-------------------------------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                                       Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------------------------------------------------
benchmark_all2all<float, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:256/manual_time            0.215 ms        0.218 ms         3252 MB (In)=0.262144 MB (Out)=0.262144
benchmark_all2all<float, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:512/manual_time            0.219 ms        0.221 ms         3204 MB (In)=1.04858 MB (Out)=1.04858
benchmark_all2all<float, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:1024/manual_time           0.253 ms        0.256 ms         2764 MB (In)=4.1943 MB (Out)=4.1943
benchmark_all2all<float, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:2048/manual_time           0.407 ms        0.410 ms         1717 MB (In)=16.7772 MB (Out)=16.7772
benchmark_all2all<float, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:4096/manual_time            1.61 ms         1.62 ms          435 MB (In)=67.1089 MB (Out)=67.1089
benchmark_all2all<float, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:256/manual_time           0.216 ms        0.219 ms         3240 MB (In)=0.262144 MB (Out)=0.262144
benchmark_all2all<float, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:512/manual_time           0.218 ms        0.221 ms         3210 MB (In)=1.04858 MB (Out)=1.04858
benchmark_all2all<float, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:1024/manual_time          0.253 ms        0.256 ms         2762 MB (In)=4.1943 MB (Out)=4.1943
benchmark_all2all<float, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:2048/manual_time          0.407 ms        0.409 ms         1744 MB (In)=16.7772 MB (Out)=16.7772
benchmark_all2all<float, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:4096/manual_time           1.61 ms         1.61 ms          429 MB (In)=67.1089 MB (Out)=67.1089
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:256/manual_time           0.217 ms        0.220 ms         3226 MB (In)=0.524288 MB (Out)=0.524288
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:512/manual_time           0.218 ms        0.221 ms         3207 MB (In)=2.09715 MB (Out)=2.09715
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:1024/manual_time          0.294 ms        0.296 ms         2382 MB (In)=8.38861 MB (Out)=8.38861
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:2048/manual_time          0.870 ms        0.872 ms          799 MB (In)=33.5544 MB (Out)=33.5544
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:4096/manual_time           3.03 ms         3.03 ms          230 MB (In)=134.218 MB (Out)=134.218
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:256/manual_time          0.217 ms        0.220 ms         3232 MB (In)=0.524288 MB (Out)=0.524288
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:512/manual_time          0.218 ms        0.221 ms         3207 MB (In)=2.09715 MB (Out)=2.09715
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:1024/manual_time         0.293 ms        0.296 ms         2380 MB (In)=8.38861 MB (Out)=8.38861
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:2048/manual_time         0.865 ms        0.867 ms          802 MB (In)=33.5544 MB (Out)=33.5544
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:4096/manual_time          3.02 ms         3.02 ms          231 MB (In)=134.218 MB (Out)=134.218
2025:11:05-08:13:07:(78273) |CCL_WARN| value of CCL_KVS_MODE changed to be mpi (default:pmi)
2025:11:05-08:13:07:(78273) |CCL_WARN| value of CCL_KVS_CONNECTION_TIMEOUT changed to be 600 (default:120)
2025:11:05-08:13:07:(78273) |CCL_WARN| value of CCL_ALLREDUCE changed to be topo (default:)
2025:11:05-08:13:07:(78273) |CCL_WARN| value of CCL_ALLREDUCE_SCALEOUT changed to be rabenseifner (default:)
2025:11:05-08:13:07:(78273) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
2025:11:05-08:13:07:(78273) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
2025:11:05-08:13:07:(78273) |CCL_WARN| topology recognition shows PCIe connection between devices. If this is not correct, you can disable topology recognition, with CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0. This will assume XeLinks across devices
benchmark_all2all<float, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:256/manual_time         0.525 ms        0.529 ms         1340 MB (In)=0.262144 MB (Out)=0.262144
benchmark_all2all<float, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:512/manual_time         0.800 ms        0.804 ms          866 MB (In)=1.04858 MB (Out)=1.04858
benchmark_all2all<float, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:1024/manual_time         1.45 ms         1.46 ms          466 MB (In)=4.1943 MB (Out)=4.1943
benchmark_all2all<float, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:2048/manual_time         5.35 ms         5.35 ms          113 MB (In)=16.7772 MB (Out)=16.7772
benchmark_all2all<float, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:4096/manual_time         58.2 ms         58.1 ms            9 MB (In)=67.1089 MB (Out)=67.1089
benchmark_all2all<float, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:256/manual_time        0.523 ms        0.527 ms         1344 MB (In)=0.262144 MB (Out)=0.262144
benchmark_all2all<float, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:512/manual_time        0.816 ms        0.820 ms          861 MB (In)=1.04858 MB (Out)=1.04858
benchmark_all2all<float, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:1024/manual_time        1.46 ms         1.47 ms          474 MB (In)=4.1943 MB (Out)=4.1943
benchmark_all2all<float, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:2048/manual_time        5.40 ms         5.40 ms          123 MB (In)=16.7772 MB (Out)=16.7772
benchmark_all2all<float, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:4096/manual_time        58.0 ms         57.9 ms           12 MB (In)=67.1089 MB (Out)=67.1089
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:256/manual_time        0.614 ms        0.618 ms         1135 MB (In)=0.524288 MB (Out)=0.524288
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:512/manual_time        0.963 ms        0.967 ms          715 MB (In)=2.09715 MB (Out)=2.09715
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:1024/manual_time        2.67 ms         2.67 ms          250 MB (In)=8.38861 MB (Out)=8.38861
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:2048/manual_time        16.4 ms         16.4 ms           42 MB (In)=33.5544 MB (Out)=33.5544
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:4096/manual_time         111 ms          111 ms            5 MB (In)=134.218 MB (Out)=134.218
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:256/manual_time       0.614 ms        0.618 ms         1138 MB (In)=0.524288 MB (Out)=0.524288
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:512/manual_time       0.947 ms        0.950 ms          735 MB (In)=2.09715 MB (Out)=2.09715
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:1024/manual_time       2.63 ms         2.63 ms          266 MB (In)=8.38861 MB (Out)=8.38861
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:2048/manual_time       15.1 ms         15.1 ms           48 MB (In)=33.5544 MB (Out)=33.5544
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:4096/manual_time        112 ms          111 ms            7 MB (In)=134.218 MB (Out)=134.218
```
