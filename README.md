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

cmake -B build \
      -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_SYCL=ON \
      -DKokkos_ARCH_INTEL_PVC=ON

cmake --build build -j 8
mpiexec -n 24 --ppn ${RANKS_PER_NODE} --depth=1 --cpu-bind depth build/oneccl-benchmark
```

## Results

```
NUM_OF_NODES= 1 TOTAL_NUM_RANKS= 4 RANKS_PER_NODE= 4 THREADS_PER_RANK= 1
-------------------------------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                                       Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------------------------------------------------
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:256/manual_time            7.07 ms         7.16 ms          103 In (MB)=0.540672 Out (MB)=0.540672
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:512/manual_time            35.8 ms         35.3 ms           19 In (MB)=2.16269 Out (MB)=2.16269
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:1024/manual_time           35.7 ms         35.1 ms           19 In (MB)=8.45414 Out (MB)=8.45414
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:2048/manual_time           49.5 ms         48.6 ms           11 In (MB)=33.8166 Out (MB)=33.8166
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:4096/manual_time           70.4 ms         67.1 ms            8 In (MB)=134.48 Out (MB)=134.48
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:256/manual_time           7.14 ms         7.10 ms           97 In (MB)=0.540672 Out (MB)=0.540672
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:512/manual_time           35.6 ms         35.1 ms           20 In (MB)=2.16269 Out (MB)=2.16269
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:1024/manual_time          35.7 ms         35.2 ms           20 In (MB)=8.45414 Out (MB)=8.45414
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:2048/manual_time          48.3 ms         47.5 ms           13 In (MB)=33.8166 Out (MB)=33.8166
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:4096/manual_time          67.1 ms         66.1 ms            9 In (MB)=134.48 Out (MB)=134.48
2025:11:05-12:53:24:(19490) |CCL_WARN| value of CCL_KVS_MODE changed to be mpi (default:pmi)
2025:11:05-12:53:24:(19490) |CCL_WARN| value of CCL_KVS_CONNECTION_TIMEOUT changed to be 600 (default:120)
2025:11:05-12:53:24:(19490) |CCL_WARN| value of CCL_ALLREDUCE changed to be topo (default:)
2025:11:05-12:53:24:(19490) |CCL_WARN| value of CCL_ALLREDUCE_SCALEOUT changed to be rabenseifner (default:)
2025:11:05-12:53:24:(19490) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
2025:11:05-12:53:24:(19490) |CCL_WARN| topology recognition shows PCIe connection between devices. If this is not correct, you can disable topology recognition, with CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0. This will assume XeLinks across devices
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:256/manual_time         2.15 ms         2.17 ms          305 In (MB)=0.540672 Out (MB)=0.540672
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:512/manual_time         2.89 ms         2.91 ms          243 In (MB)=2.16269 Out (MB)=2.16269
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:1024/manual_time        6.63 ms         6.64 ms          100 In (MB)=8.45414 Out (MB)=8.45414
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:2048/manual_time        31.8 ms         31.8 ms           21 In (MB)=33.8166 Out (MB)=33.8166
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::oneCCL_Tag>/N:4096/manual_time         110 ms          110 ms            6 In (MB)=134.48 Out (MB)=134.48
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:256/manual_time        2.16 ms         2.17 ms          324 In (MB)=0.540672 Out (MB)=0.540672
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:512/manual_time        2.88 ms         2.89 ms          242 In (MB)=2.16269 Out (MB)=2.16269
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:1024/manual_time       6.70 ms         6.71 ms           99 In (MB)=8.45414 Out (MB)=8.45414
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:2048/manual_time       30.6 ms         30.6 ms           23 In (MB)=33.8166 Out (MB)=33.8166
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::oneCCL_Tag>/N:4096/manual_time        110 ms          109 ms            6 In (MB)=134.48 Out (MB)=134.48
```
