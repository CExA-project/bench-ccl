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
mpiexec -n 24 --ppn ${RANKS_PER_NODE} --depth=1 --cpu-bind depth build/benchmark-ccls
```

## Results

```
NUM_OF_NODES= 2 TOTAL_NUM_RANKS= 8 RANKS_PER_NODE= 12 THREADS_PER_RANK= 1
----------------------------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                                    Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------------------------------------
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:256/manual_time         7.35 ms         7.43 ms           92 In (MB)=0.540672 Out (MB)=0.540672
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:512/manual_time         35.7 ms         35.0 ms           19 In (MB)=2.16269 Out (MB)=2.16269
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:1024/manual_time        35.9 ms         35.1 ms           19 In (MB)=8.45414 Out (MB)=8.45414
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:2048/manual_time        50.1 ms         48.3 ms           11 In (MB)=33.8166 Out (MB)=33.8166
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:4096/manual_time        71.8 ms         68.8 ms            7 In (MB)=134.48 Out (MB)=134.48
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:8192/manual_time         192 ms          189 ms            3 In (MB)=537.919 Out (MB)=537.919
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:256/manual_time        6.45 ms         6.47 ms          101 In (MB)=0.540672 Out (MB)=0.540672
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:512/manual_time        35.6 ms         34.8 ms           19 In (MB)=2.16269 Out (MB)=2.16269
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:1024/manual_time       36.1 ms         35.5 ms           19 In (MB)=8.45414 Out (MB)=8.45414
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:2048/manual_time       47.5 ms         46.5 ms           12 In (MB)=33.8166 Out (MB)=33.8166
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:4096/manual_time       64.2 ms         59.4 ms           10 In (MB)=134.48 Out (MB)=134.48
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:8192/manual_time        189 ms          183 ms            4 In (MB)=537.919 Out (MB)=537.919
2025:11:06-02:43:49:(72412) |CCL_WARN| value of CCL_KVS_MODE changed to be mpi (default:pmi)
2025:11:06-02:43:49:(72412) |CCL_WARN| value of CCL_KVS_CONNECTION_TIMEOUT changed to be 600 (default:120)
2025:11:06-02:43:49:(72412) |CCL_WARN| value of CCL_ALLREDUCE changed to be topo (default:)
2025:11:06-02:43:49:(72412) |CCL_WARN| value of CCL_ALLREDUCE_SCALEOUT changed to be rabenseifner (default:)
2025:11:06-02:43:49:(72412) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
2025:11:06-02:43:50:(72412) |CCL_WARN| topology recognition shows PCIe connection between devices. If this is not correct, you can disable topology recognition, with CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0. This will assume XeLinks across devices
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:256/manual_time         2.15 ms         2.16 ms          303 In (MB)=0.540672 Out (MB)=0.540672
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:512/manual_time         2.87 ms         2.88 ms          241 In (MB)=2.16269 Out (MB)=2.16269
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:1024/manual_time        6.68 ms         6.69 ms          101 In (MB)=8.45414 Out (MB)=8.45414
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:2048/manual_time        30.7 ms         30.6 ms           19 In (MB)=33.8166 Out (MB)=33.8166
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:4096/manual_time         112 ms          112 ms            6 In (MB)=134.48 Out (MB)=134.48
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:8192/manual_time         646 ms          632 ms            1 In (MB)=537.919 Out (MB)=537.919
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:256/manual_time        2.14 ms         2.16 ms          327 In (MB)=0.540672 Out (MB)=0.540672
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:512/manual_time        2.88 ms         2.89 ms          246 In (MB)=2.16269 Out (MB)=2.16269
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:1024/manual_time       6.70 ms         6.71 ms           99 In (MB)=8.45414 Out (MB)=8.45414
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:2048/manual_time       30.7 ms         30.7 ms           23 In (MB)=33.8166 Out (MB)=33.8166
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:4096/manual_time        111 ms          111 ms            6 In (MB)=134.48 Out (MB)=134.48
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:8192/manual_time        465 ms          464 ms            2 In (MB)=537.919 Out (MB)=537.919
```

## GPU (HIP backend on Adastra)

```bash
module purge
module load cpe/24.07
module load craype-x86-trento craype-accel-amd-gfx90a
module load PrgEnv-gnu-amd
module load amd-mixed/6.3.3
module load cmake/3.27.9

cmake -B build \
      -DCMAKE_CXX_COMPILER=hipcc \
      -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_HIP=ON \
      -DKokkos_ARCH_AMD_GFX90A=ON \
      -DCMAKE_EXE_LINKER_FLAGS="${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"

export MPICH_GPU_SUPPORT_ENABLED=1

srun --ntasks-per-node=8 --cpu-bind=none \
    -- build/benchmark-ccls
```

## Results

```
----------------------------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                                    Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------------------------------------
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:256/manual_time        0.168 ms        0.183 ms         3889 In (MB)=0.524288 Out (MB)=0.524288
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:512/manual_time        0.174 ms        0.188 ms         3961 In (MB)=2.09715 Out (MB)=2.09715
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:1024/manual_time       0.750 ms        0.756 ms          773 In (MB)=8.38861 Out (MB)=8.38861
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:2048/manual_time        2.80 ms         2.81 ms          239 In (MB)=33.5544 Out (MB)=33.5544
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:4096/manual_time        10.8 ms         10.8 ms           63 In (MB)=134.218 Out (MB)=134.218
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::MPI_Tag>/N:8192/manual_time        43.3 ms         43.2 ms           16 In (MB)=536.871 Out (MB)=536.871
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:256/manual_time       0.161 ms        0.169 ms         4172 In (MB)=0.524288 Out (MB)=0.524288
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:512/manual_time       0.175 ms        0.183 ms         3865 In (MB)=2.09715 Out (MB)=2.09715
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:1024/manual_time      0.756 ms        0.761 ms          752 In (MB)=8.38861 Out (MB)=8.38861
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:2048/manual_time       2.81 ms         2.81 ms          236 In (MB)=33.5544 Out (MB)=33.5544
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:4096/manual_time       10.8 ms         10.8 ms           63 In (MB)=134.218 Out (MB)=134.218
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::MPI_Tag>/N:8192/manual_time       43.4 ms         43.3 ms           16 In (MB)=536.871 Out (MB)=536.871
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:256/manual_time        0.688 ms        0.576 ms          730 In (MB)=0.524288 Out (MB)=0.524288
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:512/manual_time         1.72 ms         1.48 ms          330 In (MB)=2.09715 Out (MB)=2.09715
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:1024/manual_time        6.67 ms         5.56 ms           83 In (MB)=8.38861 Out (MB)=8.38861
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:2048/manual_time        25.2 ms         22.0 ms           28 In (MB)=33.5544 Out (MB)=33.5544
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:4096/manual_time         100 ms         86.6 ms            7 In (MB)=134.218 Out (MB)=134.218
benchmark_all2all<double, Kokkos::LayoutLeft, CommTag::CCL_Tag>/N:8192/manual_time         377 ms          333 ms            2 In (MB)=536.871 Out (MB)=536.871
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:256/manual_time       0.617 ms        0.514 ms          838 In (MB)=0.524288 Out (MB)=0.524288
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:512/manual_time        1.85 ms         1.60 ms          353 In (MB)=2.09715 Out (MB)=2.09715
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:1024/manual_time       6.24 ms         5.15 ms           87 In (MB)=8.38861 Out (MB)=8.38861
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:2048/manual_time       27.7 ms         22.9 ms           19 In (MB)=33.5544 Out (MB)=33.5544
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:4096/manual_time       98.3 ms         81.6 ms            7 In (MB)=134.218 Out (MB)=134.218
benchmark_all2all<double, Kokkos::LayoutRight, CommTag::CCL_Tag>/N:8192/manual_time        383 ms          314 ms            2 In (MB)=536.871 Out (MB)=536.871
```
