# Olivetumhash GPU Miner (ethminer fork)

Fork of ethminer focused on Olivetumhash. CUDA and OpenCL binaries work on one build across old and new NVIDIA/AMD GPUs. Ethash compatibility is kept but not the focus of this release.

## Features
- Olivetumhash mining (CUDA + OpenCL), single binary covering legacy and new GPU architectures
- Ethash compatibility preserved
- Benchmark mode for Olivetumhash (`--olivetum-benchmark <epoch>`)
- On-GPU dataset generation and upload

## Build
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Recommended for Linux + CUDA 12.x (avoid GCC 13 toolchain issues):
```
cmake -S . -B build_gcc12 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build_gcc12 -j
```
If you see _Float32/_Float64 errors with CUDA 12.0 + GCC 13, use GCC 12 or upgrade CUDA.

To add extra CUDA architectures:
```
cmake -S . -B build -DCOMPUTE=86,89
cmake --build build -j
```

## Usage (Olivetumhash)
- NVIDIA/CUDA
```
./ethminer -U --olivetum -P http://<RPC_HOST>:<PORT> --report-hashrate --display-interval 10
```
- AMD/OpenCL
```
./ethminer -G --olivetum -P http://<RPC_HOST>:<PORT> --report-hashrate --display-interval 10
```

## Benchmark
Run Olivetumhash benchmark without a pool (example: epoch 0):
```
./ethminer --olivetum-benchmark 0 -U
```

## Notes
- Requires appropriate GPU drivers with CUDA/OpenCL runtime.
- CUDA builds include sm_89 (RTX 40xx) and sm_90 when supported by the toolkit.
- Default CUDA grid size is 4096 (`--cu-grid-size`).
- `--cl-global-work` sets the OpenCL global work size multiplier.
- Based on the open-source ethminer project; original authorship retained in source history.
