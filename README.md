# Olivetumhash GPU Miner (ethminer fork)

Fork of ethminer focused on Olivetumhash. CUDA and OpenCL binaries work on one build across old and new NVIDIA/AMD GPUs. Ethash compatibility is kept but not the focus of this release.

## Features
- Olivetumhash mining (CUDA + OpenCL), single binary covering legacy and new GPU architectures
- Ethash compatibility preserved
- Benchmark mode for Olivetumhash (`--olivetum-benchmark <epoch>`)
- On-GPU dataset generation and upload

## Build
```
cmake -B build -DCMAKE_BUILD_TYPE=Release
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
- Based on the open-source ethminer project; original authorship retained in source history.
