/*
 * Simple self-test: CPU reference vs CUDA Olivetumhash kernel.
 * Runs one header/nonce and compares mixDigest with GPU.
 * If no CUDA device or ETHASHCUDA=OFF – GPU test is skipped.
 */

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include <libdevcore/FixedHash.h>
#include <libethcore/OlivetumhashAux.h>
#include <libethash-cuda/ethash_cuda_miner_kernel.h>
#include <ethash/keccak.hpp>

using namespace dev;
using namespace dev::eth;

namespace
{

std::string h256hex(const h256& h)
{
    return h.hex();
}

void be64put(uint8_t* p, uint64_t v)
{
    p[0] = static_cast<uint8_t>(v >> 56);
    p[1] = static_cast<uint8_t>(v >> 48);
    p[2] = static_cast<uint8_t>(v >> 40);
    p[3] = static_cast<uint8_t>(v >> 32);
    p[4] = static_cast<uint8_t>(v >> 24);
    p[5] = static_cast<uint8_t>(v >> 16);
    p[6] = static_cast<uint8_t>(v >> 8);
    p[7] = static_cast<uint8_t>(v);
}

}  // namespace

int main()
{
    const int epoch = 0;
    const h256 header("0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef");
    const uint64_t nonce = 0xdeadbeefull;

    auto cpuRes = OlivetumhashAux::eval(epoch, header, nonce);
    std::cout << "[CPU] mix=" << h256hex(cpuRes.mixHash)
              << " final=" << h256hex(cpuRes.value) << std::endl;

#if ETH_ETHASHCUDA
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        std::cout << "[CUDA] no devices, skipping" << std::endl;
        return 0;
    }

    cudaSetDevice(0);

    auto ds = OlivetumhashAux::datasetShared(epoch);
    const uint64_t dsSize = ds ? ds->size() : 0;
    if (dsSize == 0)
    {
        std::cerr << "[CUDA] dataset is empty" << std::endl;
        return 1;
    }

    uint8_t* d_dataset = nullptr;
    cudaMalloc(&d_dataset, dsSize);
    cudaMemcpy(d_dataset, ds->data(), dsSize, cudaMemcpyHostToDevice);

    set_olivetum_dataset(d_dataset, dsSize);
    set_olivetum_header(header.data());
    uint8_t target[32];
    std::memset(target, 0xff, sizeof(target));  // accept any finalDigest
    set_olivetum_target(target);

    Search_results* hostBuf = nullptr;
    cudaMallocHost(&hostBuf, sizeof(Search_results));
    hostBuf->count = 0;

    run_olivetumhash_search(1, 1, 0, hostBuf, nonce);
    cudaDeviceSynchronize();

    if (hostBuf->count == 0)
    {
        std::cerr << "[CUDA] no result" << std::endl;
        cudaFreeHost(hostBuf);
        cudaFree(d_dataset);
        return 2;
    }

    uint8_t mixBytes[32];
    std::memcpy(mixBytes, hostBuf->result[0].mix, 32);

    auto finalGpu = ethash::keccak256(mixBytes, sizeof(mixBytes));
    uint8_t finalBuf[32 + 32 + 8];
    std::memcpy(finalBuf, mixBytes, 32);
    std::memcpy(finalBuf + 32, header.data(), 32);
    uint8_t nonceBe[8];
    be64put(nonceBe, nonce);
    std::memcpy(finalBuf + 64, nonceBe, 8);
    auto finalFromGpu = ethash::keccak256(finalBuf, sizeof(finalBuf));

    h256 gpuMix(mixBytes, h256::ConstructFromPointer);
    h256 gpuFinal(finalFromGpu.bytes, h256::ConstructFromPointer);

    bool ok = (gpuMix == cpuRes.mixHash) && (gpuFinal == cpuRes.value);
    std::cout << "[CUDA] mix=" << h256hex(gpuMix) << " final=" << h256hex(gpuFinal) << " -> "
              << (ok ? "OK" : "MISMATCH") << std::endl;

    // Helper comparisons for endianness:
    uint8_t rev[32];
    for (int i = 0; i < 32; ++i)
        rev[i] = mixBytes[31 - i];
    h256 gpuMixRev(rev, h256::ConstructFromPointer);
    if (gpuMixRev == cpuRes.mixHash)
        std::cout << "[CUDA alt] rev-bytes MATCHES CPU" << std::endl;

    uint8_t wordsw[32];
    for (int i = 0; i < 8; ++i)
    {
        wordsw[i * 4 + 0] = mixBytes[i * 4 + 3];
        wordsw[i * 4 + 1] = mixBytes[i * 4 + 2];
        wordsw[i * 4 + 2] = mixBytes[i * 4 + 1];
        wordsw[i * 4 + 3] = mixBytes[i * 4 + 0];
    }
    h256 gpuMixWordSwap(wordsw, h256::ConstructFromPointer);
    if (gpuMixWordSwap == cpuRes.mixHash)
        std::cout << "[CUDA alt] per-word byte-swap MATCHES CPU" << std::endl;

    cudaFreeHost(hostBuf);
    cudaFree(d_dataset);

    if (!ok)
        return 3;
#else
    std::cout << "[CUDA] ETHASHCUDA=OFF - skipping GPU test" << std::endl;
#endif

    return 0;
}
