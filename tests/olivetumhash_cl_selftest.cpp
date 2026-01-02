/*
 * OpenCL self-test: CPU vs Olivetumhash OpenCL kernel.
 * Epoch 0, fixed header, nonce = 0xdeadbeef. Verifies mix/final.
 */

#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS true
#define CL_HPP_ENABLE_EXCEPTIONS true
#define CL_HPP_CL_1_2_DEFAULT_BUILD true
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <libdevcore/FixedHash.h>
#include <libethcore/OlivetumhashAux.h>
#include <ethash/keccak.hpp>

using namespace dev;
using namespace dev::eth;

namespace
{
const char* kKernelPath = "libethash-cl/kernels/cl/olivetumhash.cl";
const size_t WORKSIZE = 64;
const size_t MAX_OUTPUTS = 4;

struct Result
{
    uint32_t gid;
    uint32_t mix[8];
    uint32_t pad[7];
};

struct SearchResults
{
    Result rslt[MAX_OUTPUTS];
    uint32_t count;
    uint32_t hashCount;
    uint32_t abort;
};

std::string h256hex(const h256& h)
{
    return h.hex();
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

#if ETH_ETHASHCL
    try
    {
        // Pick the first available GPU.
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
        {
            std::cout << "[OpenCL] no platforms, skipping" << std::endl;
            return 0;
        }
        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty())
        {
            std::cout << "[OpenCL] no GPU, skipping" << std::endl;
            return 0;
        }
        cl::Device device = devices[0];
        cl::Context ctx(device);
        cl::CommandQueue q(ctx, device);

        // Wczytaj kernel z pliku.
        std::ifstream f(kKernelPath);
        if (!f.good())
        {
            std::cerr << "[OpenCL] cannot open " << kKernelPath << std::endl;
            return 1;
        }
        std::string code((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        // Prepend build-time defines.
        std::string defs;
        defs += "#define WORKSIZE " + std::to_string(WORKSIZE) + "\n";
        defs += "#define MAX_OUTPUTS " + std::to_string(MAX_OUTPUTS) + "\n";
        defs += "#define FAST_EXIT 1\n";
        code = defs + code;

        cl::Program prog(ctx, code);
        try
        {
            prog.build({device});
        }
        catch (cl::Error const&)
        {
            std::cerr << "[OpenCL] build log:\n"
                      << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            throw;
        }

        cl::Kernel kernel(prog, "search");

        auto ds = OlivetumhashAux::datasetShared(epoch);
        if (!ds || ds->empty())
        {
            std::cerr << "[OpenCL] missing dataset" << std::endl;
            return 2;
        }
        unsigned chunkCount = static_cast<unsigned>(ds->size() / 64);

        SearchResults zero{};
        cl::Buffer bufOut(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(SearchResults), &zero);
        cl::Buffer bufHeader(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)32,
            const_cast<uint8_t*>(header.data()));
        cl::Buffer bufDataset(
            ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)ds->size(), ds->data());
        cl::Buffer bufTarget(ctx, CL_MEM_READ_ONLY, (size_t)32);
        uint8_t target[32];
        std::memset(target, 0xff, sizeof(target));
        q.enqueueWriteBuffer(bufTarget, CL_TRUE, 0, 32, target);

        kernel.setArg(0, bufOut);
        kernel.setArg(1, bufHeader);
        kernel.setArg(2, bufDataset);
        kernel.setArg(3, chunkCount);
        kernel.setArg(4, static_cast<cl_ulong>(nonce));
        kernel.setArg(5, bufTarget);

        q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(WORKSIZE), cl::NDRange(WORKSIZE));
        q.finish();

        SearchResults out{};
        q.enqueueReadBuffer(bufOut, CL_TRUE, 0, sizeof(out), &out);
        if (out.count == 0)
        {
            std::cerr << "[OpenCL] no results" << std::endl;
            return 3;
        }

        // mixDigest is raw 32 bytes (uint32_t mix[8]).
        uint8_t mixBytes[32];
        std::memcpy(mixBytes, out.rslt[0].mix, 32);
        auto finalGpu = ethash::keccak256(mixBytes, sizeof(mixBytes));
        uint8_t finalBuf[32 + 32 + 8];
        std::memcpy(finalBuf, mixBytes, 32);
        std::memcpy(finalBuf + 32, header.data(), 32);
        std::memcpy(finalBuf + 64, &nonce, 8);
        auto finalFromGpu = ethash::keccak256(finalBuf, sizeof(finalBuf));

        h256 gpuMix(mixBytes, h256::ConstructFromPointer);
        h256 gpuFinal(finalFromGpu.bytes, h256::ConstructFromPointer);

        bool ok = (gpuMix == cpuRes.mixHash) && (gpuFinal == cpuRes.value);
        std::cout << "[OpenCL] mix=" << h256hex(gpuMix) << " final=" << h256hex(gpuFinal) << " -> "
                  << (ok ? "OK" : "MISMATCH") << std::endl;

        return ok ? 0 : 4;
    }
    catch (const cl::Error& e)
    {
        std::cerr << "[OpenCL] error: " << e.what() << " code=" << e.err() << std::endl;
        return 0;
    }
#else
    std::cout << "[OpenCL] ETHASHCL=OFF - skipping" << std::endl;
    return 0;
#endif
}
