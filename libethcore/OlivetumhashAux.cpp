/*
    CPU reference implementation of Olivetumhash.
*/

#include "OlivetumhashAux.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ethash/ethash.hpp>
#include <ethash/keccak.hpp>

using namespace dev;
using namespace eth;

namespace
{

constexpr uint64_t c_datasetInitBytes = 16ull * 1024ull * 1024ull;
constexpr uint64_t c_datasetGrowthBytes = 2ull * 1024ull * 1024ull;
constexpr uint64_t c_mixRounds = 64;

inline uint64_t align64(uint64_t v)
{
    return (v + 63ull) / 64ull * 64ull;
}

inline uint64_t rotl64(uint64_t x, unsigned int r)
{
    return (x << r) | (x >> (64 - r));
}

inline uint64_t le64(const uint8_t* p)
{
    return static_cast<uint64_t>(p[0]) | (static_cast<uint64_t>(p[1]) << 8) |
           (static_cast<uint64_t>(p[2]) << 16) | (static_cast<uint64_t>(p[3]) << 24) |
           (static_cast<uint64_t>(p[4]) << 32) | (static_cast<uint64_t>(p[5]) << 40) |
           (static_cast<uint64_t>(p[6]) << 48) | (static_cast<uint64_t>(p[7]) << 56);
}

inline void le64put(uint8_t* p, uint64_t v)
{
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
    p[2] = static_cast<uint8_t>(v >> 16);
    p[3] = static_cast<uint8_t>(v >> 24);
    p[4] = static_cast<uint8_t>(v >> 32);
    p[5] = static_cast<uint8_t>(v >> 40);
    p[6] = static_cast<uint8_t>(v >> 48);
    p[7] = static_cast<uint8_t>(v >> 56);
}

inline void be64put(uint8_t* p, uint64_t v)
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

uint64_t datasetSize(int epoch)
{
    const uint64_t init = align64(c_datasetInitBytes);
    const uint64_t growth = align64(c_datasetGrowthBytes);
    return align64(init + growth * static_cast<uint64_t>(epoch));
}

ethash::hash256 datasetSeed(uint64_t epoch)
{
    uint8_t buf[32] = {0};
    le64put(buf, epoch);
    static const char suffix[] = "OlivetumhashDatasetSeed..........";
    constexpr size_t suffixLen = sizeof(suffix) - 1;
    const size_t copyLen = std::min<size_t>(suffixLen, sizeof(buf) - 8);
    std::memcpy(buf + 8, suffix, copyLen);
    return ethash::hash256_from_bytes(buf);
}

void buildDataset(std::vector<uint8_t>& data, int epoch)
{
    const uint64_t size = datasetSize(epoch);
    data.resize(size);
    const uint64_t chunkCount = size / 64;

    auto seed = datasetSeed(static_cast<uint64_t>(epoch));

    const unsigned workers = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned w = 0; w < workers; ++w)
    {
        threads.emplace_back([&, w]() {
            auto localSeed = seed;
            for (uint64_t i = w; i < chunkCount; i += workers)
            {
                le64put(localSeed.bytes + 16, i);
                auto sum = ethash::keccak512(localSeed.bytes, sizeof(localSeed.bytes));
                std::memcpy(&data[i * 64], sum.bytes, 64);
            }
        });
    }
    for (auto& t : threads)
        t.join();

    std::array<uint8_t, 64> tmp{};
    for (int round = 0; round < 3; ++round)
    {
        for (uint64_t i = 0; i < chunkCount; ++i)
        {
            const uint64_t target = (i + static_cast<uint64_t>(round + 1) * 17) % chunkCount;
            uint8_t* base = &data[i * 64];
            const uint8_t* ref = &data[target * 64];
            for (int j = 0; j < 64; ++j)
                tmp[j] = static_cast<uint8_t>(base[j] ^ ref[j]);

            auto ctxSeed = seed;
            le64put(ctxSeed.bytes + 16, i);
            le64put(ctxSeed.bytes + 24, static_cast<uint64_t>(round));

            uint8_t toHash[64 + 16];
            std::memcpy(toHash, tmp.data(), 64);
            std::memcpy(toHash + 64, ctxSeed.bytes + 16, 16);
            auto newChunk = ethash::keccak512(toHash, sizeof(toHash));
            std::memcpy(base, newChunk.bytes, 64);
        }
    }
}

struct DatasetCache
{
    std::mutex mtx;
    std::unordered_map<int, std::shared_ptr<std::vector<uint8_t>>> cache;
};

DatasetCache& datasetCache()
{
    static DatasetCache dc;
    return dc;
}

std::shared_ptr<std::vector<uint8_t>> getDatasetShared(int epoch)
{
    auto& dc = datasetCache();
    {
        std::lock_guard<std::mutex> lg(dc.mtx);
        auto it = dc.cache.find(epoch);
        if (it != dc.cache.end())
            return it->second;
    }
    auto data = std::make_shared<std::vector<uint8_t>>();
    buildDataset(*data, epoch);
    {
        std::lock_guard<std::mutex> lg(dc.mtx);
        dc.cache[epoch] = data;
    }
    return data;
}

OlivetumhashResult computeHash(const std::vector<uint8_t>& dataset, h256 const& headerHash,
    uint64_t nonce, uint64_t rounds)
{
    uint8_t seed[40];
    std::memcpy(seed, headerHash.data(), 32);

    uint8_t nonceBytes[8];
    be64put(nonceBytes, nonce);  // nonce encoded big-endian like BlockNonce in core-geth
    uint64_t nonceLE = le64(nonceBytes);
    std::memcpy(seed + 32, nonceBytes, 8);

    auto h512 = ethash::keccak512(seed, sizeof(seed));
    std::array<uint8_t, 64> mixBytes;
    std::memcpy(mixBytes.data(), h512.bytes, 64);

    uint64_t mixWords[8];
    for (int i = 0; i < 8; ++i)
        mixWords[i] = le64(&mixBytes[i * 8]);

    uint64_t chunkCount = dataset.size() / 64;
    if (chunkCount == 0)
        chunkCount = 1;

    // Program schedule (two keccak512 concatenated)
    uint8_t progBuf1[32 + 8];
    std::memcpy(progBuf1, headerHash.data(), 32);
    std::memcpy(progBuf1 + 32, nonceBytes, 8);
    auto progBytes1 = ethash::keccak512(progBuf1, sizeof(progBuf1));

    uint8_t progBuf2[64 + 32 + 8];
    std::memcpy(progBuf2, progBytes1.bytes, 64);
    std::memcpy(progBuf2 + 64, headerHash.data(), 32);
    std::memcpy(progBuf2 + 96, nonceBytes, 8);
    auto progBytes2 = ethash::keccak512(progBuf2, sizeof(progBuf2));

    std::vector<uint64_t> program;
    program.reserve(16);
    for (int i = 0; i < 8; ++i)
        program.push_back(le64(progBytes1.bytes + i * 8));
    for (int i = 0; i < 8; ++i)
        program.push_back(le64(progBytes2.bytes + i * 8));
    if (program.empty())
        program.push_back(0);

    uint64_t dynamicSalt = le64(headerHash.data() + 8) ^ nonceLE;
    constexpr uint64_t refreshInterval = 8;

    for (uint64_t i = 0; i < rounds; ++i)
    {
        if (refreshInterval != 0 && i != 0 && i % refreshInterval == 0)
        {
            uint8_t buf[64 + 32 + 8];
            std::memcpy(buf, mixBytes.data(), 64);
            std::memcpy(buf + 64, headerHash.data(), 32);
            std::memcpy(buf + 96, nonceBytes, 8);
            auto sum = ethash::keccak512(buf, sizeof(buf));
            for (size_t j = 0; j < program.size(); ++j)
            {
                size_t off = (j * 8) % 64;
                uint64_t word = le64(sum.bytes + off);
                program[j] ^= word;
            }
            dynamicSalt ^= le64(sum.bytes);
        }

        uint64_t progWord = program[i % program.size()] ^ (i * 0x9e3779b97f4a7c15ull);
        int sourceLane = static_cast<int>((progWord >> 5) & 7);
        int rotateAmt = static_cast<int>(progWord & 63) + 1;

        uint64_t index = mixWords[sourceLane] ^ progWord ^ le64(headerHash.data());
        index ^= (i + static_cast<uint64_t>(sourceLane)) * 0x517cc1b727220a95ull;
        uint64_t chunkOffset = (index % chunkCount) * 64;
        const uint8_t* chunk = &dataset[chunkOffset];
        uint64_t chunkWords[8];
        for (int j = 0; j < 8; ++j)
            chunkWords[j] = le64(chunk + j * 8);

        uint64_t index2 = mixWords[(sourceLane + 3) & 7] ^ progWord ^ dynamicSalt ^
                          (rotl64(static_cast<uint64_t>(i), sourceLane) & 0xffffull);
        index2 ^= le64(headerHash.data() + 16);
        index2 ^= (i + static_cast<uint64_t>(sourceLane * 3 + 1)) * 0x94d049bb133111ebull;
        uint64_t chunkOffset2 = (index2 % chunkCount) * 64;
        const uint8_t* chunk2 = &dataset[chunkOffset2];
        uint64_t chunkWords2[8];
        for (int j = 0; j < 8; ++j)
            chunkWords2[j] = le64(chunk2 + j * 8);

        uint64_t index3 = mixWords[(sourceLane + 5) & 7] ^ dynamicSalt ^ progWord ^
                          le64(headerHash.data() + 24);
        index3 ^= (i * 0x2545f4914f6cdd1dull) + static_cast<uint64_t>(sourceLane << 3);
        uint64_t chunkOffset3 = (index3 % chunkCount) * 64;
        const uint8_t* chunk3 = &dataset[chunkOffset3];
        uint64_t chunkWords3[8];
        for (int j = 0; j < 8; ++j)
            chunkWords3[j] = le64(chunk3 + j * 8);

        for (int lane = 0; lane < 8; ++lane)
        {
            uint64_t data1 = chunkWords[(lane + sourceLane) & 7];
            uint64_t data2 = chunkWords2[(lane + (sourceLane ^ 3)) & 7];
            uint64_t data3 = chunkWords3[(lane + (sourceLane ^ 5)) & 7];
            mixWords[lane] ^= data1 ^ data3;
            mixWords[lane] = rotl64(
                mixWords[lane] + data1 * 0x9e3779b97f4a7c15ull + data2 + data3 * 0x6a09e667f3bcc908ull,
                rotateAmt + (lane & 7));
            mixWords[lane] ^= rotl64(progWord ^ dynamicSalt ^ data2 ^ data3, lane + 1);
        }

        dynamicSalt ^= mixWords[(sourceLane + 1) & 7] + mixWords[(sourceLane + 2) & 7];
        dynamicSalt = rotl64(dynamicSalt, static_cast<unsigned>(rotateAmt & 31));
    }

    for (int i = 0; i < 8; ++i)
        le64put(mixBytes.data() + i * 8, mixWords[i]);

    auto mixDigest = ethash::keccak256(mixBytes.data(), mixBytes.size());

    uint8_t finalBuf[32 + 32 + 8];
    std::memcpy(finalBuf, mixDigest.bytes, 32);
    std::memcpy(finalBuf + 32, headerHash.data(), 32);
    std::memcpy(finalBuf + 64, nonceBytes, 8);
    auto finalDigest = ethash::keccak256(finalBuf, sizeof(finalBuf));

    h256 mixOut{mixDigest.bytes, h256::ConstructFromPointer};
    h256 finalOut{finalDigest.bytes, h256::ConstructFromPointer};
    return {finalOut, mixOut};
}

}  // namespace

std::vector<uint8_t> const& OlivetumhashAux::dataset(int epoch)
{
    return *getDatasetShared(epoch);
}

std::shared_ptr<std::vector<uint8_t>> OlivetumhashAux::datasetShared(int epoch)
{
    return getDatasetShared(epoch);
}

OlivetumhashResult OlivetumhashAux::eval(int epoch, h256 const& headerHash, uint64_t nonce) noexcept
{
    auto ds = dataset(epoch);
    return computeHash(ds, headerHash, nonce, c_mixRounds);
}
