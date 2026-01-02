#include "ethash_cuda_miner_kernel.h"

#include "cuda_helper.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Olivetumhash GPU path (naive, referencyjna implementacja)

__device__ __constant__ uint8_t* d_oliv_dataset = nullptr;
__device__ __constant__ uint64_t d_oliv_dataset_bytes = 0;
__device__ __constant__ uint32_t d_oliv_chunk_count = 0;
__device__ __constant__ uint8_t d_oliv_header[32];
__device__ __constant__ uint8_t d_oliv_target[32];

__device__ __forceinline__ uint64_t rotl64(uint64_t v, unsigned int r)
{
    r &= 63;
    if (r == 0)
        return v;
    return (v << r) | (v >> (64 - r));
}

__device__ __forceinline__ uint64_t le64(const uint8_t* p)
{
    return ((uint64_t)p[0]) | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

__device__ __forceinline__ void le64put(uint8_t* p, uint64_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
    p[4] = (uint8_t)(v >> 32);
    p[5] = (uint8_t)(v >> 40);
    p[6] = (uint8_t)(v >> 48);
    p[7] = (uint8_t)(v >> 56);
}

__device__ __forceinline__ void be64put(uint8_t* p, uint64_t v)
{
    p[0] = (uint8_t)(v >> 56);
    p[1] = (uint8_t)(v >> 48);
    p[2] = (uint8_t)(v >> 40);
    p[3] = (uint8_t)(v >> 32);
    p[4] = (uint8_t)(v >> 24);
    p[5] = (uint8_t)(v >> 16);
    p[6] = (uint8_t)(v >> 8);
    p[7] = (uint8_t)(v);
}

__device__ __forceinline__ uint64_t ldg_u64(const uint64_t* p)
{
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ ulonglong2 ldg_u64x2(const ulonglong2* p)
{
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ int cmp256(const uint8_t* a, const uint8_t* b)
{
    for (int i = 0; i < 32; ++i)
    {
        uint8_t av = a[i];
        uint8_t bv = b[i];
        if (av < bv)
            return -1;
        if (av > bv)
            return 1;
    }
    return 0;
}

__device__ __constant__ uint64_t c_keccakf_rndc[24] = {0x0000000000000001ULL,
    0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL,
    0x8000000000008009ULL, 0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL, 0x000000008000808bULL,
    0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL,
    0x800000008000000aULL, 0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL};

// Keccak-f[1600] permutation from ethash reference (pad10*1).
__device__ __forceinline__ void keccakf(uint64_t state[25])
{
    uint64_t Aba, Abe, Abi, Abo, Abu;
    uint64_t Aga, Age, Agi, Ago, Agu;
    uint64_t Aka, Ake, Aki, Ako, Aku;
    uint64_t Ama, Ame, Ami, Amo, Amu;
    uint64_t Asa, Ase, Asi, Aso, Asu;

    uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
    uint64_t Ega, Ege, Egi, Ego, Egu;
    uint64_t Eka, Eke, Eki, Eko, Eku;
    uint64_t Ema, Eme, Emi, Emo, Emu;
    uint64_t Esa, Ese, Esi, Eso, Esu;

    uint64_t Ba, Be, Bi, Bo, Bu;

    uint64_t Da, De, Di, Do, Du;

    Aba = state[0];
    Abe = state[1];
    Abi = state[2];
    Abo = state[3];
    Abu = state[4];
    Aga = state[5];
    Age = state[6];
    Agi = state[7];
    Ago = state[8];
    Agu = state[9];
    Aka = state[10];
    Ake = state[11];
    Aki = state[12];
    Ako = state[13];
    Aku = state[14];
    Ama = state[15];
    Ame = state[16];
    Ami = state[17];
    Amo = state[18];
    Amu = state[19];
    Asa = state[20];
    Ase = state[21];
    Asi = state[22];
    Aso = state[23];
    Asu = state[24];

    for (int n = 0; n < 24; n += 2)
    {
        Ba = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
        Be = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        Bi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
        Bo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        Bu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

        Da = Bu ^ rotl64(Be, 1);
        De = Ba ^ rotl64(Bi, 1);
        Di = Be ^ rotl64(Bo, 1);
        Do = Bi ^ rotl64(Bu, 1);
        Du = Bo ^ rotl64(Ba, 1);

        Ba = Aba ^ Da;
        Be = rotl64(Age ^ De, 44);
        Bi = rotl64(Aki ^ Di, 43);
        Bo = rotl64(Amo ^ Do, 21);
        Bu = rotl64(Asu ^ Du, 14);
        Eba = Ba ^ (~Be & Bi) ^ c_keccakf_rndc[n];
        Ebe = Be ^ (~Bi & Bo);
        Ebi = Bi ^ (~Bo & Bu);
        Ebo = Bo ^ (~Bu & Ba);
        Ebu = Bu ^ (~Ba & Be);

        Ba = rotl64(Abo ^ Do, 28);
        Be = rotl64(Agu ^ Du, 20);
        Bi = rotl64(Aka ^ Da, 3);
        Bo = rotl64(Ame ^ De, 45);
        Bu = rotl64(Asi ^ Di, 61);
        Ega = Ba ^ (~Be & Bi);
        Ege = Be ^ (~Bi & Bo);
        Egi = Bi ^ (~Bo & Bu);
        Ego = Bo ^ (~Bu & Ba);
        Egu = Bu ^ (~Ba & Be);

        Ba = rotl64(Abe ^ De, 1);
        Be = rotl64(Agi ^ Di, 6);
        Bi = rotl64(Ako ^ Do, 25);
        Bo = rotl64(Amu ^ Du, 8);
        Bu = rotl64(Asa ^ Da, 18);
        Eka = Ba ^ (~Be & Bi);
        Eke = Be ^ (~Bi & Bo);
        Eki = Bi ^ (~Bo & Bu);
        Eko = Bo ^ (~Bu & Ba);
        Eku = Bu ^ (~Ba & Be);

        Ba = rotl64(Abu ^ Du, 27);
        Be = rotl64(Aga ^ Da, 36);
        Bi = rotl64(Ake ^ De, 10);
        Bo = rotl64(Ami ^ Di, 15);
        Bu = rotl64(Aso ^ Do, 56);
        Ema = Ba ^ (~Be & Bi);
        Eme = Be ^ (~Bi & Bo);
        Emi = Bi ^ (~Bo & Bu);
        Emo = Bo ^ (~Bu & Ba);
        Emu = Bu ^ (~Ba & Be);

        Ba = rotl64(Abi ^ Di, 62);
        Be = rotl64(Ago ^ Do, 55);
        Bi = rotl64(Aku ^ Du, 39);
        Bo = rotl64(Ama ^ Da, 41);
        Bu = rotl64(Ase ^ De, 2);
        Esa = Ba ^ (~Be & Bi);
        Ese = Be ^ (~Bi & Bo);
        Esi = Bi ^ (~Bo & Bu);
        Eso = Bo ^ (~Bu & Ba);
        Esu = Bu ^ (~Ba & Be);

        Ba = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
        Be = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        Bi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
        Bo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        Bu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

        Da = Bu ^ rotl64(Be, 1);
        De = Ba ^ rotl64(Bi, 1);
        Di = Be ^ rotl64(Bo, 1);
        Do = Bi ^ rotl64(Bu, 1);
        Du = Bo ^ rotl64(Ba, 1);

        Ba = Eba ^ Da;
        Be = rotl64(Ege ^ De, 44);
        Bi = rotl64(Eki ^ Di, 43);
        Bo = rotl64(Emo ^ Do, 21);
        Bu = rotl64(Esu ^ Du, 14);
        Aba = Ba ^ (~Be & Bi) ^ c_keccakf_rndc[n + 1];
        Abe = Be ^ (~Bi & Bo);
        Abi = Bi ^ (~Bo & Bu);
        Abo = Bo ^ (~Bu & Ba);
        Abu = Bu ^ (~Ba & Be);

        Ba = rotl64(Ebo ^ Do, 28);
        Be = rotl64(Egu ^ Du, 20);
        Bi = rotl64(Eka ^ Da, 3);
        Bo = rotl64(Eme ^ De, 45);
        Bu = rotl64(Esi ^ Di, 61);
        Aga = Ba ^ (~Be & Bi);
        Age = Be ^ (~Bi & Bo);
        Agi = Bi ^ (~Bo & Bu);
        Ago = Bo ^ (~Bu & Ba);
        Agu = Bu ^ (~Ba & Be);

        Ba = rotl64(Ebe ^ De, 1);
        Be = rotl64(Egi ^ Di, 6);
        Bi = rotl64(Eko ^ Do, 25);
        Bo = rotl64(Emu ^ Du, 8);
        Bu = rotl64(Esa ^ Da, 18);
        Aka = Ba ^ (~Be & Bi);
        Ake = Be ^ (~Bi & Bo);
        Aki = Bi ^ (~Bo & Bu);
        Ako = Bo ^ (~Bu & Ba);
        Aku = Bu ^ (~Ba & Be);

        Ba = rotl64(Ebu ^ Du, 27);
        Be = rotl64(Ega ^ Da, 36);
        Bi = rotl64(Eke ^ De, 10);
        Bo = rotl64(Emi ^ Di, 15);
        Bu = rotl64(Eso ^ Do, 56);
        Ama = Ba ^ (~Be & Bi);
        Ame = Be ^ (~Bi & Bo);
        Ami = Bi ^ (~Bo & Bu);
        Amo = Bo ^ (~Bu & Ba);
        Amu = Bu ^ (~Ba & Be);

        Ba = rotl64(Ebi ^ Di, 62);
        Be = rotl64(Ego ^ Do, 55);
        Bi = rotl64(Eku ^ Du, 39);
        Bo = rotl64(Ema ^ Da, 41);
        Bu = rotl64(Ese ^ De, 2);
        Asa = Ba ^ (~Be & Bi);
        Ase = Be ^ (~Bi & Bo);
        Asi = Bi ^ (~Bo & Bu);
        Aso = Bo ^ (~Bu & Ba);
        Asu = Bu ^ (~Ba & Be);
    }

    state[0] = Aba;
    state[1] = Abe;
    state[2] = Abi;
    state[3] = Abo;
    state[4] = Abu;
    state[5] = Aga;
    state[6] = Age;
    state[7] = Agi;
    state[8] = Ago;
    state[9] = Agu;
    state[10] = Aka;
    state[11] = Ake;
    state[12] = Aki;
    state[13] = Ako;
    state[14] = Aku;
    state[15] = Ama;
    state[16] = Ame;
    state[17] = Ami;
    state[18] = Amo;
    state[19] = Amu;
    state[20] = Asa;
    state[21] = Ase;
    state[22] = Asi;
    state[23] = Aso;
    state[24] = Asu;
}

__device__ __forceinline__ void keccak_hash(
    uint8_t* out, size_t outlen, size_t rateBytes, const uint8_t* in, size_t inlen)
{
    uint64_t st[25];
    for (int i = 0; i < 25; ++i)
        st[i] = 0;

    // Absorb
    while (inlen >= rateBytes)
    {
        for (size_t i = 0; i < rateBytes / 8; ++i)
            st[i] ^= le64(in + i * 8);
        keccakf(st);
        in += rateBytes;
        inlen -= rateBytes;
    }

    // rateBytes is 72 (keccak512) or 136 (keccak256).
    uint8_t block[136];
    for (size_t i = 0; i < rateBytes; ++i)
        block[i] = 0;
    for (size_t i = 0; i < inlen; ++i)
        block[i] = in[i];
    block[inlen] ^= 0x01;
    block[rateBytes - 1] ^= 0x80;

    for (size_t i = 0; i < rateBytes / 8; ++i)
        st[i] ^= le64(block + i * 8);
    keccakf(st);

    // Squeeze
    size_t produced = 0;
    while (produced < outlen)
    {
        size_t chunk = (outlen - produced) < rateBytes ? (outlen - produced) : rateBytes;
        for (size_t i = 0; i < chunk / 8; ++i)
            le64put(out + produced + i * 8, st[i]);
        produced += chunk;
        if (produced < outlen)
            keccakf(st);
    }
}

__device__ __forceinline__ void keccak512(uint8_t* out, const uint8_t* in, size_t inlen)
{
    keccak_hash(out, 64, 72, in, inlen);
}

__device__ __forceinline__ void keccak256(uint8_t* out, const uint8_t* in, size_t inlen)
{
    keccak_hash(out, 32, 136, in, inlen);
}

// launch_bounds helps NVCC keep register pressure in check for block sizes up to 256.
__device__ __forceinline__ void olivetumhash_search_kernel_body(
    volatile Search_results* g_output, uint64_t start_nonce)
{
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + gid;

    const uint64_t header0 = le64(d_oliv_header);
    const uint64_t header1 = le64(d_oliv_header + 8);
    const uint64_t header2 = le64(d_oliv_header + 16);
    const uint64_t header3 = le64(d_oliv_header + 24);

    uint8_t nonceBytes[8];
    be64put(nonceBytes, nonce);
    uint64_t nonceLE = le64(nonceBytes);

    uint8_t seed[40];
    for (int i = 0; i < 32; ++i)
        seed[i] = d_oliv_header[i];
    for (int i = 0; i < 8; ++i)
        seed[32 + i] = nonceBytes[i];

    alignas(8) uint8_t h512[64];
    keccak512(h512, seed, sizeof(seed));

    uint64_t mixWords[8];
    const uint64_t* mixInitWords = reinterpret_cast<const uint64_t*>(h512);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        mixWords[i] = mixInitWords[i];

    uint64_t chunkCount = d_oliv_chunk_count;
    if (chunkCount == 0)
        chunkCount = 1;

    alignas(8) uint8_t progBuf2[104];
    for (int i = 0; i < 64; ++i)
        progBuf2[i] = h512[i];
    for (int i = 0; i < 32; ++i)
        progBuf2[64 + i] = d_oliv_header[i];
    for (int i = 0; i < 8; ++i)
        progBuf2[96 + i] = nonceBytes[i];
    alignas(8) uint8_t progBytes2[64];
    keccak512(progBytes2, progBuf2, sizeof(progBuf2));

    uint64_t program[16];
    const uint64_t* progWords1 = reinterpret_cast<const uint64_t*>(h512);
    const uint64_t* progWords2 = reinterpret_cast<const uint64_t*>(progBytes2);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        program[i] = progWords1[i];
        program[8 + i] = progWords2[i];
    }
    const int programLen = 16;

    uint64_t dynamicSalt = header1 ^ nonceLE;
    const uint64_t refreshInterval = 8;
    const uint64_t* dataset64 = reinterpret_cast<const uint64_t*>(d_oliv_dataset);

    for (uint64_t i = 0; i < 64; ++i)
    {
        if (refreshInterval != 0 && i != 0 && i % refreshInterval == 0)
        {
            alignas(8) uint8_t buf[104];
            for (int j = 0; j < 64; ++j)
                buf[j] = h512[j];
            for (int j = 0; j < 32; ++j)
                buf[64 + j] = d_oliv_header[j];
            for (int j = 0; j < 8; ++j)
                buf[96 + j] = nonceBytes[j];
            uint64_t sumWords[8];
            keccak512(reinterpret_cast<uint8_t*>(sumWords), buf, sizeof(buf));
#pragma unroll
            for (int j = 0; j < programLen; ++j)
                program[j] ^= sumWords[j & 7];
            dynamicSalt ^= sumWords[0];
        }

        uint64_t progWord = program[i % programLen] ^ (i * 0x9e3779b97f4a7c15ULL);
        int sourceLane = (int)((progWord >> 5) & 7);
        int rotateAmt = (int)(progWord & 63) + 1;

        uint64_t index = mixWords[sourceLane] ^ progWord ^ header0;
        index ^= (i + (uint64_t)sourceLane) * 0x517cc1b727220a95ULL;
        uint64_t chunkIndex = index % chunkCount;
        const uint64_t* chunk = dataset64 + (chunkIndex * 8);
        uint64_t chunkWords[8];
        const ulonglong2* chunkVec = reinterpret_cast<const ulonglong2*>(chunk);
        ulonglong2 c0 = ldg_u64x2(chunkVec + 0);
        ulonglong2 c1 = ldg_u64x2(chunkVec + 1);
        ulonglong2 c2 = ldg_u64x2(chunkVec + 2);
        ulonglong2 c3 = ldg_u64x2(chunkVec + 3);
        chunkWords[0] = c0.x;
        chunkWords[1] = c0.y;
        chunkWords[2] = c1.x;
        chunkWords[3] = c1.y;
        chunkWords[4] = c2.x;
        chunkWords[5] = c2.y;
        chunkWords[6] = c3.x;
        chunkWords[7] = c3.y;

        uint64_t index2 = mixWords[(sourceLane + 3) & 7] ^ progWord ^ dynamicSalt ^
                          (rotl64((uint64_t)i, sourceLane) & 0xffffULL);
        index2 ^= header2;
        index2 ^= (i + (uint64_t)(sourceLane * 3 + 1)) * 0x94d049bb133111ebULL;
        uint64_t chunkIndex2 = index2 % chunkCount;
        const uint64_t* chunk2 = dataset64 + (chunkIndex2 * 8);
        uint64_t chunkWords2[8];
        const ulonglong2* chunk2v = reinterpret_cast<const ulonglong2*>(chunk2);
        ulonglong2 c20 = ldg_u64x2(chunk2v + 0);
        ulonglong2 c21 = ldg_u64x2(chunk2v + 1);
        ulonglong2 c22 = ldg_u64x2(chunk2v + 2);
        ulonglong2 c23 = ldg_u64x2(chunk2v + 3);
        chunkWords2[0] = c20.x;
        chunkWords2[1] = c20.y;
        chunkWords2[2] = c21.x;
        chunkWords2[3] = c21.y;
        chunkWords2[4] = c22.x;
        chunkWords2[5] = c22.y;
        chunkWords2[6] = c23.x;
        chunkWords2[7] = c23.y;

        uint64_t index3 = mixWords[(sourceLane + 5) & 7] ^ dynamicSalt ^ progWord ^
                          header3;
        index3 ^= (i * 0x2545f4914f6cdd1dULL) + (uint64_t)(sourceLane << 3);
        uint64_t chunkIndex3 = index3 % chunkCount;
        const uint64_t* chunk3 = dataset64 + (chunkIndex3 * 8);
        uint64_t chunkWords3[8];
        const ulonglong2* chunk3v = reinterpret_cast<const ulonglong2*>(chunk3);
        ulonglong2 c30 = ldg_u64x2(chunk3v + 0);
        ulonglong2 c31 = ldg_u64x2(chunk3v + 1);
        ulonglong2 c32 = ldg_u64x2(chunk3v + 2);
        ulonglong2 c33 = ldg_u64x2(chunk3v + 3);
        chunkWords3[0] = c30.x;
        chunkWords3[1] = c30.y;
        chunkWords3[2] = c31.x;
        chunkWords3[3] = c31.y;
        chunkWords3[4] = c32.x;
        chunkWords3[5] = c32.y;
        chunkWords3[6] = c33.x;
        chunkWords3[7] = c33.y;

        for (int lane = 0; lane < 8; ++lane)
        {
            uint64_t data1 = chunkWords[(lane + sourceLane) & 7];
            uint64_t data2 = chunkWords2[(lane + (sourceLane ^ 3)) & 7];
            uint64_t data3 = chunkWords3[(lane + (sourceLane ^ 5)) & 7];
            mixWords[lane] ^= data1 ^ data3;
            mixWords[lane] = rotl64(
                mixWords[lane] + data1 * 0x9e3779b97f4a7c15ULL + data2 +
                    data3 * 0x6a09e667f3bcc908ULL,
                rotateAmt + (lane & 7));
            mixWords[lane] ^= rotl64(progWord ^ dynamicSalt ^ data2 ^ data3, lane + 1);
        }

        dynamicSalt ^= mixWords[(sourceLane + 1) & 7] + mixWords[(sourceLane + 2) & 7];
        dynamicSalt = rotl64(dynamicSalt, (unsigned)(rotateAmt & 31));
    }

    alignas(8) uint8_t mixBytes[64];
    uint64_t* mixBytes64 = reinterpret_cast<uint64_t*>(mixBytes);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        mixBytes64[i] = mixWords[i];

    uint8_t mixDigest[32];
    keccak256(mixDigest, mixBytes, sizeof(mixBytes));

    uint8_t finalBuf[72];
    for (int i = 0; i < 32; ++i)
        finalBuf[i] = mixDigest[i];
    for (int i = 0; i < 32; ++i)
        finalBuf[32 + i] = d_oliv_header[i];
    for (int i = 0; i < 8; ++i)
        finalBuf[64 + i] = nonceBytes[i];
    uint8_t finalDigest[32];
    keccak256(finalDigest, finalBuf, sizeof(finalBuf));

    if (cmp256(finalDigest, d_oliv_target) > 0)
        return;

    uint32_t indexOut = atomicInc((uint32_t*)&g_output->count, 0xffffffff);
    if (indexOut >= MAX_SEARCH_RESULTS)
        return;

    g_output->result[indexOut].gid = gid;
    // Store mixDigest as raw 32 bytes (CPU-compatible).
    uint8_t* outBytes = (uint8_t*)(g_output->result[indexOut].mix);
    for (int i = 0; i < 32; ++i)
        outBytes[i] = mixDigest[i];
}

__global__ __launch_bounds__(256, 2) void olivetumhash_search_kernel(
    volatile Search_results* g_output, uint64_t start_nonce)
{
    olivetumhash_search_kernel_body(g_output, start_nonce);
}

__global__ __launch_bounds__(128, 3) void olivetumhash_search_kernel_128(
    volatile Search_results* g_output, uint64_t start_nonce)
{
    olivetumhash_search_kernel_body(g_output, start_nonce);
}

void set_olivetum_dataset(uint8_t* dataset, uint64_t datasetSize)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_oliv_dataset, &dataset, sizeof(uint8_t*)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_oliv_dataset_bytes, &datasetSize, sizeof(uint64_t)));
    uint32_t chunkCount = (uint32_t)(datasetSize / 64);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_oliv_chunk_count, &chunkCount, sizeof(uint32_t)));
}

void set_olivetum_header(const uint8_t* header)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_oliv_header, header, 32));
}

void set_olivetum_target(const uint8_t* targetBytes)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_oliv_target, targetBytes, 32));
}

void run_olivetumhash_search(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
    volatile Search_results* g_output, uint64_t start_nonce)
{
    if (blockSize <= 128)
        olivetumhash_search_kernel_128<<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);
    else
        olivetumhash_search_kernel<<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);
    CUDA_SAFE_CALL(cudaGetLastError());
}
