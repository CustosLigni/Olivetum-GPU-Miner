// Olivetumhash OpenCL kernel (reference, unoptimized).

#define FNV_PRIME 0x01000193U

// Parameters injected by the host
// WORKSIZE
// MAX_OUTPUTS

typedef struct
{
    uint gid;
    uint mix[8];
    uint pad[7];
} result_t;

typedef struct
{
    result_t rslt[MAX_OUTPUTS];
    uint count;
    uint hashCount;
    uint abort;
} SearchResults;

inline ulong rotl64(ulong v, uint r)
{
    r &= 63;
    if (r == 0)
        return v;
    return (v << r) | (v >> (64 - r));
}

// Use generic pointer qualifiers so kernels compile across drivers that are strict about
// address spaces.
inline ulong le64(const uchar* p)
{
    return ((ulong)p[0]) | ((ulong)p[1] << 8) | ((ulong)p[2] << 16) | ((ulong)p[3] << 24) |
           ((ulong)p[4] << 32) | ((ulong)p[5] << 40) | ((ulong)p[6] << 48) | ((ulong)p[7] << 56);
}

inline void le64put(__private uchar* p, ulong v)
{
    p[0] = (uchar)v;
    p[1] = (uchar)(v >> 8);
    p[2] = (uchar)(v >> 16);
    p[3] = (uchar)(v >> 24);
    p[4] = (uchar)(v >> 32);
    p[5] = (uchar)(v >> 40);
    p[6] = (uchar)(v >> 48);
    p[7] = (uchar)(v >> 56);
}

inline void be64put(__private uchar* p, ulong v)
{
    p[0] = (uchar)(v >> 56);
    p[1] = (uchar)(v >> 48);
    p[2] = (uchar)(v >> 40);
    p[3] = (uchar)(v >> 32);
    p[4] = (uchar)(v >> 24);
    p[5] = (uchar)(v >> 16);
    p[6] = (uchar)(v >> 8);
    p[7] = (uchar)(v);
}

inline int cmp256(const uchar* a, const uchar* b)
{
    for (int i = 0; i < 32; ++i)
    {
        uchar av = a[i];
        uchar bv = b[i];
        if (av < bv)
            return -1;
        if (av > bv)
            return 1;
    }
    return 0;
}

__constant ulong c_keccakf_rndc[24] = {0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808aUL, 0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000aUL, 0x000000008000808bUL, 0x800000000000008bUL,
    0x8000000000008089UL, 0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL};

// Keccak-f[1600] permutation from ethash reference (pad10*1).
inline void keccakf(__private ulong st[25])
{
    ulong Aba, Abe, Abi, Abo, Abu;
    ulong Aga, Age, Agi, Ago, Agu;
    ulong Aka, Ake, Aki, Ako, Aku;
    ulong Ama, Ame, Ami, Amo, Amu;
    ulong Asa, Ase, Asi, Aso, Asu;

    ulong Eba, Ebe, Ebi, Ebo, Ebu;
    ulong Ega, Ege, Egi, Ego, Egu;
    ulong Eka, Eke, Eki, Eko, Eku;
    ulong Ema, Eme, Emi, Emo, Emu;
    ulong Esa, Ese, Esi, Eso, Esu;

    ulong Ba, Be, Bi, Bo, Bu;

    ulong Da, De, Di, Do, Du;

    Aba = st[0];
    Abe = st[1];
    Abi = st[2];
    Abo = st[3];
    Abu = st[4];
    Aga = st[5];
    Age = st[6];
    Agi = st[7];
    Ago = st[8];
    Agu = st[9];
    Aka = st[10];
    Ake = st[11];
    Aki = st[12];
    Ako = st[13];
    Aku = st[14];
    Ama = st[15];
    Ame = st[16];
    Ami = st[17];
    Amo = st[18];
    Amu = st[19];
    Asa = st[20];
    Ase = st[21];
    Asi = st[22];
    Aso = st[23];
    Asu = st[24];

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

    st[0] = Aba;
    st[1] = Abe;
    st[2] = Abi;
    st[3] = Abo;
    st[4] = Abu;
    st[5] = Aga;
    st[6] = Age;
    st[7] = Agi;
    st[8] = Ago;
    st[9] = Agu;
    st[10] = Aka;
    st[11] = Ake;
    st[12] = Aki;
    st[13] = Ako;
    st[14] = Aku;
    st[15] = Ama;
    st[16] = Ame;
    st[17] = Ami;
    st[18] = Amo;
    st[19] = Amu;
    st[20] = Asa;
    st[21] = Ase;
    st[22] = Asi;
    st[23] = Aso;
    st[24] = Asu;
}

inline void keccak_hash(__private uchar* out, uint outlen, uint rateBytes,
    const __private uchar* in, uint inlen)
{
    __private ulong st[25];
    for (int i = 0; i < 25; ++i)
        st[i] = 0;

    uint offset = 0;
    while (inlen >= rateBytes)
    {
        for (uint i = 0; i < rateBytes / 8; ++i)
            st[i] ^= le64(in + offset + i * 8);
        keccakf(st);
        offset += rateBytes;
        inlen -= rateBytes;
    }

    __private uchar block[200];
    for (uint i = 0; i < 200; ++i)
        block[i] = 0;
    for (uint i = 0; i < inlen; ++i)
        block[i] = in[offset + i];
    block[inlen] ^= 0x01;
    block[rateBytes - 1] ^= 0x80;

    for (uint i = 0; i < rateBytes / 8; ++i)
        st[i] ^= le64(block + i * 8);
    keccakf(st);

    uint produced = 0;
    while (produced < outlen)
    {
        uint chunk = (outlen - produced) < rateBytes ? (outlen - produced) : rateBytes;
        for (uint i = 0; i < chunk / 8; ++i)
            le64put(out + produced + i * 8, st[i]);
        produced += chunk;
        if (produced < outlen)
            keccakf(st);
    }
}

inline void keccak512(__private uchar* out, const __private uchar* in, uint inlen)
{
    keccak_hash(out, 64, 72, in, inlen);
}

inline void keccak256(__private uchar* out, const __private uchar* in, uint inlen)
{
    keccak_hash(out, 32, 136, in, inlen);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(__global volatile SearchResults* g_output, __global const uchar* header,
    __global const uchar* dataset, uint chunkCount, ulong startNonce, __global const uchar* target)
{
#ifdef FAST_EXIT
    if (g_output->abort)
        return;
#endif

    const uint gid = get_global_id(0);
    ulong nonce = startNonce + (ulong)gid;

    __private uchar nonceBytes[8];
    be64put(nonceBytes, nonce);
    ulong nonceLE = le64(nonceBytes);

    __private uchar seed[40];
    for (int i = 0; i < 32; ++i)
        seed[i] = header[i];
    for (int i = 0; i < 8; ++i)
        seed[32 + i] = nonceBytes[i];

    __private uchar h512[64];
    keccak512(h512, seed, 40);

    __private uchar mixBytes[64];
    for (int i = 0; i < 64; ++i)
        mixBytes[i] = h512[i];

    __private ulong mixWords[8];
    for (int i = 0; i < 8; ++i)
        mixWords[i] = le64(mixBytes + i * 8);

    __private uchar progBuf2[104];
    for (int i = 0; i < 64; ++i)
        progBuf2[i] = h512[i];
    for (int i = 0; i < 32; ++i)
        progBuf2[64 + i] = header[i];
    for (int i = 0; i < 8; ++i)
        progBuf2[96 + i] = nonceBytes[i];
    __private uchar progBytes2[64];
    keccak512(progBytes2, progBuf2, 104);

    __private ulong program[16];
    for (int i = 0; i < 8; ++i)
        program[i] = le64(h512 + i * 8);
    for (int i = 0; i < 8; ++i)
        program[8 + i] = le64(progBytes2 + i * 8);
    // Refresh hash is constant for this nonce: keccak512(h512||header||nonce) == progBytes2.
    __private ulong refreshWords[8];
    for (int i = 0; i < 8; ++i)
        refreshWords[i] = le64(progBytes2 + i * 8);

    ulong dynamicSalt = le64(header + 8) ^ nonceLE;
    const ulong refreshInterval = 8;

    for (ulong i = 0; i < 64; ++i)
    {
        if (refreshInterval != 0 && i != 0 && (i % refreshInterval) == 0)
        {
            for (int j = 0; j < 16; ++j)
            {
                program[j] ^= refreshWords[j & 7];
            }
            dynamicSalt ^= refreshWords[0];
        }

        ulong progWord = program[i & 15] ^ (i * 0x9e3779b97f4a7c15UL);
        int sourceLane = (int)((progWord >> 5) & 7);
        int rotateAmt = (int)(progWord & 63) + 1;

        ulong index = mixWords[sourceLane] ^ progWord ^ le64(header);
        index ^= (i + (ulong)sourceLane) * 0x517cc1b727220a95UL;
        uint chunkOffset = (uint)((index % chunkCount) * 64);

        __private ulong chunkWords[8];
        for (int j = 0; j < 8; ++j)
            chunkWords[j] = le64(dataset + chunkOffset + j * 8);

        ulong index2 = mixWords[(sourceLane + 3) & 7] ^ progWord ^ dynamicSalt ^
                      (rotl64(i, (uint)sourceLane) & 0xffffUL);
        index2 ^= le64(header + 16);
        index2 ^= (i + (ulong)(sourceLane * 3 + 1)) * 0x94d049bb133111ebUL;
        uint chunkOffset2 = (uint)((index2 % chunkCount) * 64);

        __private ulong chunkWords2[8];
        for (int j = 0; j < 8; ++j)
            chunkWords2[j] = le64(dataset + chunkOffset2 + j * 8);

        ulong index3 = mixWords[(sourceLane + 5) & 7] ^ dynamicSalt ^ progWord ^ le64(header + 24);
        index3 ^= (i * 0x2545f4914f6cdd1dUL) + (ulong)(sourceLane << 3);
        uint chunkOffset3 = (uint)((index3 % chunkCount) * 64);

        __private ulong chunkWords3[8];
        for (int j = 0; j < 8; ++j)
            chunkWords3[j] = le64(dataset + chunkOffset3 + j * 8);

        for (int lane = 0; lane < 8; ++lane)
        {
            ulong data1 = chunkWords[(lane + sourceLane) & 7];
            ulong data2 = chunkWords2[(lane + (sourceLane ^ 3)) & 7];
            ulong data3 = chunkWords3[(lane + (sourceLane ^ 5)) & 7];
            mixWords[lane] ^= data1 ^ data3;
            mixWords[lane] = rotl64(mixWords[lane] + data1 * 0x9e3779b97f4a7c15UL + data2 +
                                        data3 * 0x6a09e667f3bcc908UL,
                rotateAmt + (lane & 7));
            mixWords[lane] ^= rotl64(progWord ^ dynamicSalt ^ data2 ^ data3, (uint)lane + 1);
        }

        dynamicSalt ^= mixWords[(sourceLane + 1) & 7] + mixWords[(sourceLane + 2) & 7];
        dynamicSalt = rotl64(dynamicSalt, (uint)(rotateAmt & 31));
    }

    for (int i = 0; i < 8; ++i)
        le64put(mixBytes + i * 8, mixWords[i]);

    __private uchar mixDigest[32];
    keccak256(mixDigest, mixBytes, 64);

    __private uchar finalBuf[72];
    for (int i = 0; i < 32; ++i)
        finalBuf[i] = mixDigest[i];
    for (int i = 0; i < 32; ++i)
        finalBuf[32 + i] = header[i];
    for (int i = 0; i < 8; ++i)
        finalBuf[64 + i] = nonceBytes[i];
    __private uchar finalDigest[32];
    keccak256(finalDigest, finalBuf, 72);

    if (get_local_id(0) == 0)
        atomic_inc((volatile __global uint*)&g_output->hashCount);

    if (cmp256(finalDigest, target) > 0)
        return;

#ifdef FAST_EXIT
    atomic_inc((volatile __global uint*)&g_output->abort);
#endif

    uint indexOut = atomic_inc((volatile __global uint*)&g_output->count);
    if (indexOut >= MAX_OUTPUTS)
        return;

    g_output->rslt[indexOut].gid = gid;
    // Convert mixDigest to uint32 words
    for (int i = 0; i < 8; ++i)
    {
        uint w = ((const __private uint*)mixDigest)[i];
        g_output->rslt[indexOut].mix[i] = w;
    }
}
