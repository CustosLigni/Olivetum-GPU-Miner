/*
    Olivetumhash auxiliary helpers for CPU-side hashing.

    This code implements the Olivetumhash PoW (ethash-like) so the miner can
    validate and CPU-mine blocks without relying on GPU kernels.
*/

#pragma once

#include <libdevcore/Common.h>
#include <libdevcore/FixedHash.h>
#include <vector>
#include <memory>

namespace dev
{
namespace eth
{

struct OlivetumhashResult
{
    h256 value;
    h256 mixHash;
};

class OlivetumhashAux
{
public:
    static OlivetumhashResult eval(int epoch, h256 const& headerHash, uint64_t nonce) noexcept;
    static std::shared_ptr<std::vector<uint8_t>> datasetShared(int epoch);

private:
    static std::vector<uint8_t> const& dataset(int epoch);
};

}  // namespace eth
}  // namespace dev
