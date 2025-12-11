#include <iostream>
#include <vector>

#include <libdevcore/FixedHash.h>
#include <libethcore/OlivetumhashAux.h>

using namespace dev;
using namespace dev::eth;

namespace
{

void printResult(const std::string& label, const OlivetumhashResult& r)
{
    std::cout << label << " mixDigest=" << r.mixHash.hex() << " finalDigest=" << r.value.hex()
              << std::endl;
}

}  // namespace

int main()
{
    // Simple vector set: header + a few nonces for epoch 0.
    const h256 header("0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef");
    const int epoch = 0;

    std::vector<uint64_t> nonces = {0x0ull, 0x1ull, 0xabcdefull, 0xdeadbeefull};
    for (size_t i = 0; i < nonces.size(); ++i)
    {
        auto r = OlivetumhashAux::eval(epoch, header, nonces[i]);
        printResult("nonce 0x" + toHex(nonces[i]), r);
    }

    return 0;
}
