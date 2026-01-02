/*
 * Build-time test: verifies CUDA target architectures baked into the binary.
 * Does not run on GPU; catches the case where only one arch (e.g. sm_90) is left
 * and older cards have no SASS/PTX available.
 */

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_arches.h>

namespace
{
std::vector<int> parse_arches(const std::string& s)
{
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        if (tok.empty())
            continue;
        std::stringstream val(tok);
        int arch = 0;
        val >> arch;
        if (!val.fail())
            out.push_back(arch);
    }
    return out;
}
}  // namespace

int main()
{
    const std::string archStr = ETHMINER_CUDA_ARCH_LIST;
    if (archStr.empty())
    {
        std::cerr << "arch list is empty" << std::endl;
        return 1;
    }

    std::vector<int> arches = parse_arches(archStr);
    if (arches.empty())
    {
        std::cerr << "arch list did not parse: " << archStr << std::endl;
        return 2;
    }

    std::set<int> uniq(arches.begin(), arches.end());
    if (uniq.size() != arches.size())
    {
        std::cerr << "arch list has duplicates: " << archStr << std::endl;
        return 3;
    }

    auto has = [&](int a) { return uniq.find(a) != uniq.end(); };
    if (!has(50))
    {
        std::cerr << "baseline sm_50 missing, got: " << archStr << std::endl;
        return 4;
    }

    const int maxArch = ETHMINER_CUDA_MAX_ARCH;
    if (maxArch <= 0 || !has(maxArch))
    {
        std::cerr << "max arch invalid/missing: " << maxArch << " list: " << archStr << std::endl;
        return 5;
    }

    const int computedMax = *std::max_element(arches.begin(), arches.end());
    if (computedMax != maxArch)
    {
        std::cerr << "max arch mismatch: computed " << computedMax << " vs macro " << maxArch
                  << " list: " << archStr << std::endl;
        return 6;
    }

    std::cout << "CUDA arches: " << archStr << " (max " << maxArch << ")" << std::endl;
    return 0;
}
