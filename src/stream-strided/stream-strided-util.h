#include "../util.h"


template <typename tpe>
inline void initStreamStrided(tpe *__restrict__ dest, tpe *__restrict__ src, size_t nx, size_t strideRead, size_t strideWrite) {
    for (size_t i0 = 0; i0 < nx * std::max(strideRead, strideWrite); ++i0) {
        src[i0] = (tpe)0;
        dest[i0] = (tpe)0;
    }
}

template <typename tpe>
inline void checkSolutionStreamStrided(const tpe *const __restrict__ dest, const tpe *const __restrict__ src, size_t nx, size_t nIt, size_t strideRead, size_t strideWrite) {
    tpe total = 0;
    for (size_t i0 = 0; i0 < nx * std::max(strideRead, strideWrite); ++i0) {
        total += src[i0];
    }

    if (total <= 0 || total > nx * nIt)
        std::cerr << "StreamStrided check failed "
                  << " (expected value between 0+ and " << nx * nIt << " but got " << total << ")" << std::endl;
}

inline void parseCLA_1d(int argc, char **argv, char *&tpeName, size_t &nx, size_t &strideRead, size_t &strideWrite, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 67108864;
    strideRead = 1;
    strideWrite = 1;
    nItWarmUp = 2;
    nIt = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i)
        tpeName = argv[i];
    ++i;
    if (argc > i)
        nx = atoi(argv[i]);
    ++i;
    if (argc > i)
        strideRead = atoi(argv[i]);
    ++i;
    if (argc > i)
        strideWrite = atoi(argv[i]);
    ++i;
    if (argc > i)
        nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i)
        nIt = atoi(argv[i]);
    ++i;
}
