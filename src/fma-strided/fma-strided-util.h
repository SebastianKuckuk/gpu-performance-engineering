#include "../util.h"


template <typename tpe>
inline void initFmaStrided(tpe *__restrict__ data, size_t nx, size_t stride) {
    for (size_t i0 = 0; i0 < nx * stride; ++i0) {
        data[i0] = (tpe)1;
    }
}

template <typename tpe>
inline void checkSolutionFmaStrided(const tpe *const __restrict__ data, size_t nx, size_t nIt, size_t stride) {
    for (size_t i0 = 0; i0 < nx * stride; ++i0) {
        if ((tpe)((tpe)1) != data[i0]) {
            std::cerr << "FmaStrided check failed for element " << i0 << " (expected " << (tpe)1 << " but got " << data[i0] << ")" << std::endl;
            return;
        }
    }
}

inline void parseCLA_1d(int argc, char **argv, char *&tpeName, size_t &nx, size_t &stride, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 64;
    stride = 1;
    nItWarmUp = 0;
    nIt = 4;

    // override with command line arguments
    int i = 1;
    if (argc > i)
        tpeName = argv[i];
    ++i;
    if (argc > i)
        nx = atoi(argv[i]);
    ++i;
    if (argc > i)
        stride = atoi(argv[i]);
    ++i;
    if (argc > i)
        nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i)
        nIt = atoi(argv[i]);
    ++i;
}
