#include "stream-strided-util.h"


template <typename tpe>
inline void streamstrided(const tpe *const __restrict__ src, tpe *__restrict__ dest, size_t nx, size_t strideRead, size_t strideWrite) {
#pragma omp target teams distribute parallel for
    for (size_t i0 = 0; i0 < nx; ++i0) {
        dest[i0 * strideWrite] = src[i0 * strideRead] + 1;
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    size_t strideRead;
    size_t strideWrite;
    parseCLA_1d(argc, argv, tpeName, nx, strideRead, strideWrite, nItWarmUp, nIt);

    tpe *dest;
    dest = new tpe[nx * std::max(strideRead, strideWrite)];
    tpe *src;
    src = new tpe[nx * std::max(strideRead, strideWrite)];

    // init
    initStreamStrided(dest, src, nx, strideRead, strideWrite);

#pragma omp target enter data map(to \
                                  : dest [0:nx * std::max(strideRead, strideWrite)])
#pragma omp target enter data map(to \
                                  : src [0:nx * std::max(strideRead, strideWrite)])

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        streamstrided(src, dest, nx, strideRead, strideWrite);
        std::swap(src, dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        streamstrided(src, dest, nx, strideRead, strideWrite);
        std::swap(src, dest);
    }

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 1);

#pragma omp target exit data map(from \
                                 : dest [0:nx * std::max(strideRead, strideWrite)])
#pragma omp target exit data map(from \
                                 : src [0:nx * std::max(strideRead, strideWrite)])

    // check solution
    checkSolutionStreamStrided(dest, src, nx, nIt + nItWarmUp, strideRead, strideWrite);

    delete[] dest;
    delete[] src;

    return 0;
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Missing type specification" << std::endl;
        return -1;
    }

    std::string tpeName(argv[1]);

    if ("int" == tpeName)
        return realMain<int>(argc, argv);
    if ("long" == tpeName)
        return realMain<long>(argc, argv);
    if ("float" == tpeName)
        return realMain<float>(argc, argv);
    if ("double" == tpeName)
        return realMain<double>(argc, argv);

    std::cout << "Invalid type specification (" << argv[1] << "); supported types are" << std::endl;
    std::cout << "  int, long, float, double" << std::endl;
    return -1;
}
