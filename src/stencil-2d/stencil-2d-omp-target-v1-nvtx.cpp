#include "stencil-2d-util.h"

#include <nvtx3/nvtx3.hpp>


template <typename tpe>
inline void stencil2d(const tpe *const __restrict__ u, tpe *__restrict__ uNew, size_t nx, size_t ny) {
    NVTX3_FUNC_RANGE();

#pragma omp \
    target \
    teams distribute parallel for \
    map(tofrom : u [0:nx * ny], uNew[0:nx * ny])
    for (size_t i1 = 1; i1 < ny - 1; ++i1) {
        for (size_t i0 = 1; i0 < nx - 1; ++i0) {
            uNew[i0 + i1 * nx] = 0.25 * u[i0 + i1 * nx + 1] + 0.25 * u[i0 + i1 * nx - 1] + 0.25 * u[i0 + nx * (i1 + 1)] + 0.25 * u[i0 + nx * (i1 - 1)];
        }
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt);

    tpe *u;
    u = new tpe[nx * ny];
    tpe *uNew;
    uNew = new tpe[nx * ny];

    // init
    initStencil2D(u, uNew, nx, ny);

    nvtxRangePushA("enter data");
#pragma omp target enter data map(to : u [0:nx * ny], uNew [0:nx * ny])
    nvtxRangePop(); // enter data

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil2d(u, uNew, nx, ny);
        std::swap(u, uNew);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    nvtxRangePushA("timed iterations");
    for (size_t i = 0; i < nIt; ++i) {
        nvtx3::scoped_range sr("iteration" + std::to_string(i));
        stencil2d(u, uNew, nx, ny);
        std::swap(u, uNew);
    }
    nvtxRangePop(); // timed iterations

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx * ny, tpeName, sizeof(tpe) + sizeof(tpe), 7);

    nvtxRangePushA("exit data");
#pragma omp target exit data map(from : u [0:nx * ny], uNew [0:nx * ny])
    nvtxRangePop(); // exit data

    // check solution
    checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp);

    delete[] u;
    delete[] uNew;

    return 0;
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Missing type specification" << std::endl;
        return -1;
    }

    std::string tpeName(argv[1]);

    if ("float" == tpeName)
        return realMain<float>(argc, argv);
    if ("double" == tpeName)
        return realMain<double>(argc, argv);

    std::cout << "Invalid type specification (" << argv[1] << "); supported types are" << std::endl;
    std::cout << "  float, double" << std::endl;
    return -1;
}
