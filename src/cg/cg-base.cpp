#include "cg-util.h"

#include <nvtx3/nvtx3.hpp>


template <typename tpe>
inline size_t conjugateGradient(const tpe *const __restrict__ rhs, tpe *__restrict__ u, tpe *__restrict__ res,
                                tpe *__restrict__ p, tpe *__restrict__ ap,
                                const size_t nx, const size_t ny, const size_t maxIt) {

    NVTX3_FUNC_RANGE();  // range around the whole function body

    nvtxRangePushA("initialization");

    // initialization
    tpe initResSq = (tpe)0;

    // compute initial residual
    for (size_t j = 1; j < ny - 1; ++j) {
        for (size_t i = 1; i < nx - 1; ++i) {
            res[j * nx + i] = rhs[j * nx + i] - (4 * u[j * nx + i] - (u[j * nx + i - 1] + u[j * nx + i + 1] + u[(j - 1) * nx + i] + u[(j + 1) * nx + i]));
        }
    }

    // compute residual norm
    for (size_t j = 1; j < ny - 1; ++j) {
        for (size_t i = 1; i < nx - 1; ++i) {
            initResSq += res[j * nx + i] * res[j * nx + i];
        }
    }

    // init p
    for (size_t j = 1; j < ny - 1; ++j) {
        for (size_t i = 1; i < nx - 1; ++i) {
            p[j * nx + i] = res[j * nx + i];
        }
    }

    tpe curResSq = initResSq;

    nvtxRangePop(); // "initialization"

    // main loop
    for (size_t it = 0; it < maxIt; ++it) {
        nvtx3::scoped_range loop{"main loop"};

        nvtxRangePushA("Ap");
        // compute A * p
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                ap[j * nx + i] = 4 * p[j * nx + i] - (p[j * nx + i - 1] + p[j * nx + i + 1] + p[(j - 1) * nx + i] + p[(j + 1) * nx + i]);
            }
        }
        nvtxRangePop();

        nvtxRangePushA("alpha");
        tpe alphaNominator = curResSq;
        tpe alphaDenominator = (tpe)0;
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                alphaDenominator += p[j * nx + i] * ap[j * nx + i];
            }
        }
        tpe alpha = alphaNominator / alphaDenominator;
        nvtxRangePop();

        // update solution
        nvtxRangePushA("solution");
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                u[j * nx + i] += alpha * p[j * nx + i];
            }
        }
        nvtxRangePop();

        // update residual
        nvtxRangePushA("residual");
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                res[j * nx + i] -= alpha * ap[j * nx + i];
            }
        }
        nvtxRangePop();

        // compute residual norm
        nvtxRangePushA("resNorm");
        tpe nextResSq = (tpe)0;
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                nextResSq += res[j * nx + i] * res[j * nx + i];
            }
        }
        nvtxRangePop();
        
        // check exit criterion
        if (sqrt(nextResSq) <= 1e-12)
            return it;

        // if (0 == it % 100)
        //     std::cout << "    " << it << " : " << sqrt(nextResSq) << std::endl;

        // compute beta
        nvtxRangePushA("beta");
        tpe beta = nextResSq / curResSq;
        curResSq = nextResSq;
        nvtxRangePop();

        // update p
        nvtxRangePushA("p");
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                p[j * nx + i] = res[j * nx + i] + beta * p[j * nx + i];
            }
        }
        nvtxRangePop();
    }

    return maxIt;
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt);

    tpe *u;
    u = new tpe[nx * ny];
    tpe *rhs;
    rhs = new tpe[nx * ny];

    tpe *res;
    res = new tpe[nx * ny];
    tpe *p;
    p = new tpe[nx * ny];
    tpe *ap;
    ap = new tpe[nx * ny];

    // init
    initConjugateGradient(u, rhs, nx, ny);

    memset(res, 0, nx * ny * sizeof(tpe));
    memset(p, 0, nx * ny * sizeof(tpe));
    memset(ap, 0, nx * ny * sizeof(tpe));

    // warm-up
    nItWarmUp = conjugateGradient(rhs, u, res, p, ap, nx, ny, nItWarmUp);

    // measurement
    auto start = std::chrono::steady_clock::now();

    nIt = conjugateGradient(rhs, u, res, p, ap, nx, ny, nIt);
    std::cout << "  CG steps:      " << nIt << std::endl;

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx * ny, tpeName, 8 * sizeof(tpe), 15);

    // check solution
    checkSolutionConjugateGradient(u, rhs, nx, ny);

    delete[] u;
    delete[] rhs;

    delete[] res;
    delete[] p;
    delete[] ap;

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
