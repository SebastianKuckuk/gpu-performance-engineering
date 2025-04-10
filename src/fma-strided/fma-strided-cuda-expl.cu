#include "fma-strided-util.h"

#include "../cuda-util.h"


template <typename tpe>
__global__ void fmastrided(tpe *__restrict__ data, size_t nx, size_t stride) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx * stride) {
        tpe a = (tpe)0.5, b = (tpe)1;
        // dummy op to prevent compiler from solving loop analytically
        if (1 == nx) {
            auto tmp = b;
            b = a;
            a = tmp;
        }

        tpe acc = i0;

        if (0 == i0 % stride) {
            acc = data[i0];
            for (auto r = 0; r < 65536; ++r)
                acc = a * acc + b;
        }

        // dummy check to prevent compiler from eliminating loop
        if ((tpe)0 == acc)
            data[i0 / stride] = acc;
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    size_t stride;
    parseCLA_1d(argc, argv, tpeName, nx, stride, nItWarmUp, nIt);

    tpe *data;
    checkCudaError(cudaMallocHost((void **)&data, sizeof(tpe) * nx * stride));

    tpe *d_data;
    checkCudaError(cudaMalloc((void **)&d_data, sizeof(tpe) * nx * stride));

    // init
    initFmaStrided(data, nx, stride);

    checkCudaError(cudaMemcpy(d_data, data, sizeof(tpe) * nx * stride, cudaMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        fmastrided<<<ceilingDivide(nx * stride, 256), 256>>>(d_data, nx, stride);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        fmastrided<<<ceilingDivide(nx * stride, 256), 256>>>(d_data, nx, stride);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe), 131072);

    checkCudaError(cudaMemcpy(data, d_data, sizeof(tpe) * nx * stride, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionFmaStrided(data, nx, nIt + nItWarmUp, stride);

    checkCudaError(cudaFree(d_data));

    checkCudaError(cudaFreeHost(data));

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
