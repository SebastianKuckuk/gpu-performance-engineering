#include "stream-strided-util.h"

#include "../cuda-util.h"


template <typename tpe>
__global__ void streamstrided(const tpe *const __restrict__ src, tpe *__restrict__ dest, size_t nx, size_t strideRead, size_t strideWrite) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
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
    checkCudaError(cudaMallocHost((void **)&dest, sizeof(tpe) * nx * std::max(strideRead, strideWrite)));
    tpe *src;
    checkCudaError(cudaMallocHost((void **)&src, sizeof(tpe) * nx * std::max(strideRead, strideWrite)));

    tpe *d_dest;
    checkCudaError(cudaMalloc((void **)&d_dest, sizeof(tpe) * nx * std::max(strideRead, strideWrite)));
    tpe *d_src;
    checkCudaError(cudaMalloc((void **)&d_src, sizeof(tpe) * nx * std::max(strideRead, strideWrite)));

    // init
    initStreamStrided(dest, src, nx, strideRead, strideWrite);

    checkCudaError(cudaMemcpy(d_dest, dest, sizeof(tpe) * nx * std::max(strideRead, strideWrite), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_src, src, sizeof(tpe) * nx * std::max(strideRead, strideWrite), cudaMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        streamstrided<<<ceilingDivide(nx, 256), 256>>>(d_src, d_dest, nx, strideRead, strideWrite);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        streamstrided<<<ceilingDivide(nx, 256), 256>>>(d_src, d_dest, nx, strideRead, strideWrite);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 1);

    checkCudaError(cudaMemcpy(dest, d_dest, sizeof(tpe) * nx * std::max(strideRead, strideWrite), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(src, d_src, sizeof(tpe) * nx * std::max(strideRead, strideWrite), cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionStreamStrided(dest, src, nx, nIt + nItWarmUp, strideRead, strideWrite);

    checkCudaError(cudaFree(d_dest));
    checkCudaError(cudaFree(d_src));

    checkCudaError(cudaFreeHost(dest));
    checkCudaError(cudaFreeHost(src));

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
