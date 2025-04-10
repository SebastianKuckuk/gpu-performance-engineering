# GPU Performance Engineering

This repository collects the material for the interactive course *GPU Performance Engineering*.

## Prerequisites

Access to a system with a recent Nvidia GPU, as well as the Nvidia HPC SDK installed.

Profiling data will be obtained on that system.
The generated report files can then be visualized and analyzed locally.
This requires a local installation of Nsight Compute and Nsight Systems.
Both tools can either be installed separately ([nsight compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started), [nsight systems](https://developer.nvidia.com/nsight-systems/get-started), might require a free NVIDIA developer account), or bundled in the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) or [Nvidia HPC SDK](https://developer.nvidia.com/hpc-sdk-downloads).

A copy of all profiles obtained is also included in this repository.

## Course Content

All course material is collected and available at [https://github.com/SebastianKuckuk/gpu-performance-engineering](https://github.com/SebastianKuckuk/gpu-performance-engineering) (this repository).

It follows this general agenda:
1. [Introduction](./material/introduction.ipynb)
1. [Test Case: 2D Stencil](./material/stencil-test-case.ipynb)
1. [GPU Architecture](./material/gpu-architecture.ipynb)
1. [Performance Models](./material/performance-models.ipynb)
1. [Micro Benchmarks](./material/micro-benchmarks.ipynb)
1. [Application Level Profiling](./material/application-level-profiling.ipynb)
1. [Kernel Level Profiling](./material/kernel-level-profiling.ipynb)
1. [Challenge: Conjugate Gradient](./material/conjugate-gradient.ipynb)

## Start

To start, clone the repository on your target system (and on your notebook/ workstation to visualize the profiles locally)
```bash
git clone https://github.com/SebastianKuckuk/gpu-performance-engineering.git
```

Then head over to the [Introduction](./material/introduction.ipynb) notebook.
