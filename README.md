# GPU Performance Engineering

This repository collects the material for the interactive course *GPU Performance Engineering*.

## Prerequisites

Access to a system with a recent Nvidia GPU, as well as the Nvidia HPC SDK installed.

Profiling data will be obtained on that system.
The generated report files can then be visualized and analyzed locally.
This requires a local installation of Nsight Compute and Nsight Systems.
Both tools can either be installed separately ([nsight compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started), [nsight systems](https://developer.nvidia.com/nsight-systems/get-started), might require a free NVIDIA developer account), or bundled in the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) or [Nvidia HPC SDK](https://developer.nvidia.com/hpc-sdk-downloads).

A copy of all profiles obtained is available at [https://github.com/SebastianKuckuk/gpu-performance-engineering-profiles](https://github.com/SebastianKuckuk/gpu-performance-engineering-profiles).

## Course Content

All course material is collected and available at [https://github.com/SebastianKuckuk/gpu-performance-engineering](https://github.com/SebastianKuckuk/gpu-performance-engineering) (this repository) and [https://github.com/SebastianKuckuk/gpu-performance-engineering-profiles](https://github.com/SebastianKuckuk/gpu-performance-engineering-profiles).

It follows this general agenda:
1. [Introduction](./material/00-introduction.ipynb)
1. [Use Case: 2D Stencil](./material/01-stencil-use-case.ipynb)
1. [Performance Models](./material/02-performance-models.ipynb)
1. [Application Level Profiling](./material/03-application-level-profiling.ipynb)
1. [Kernel Level Profiling](./material/04-kernel-level-profiling.ipynb)
1. [GPU Architecture](./material/05-gpu-architecture.ipynb)
1. [Occupancy Optimization](./material/06-occupancy-optimization.ipynb)
1. [Micro Benchmarks](./material/07-micro-benchmarks.ipynb)
1. [Parallelism Optimization](./material/08-parallelism-optimization.ipynb)
1. [Additional Nsight Compute Options](./material/09-nsight-compute-options.ipynb)
1. [Nsight Compute GUI](./material/10-nsight-compute-gui.ipynb)
1. [Challenge: Conjugate Gradient](./material/11-conjugate-gradient.ipynb)

## Start

To start, clone the repository on your target system
```bash
git clone https://github.com/SebastianKuckuk/gpu-performance-engineering.git
```

To get a copy of the profiles (on your notebook/ workstation to visualize them locally) use
```bash
git clone https://github.com/SebastianKuckuk/gpu-performance-engineering-profiles.git
```
NOTE: Cloning the profiles repository requires [Git LFS](https://git-lfs.com/).
Downloading individual files via the web interface is possible regardless.

Then head over to the [Introduction](./material/00-introduction.ipynb) notebook.
