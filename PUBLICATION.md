# Publication Notes

Parallel-LU is released as a systems/HPC engineering project under the MIT License.

The repository currently demonstrates a Windows-oriented C++17 implementation of distributed LU factorization with partial pivoting using a custom TCP-based MPI-lite path, standard MS-MPI, and OpenMP.

## Current scope

- custom driver/worker TCP communication;
- standard MPI execution;
- OpenMP intra-process parallelism;
- block-row distribution;
- partial pivoting and row exchange;
- deterministic matrix generation;
- residual and NaN/Inf verification;
- timing and load diagnostics;
- optional CSV benchmark output.

## Presentation guidance

Benchmark results should always include the matrix size, process count, thread count, communication mode, and hardware/network context. The project does not claim a universal parallel speedup.

A future portability pass could add CMake, Linux/OpenMPI support, automated correctness tests, and reproducible benchmark plots. Those are useful extensions rather than prerequisites for understanding the current implementation.
