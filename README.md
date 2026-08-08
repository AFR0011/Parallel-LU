# Parallel-LU

> **MSc Computer Engineering course project.** This repository is kept public as supporting coursework demonstrating parallel/distributed-computing concepts in C++ rather than as a flagship or production project.

Distributed LU factorization with partial pivoting, implemented in C++17 with two inter-process communication paths:

- a custom TCP-based MPI-lite driver/worker layer; and
- a standard MS-MPI implementation.

OpenMP is used for intra-process parallelism. Dense matrices are distributed by block rows, with pivot selection, row exchange, elimination, verification, timing, and optional benchmark logging built into the executables.

## What the project explores

- distributed dense-matrix decomposition;
- partial pivoting and row-swap coordination;
- custom TCP message transport using Winsock2;
- standard MPI collective/process communication;
- OpenMP intra-node parallelism;
- deterministic matrix generation for repeatable experiments;
- residual-based correctness checks and NaN/Inf detection;
- timing, pivot, and load-imbalance diagnostics;
- CSV benchmark output;
- serial and distributed execution paths.

## Programs

| Program | Purpose |
| --- | --- |
| `lu_driver.cpp` | MPI-lite coordinator for TCP workers |
| `lu_driver1.cpp` | MPI-lite coordinator with serial fallback/baseline |
| `lu_worker.cpp` | MPI-lite worker process |
| `lu_mpi.cpp` | Standard MPI implementation |
| `lu_common.cpp/.h` | Shared matrix generation, distribution, solving, and verification helpers |
| `lu_net.h` | TCP/Winsock2 networking helpers |

## Computation model

For a dense matrix `A`, the implementation performs Gaussian elimination with partial pivoting to obtain an LU factorization while distributing matrix rows across processes.

```text
Matrix generation
      |
      v
1D block-row distribution
      |
      v
Pivot scan / selection
      |
      +--> row exchange
      +--> pivot-row communication
      |
      v
Parallel elimination
      |
      v
Solve / verification
      |
      +--> residual checks
      +--> timing diagnostics
      +--> optional CSV output
```

The custom MPI-lite path uses a driver/worker topology over TCP. The standard MPI path uses MS-MPI and is launched through `mpiexec`.

## Matrix modes

The executables support deterministic generation modes for correctness and performance experiments:

- `dd`: diagonally dominant;
- `weakdd`: weaker configurable diagonal dominance;
- `rand`: random dense matrix without a diagonal boost;
- `near_singular`: near-singular construction with configurable noise.

Common parameters include:

```text
--alpha A
--beta B
--eps E
--seed S
--threads T
--timing
--verify
--csv PATH
```

## Requirements

The current implementation is Windows-oriented and uses:

- Visual Studio 2019 or later with C++17 support;
- OpenMP;
- Winsock2 for the MPI-lite transport;
- Microsoft MPI for `lu_mpi.cpp`.

## Build

### MPI-lite driver

From a Visual Studio developer command prompt:

```bat
cl /std:c++17 /openmp /EHsc lu_driver.cpp lu_common.cpp /link Ws2_32.lib /out:lu_driver.exe
```

### MPI-lite driver with serial baseline

```bat
cl /std:c++17 /openmp /EHsc lu_driver1.cpp lu_common.cpp /link Ws2_32.lib /out:lu_driver1.exe
```

### Worker

```bat
cl /std:c++17 /openmp /EHsc lu_worker.cpp lu_common.cpp /link Ws2_32.lib /out:lu_worker.exe
```

### Standard MPI

Compile `lu_mpi.cpp` with the MS-MPI include/library paths configured and link `msmpi.lib`.

## Run the MPI-lite implementation

Create a host file containing one worker endpoint per line:

```text
192.168.1.100:5000
192.168.1.101:5000
```

Start a worker on each machine:

```bat
lu_worker.exe --bind 0.0.0.0:5000 --threads 4
```

Then run the driver:

```bat
lu_driver.exe 2000 --hosts hosts.txt --matrix weakdd --alpha 0.2 --timing --verify
```

`lu_driver1.exe` can run locally as a serial baseline when no worker hosts are provided.

## Run the MPI implementation

```bat
mpiexec -n 4 lu_mpi.exe 2000 --matrix rand --timing --verify --csv results.csv
```

## Verification and diagnostics

With `--verify`, the program reports residual information including `||Ax-b||∞`, `||b||∞`, and a relative residual. The implementation also checks for NaN/Inf values.

With `--timing`, execution reports timing breakdowns for major phases such as pivot scanning, row swaps, communication, and elimination. `--csv` appends run data for later benchmark analysis.

A weighted checksum of the first solved values is also emitted as a lightweight run-to-run validation signal.

## Current boundaries

- This is coursework rather than a production or maintained HPC library.
- The codebase is Windows/MS-MPI oriented rather than cross-platform.
- Dense storage and factorization remain `O(N²)` in memory footprint per relevant process allocation.
- Benchmark results depend heavily on matrix size, process layout, network conditions, and OpenMP configuration, so this repository does not claim a universal speedup.
- Correctness verification is implemented in the executables; a broader automated cross-platform CI/benchmark harness would be a useful future extension.

## License

Parallel-LU is released under the MIT License. See [`LICENSE`](LICENSE).
