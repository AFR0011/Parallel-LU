# Parallel-LU: Distributed LU Factorization with Partial Pivoting

This project implements parallel LU factorization with partial pivoting for dense matrices, supporting both a custom MPI-lite networking layer (using TCP sockets) and a standard MPI implementation. It uses OpenMP for intra-node parallelism and distributes matrix rows across worker processes/machines.

## Overview

LU factorization decomposes a matrix \(A\) into lower triangular \(L\) and upper triangular \(U\) such that \(A = L \cdot U\). This implementation uses Gaussian elimination with partial pivoting to ensure numerical stability. The matrix is distributed in a 1D block-row fashion across processes.

Key features:
- **Matrix Generation Modes**: Deterministic generation with various diagonal dominance levels and near-singular options.
- **Parallelism**: MPI-lite (custom TCP-based) or standard MPI for inter-node communication; OpenMP for intra-node.
- **Verification**: Computes residual norms and checks for NaN/Inf values.
- **Timing and Diagnostics**: Detailed timing breakdowns, pivot statistics, and load imbalance metrics.
- **CSV Logging**: Optional output to CSV for benchmarking.

The project consists of multiple executables:
- `lu_driver.exe`: MPI-lite driver (orchestrator).
- `lu_driver1.exe`: Variant of the driver with serial baseline support.
- `lu_worker.exe`: MPI-lite worker process.
- `lu_mpi.exe`: Standard MPI implementation (single executable).

## Project Structure

- `lu_common.h` / `lu_common.cpp`: Shared utilities for matrix generation, distribution helpers, solving, and verification.
- `lu_net.h`: Networking utilities for MPI-lite (Winsock2-based TCP messaging).
- `lu_driver.cpp`: MPI-lite driver that connects to workers via TCP and orchestrates the factorization.
- `lu_driver1.cpp`: Similar to `lu_driver.cpp` but includes a serial baseline mode (runs locally if no hosts provided).
- `lu_worker.cpp`: Worker process that holds a block of rows and performs elimination steps.
- `lu_mpi.cpp`: Standalone MPI program using MS-MPI for distributed computation.

## Compilation

This project is designed for Windows using Visual Studio. It requires:
- Visual Studio 2019 or later (for C++17 support).
- MS-MPI (Microsoft MPI) if building `lu_mpi.cpp`.
- OpenMP support (enabled by default in VS).

### Build Steps

1. **Open in Visual Studio**:
   - Create a new Visual Studio project or use an existing one.
   - Add all `.cpp` and `.h` files to the project.

2. **Project Settings**:
   - **C/C++ > General > Additional Include Directories**: Ensure paths to headers are set (usually automatic).
   - **C/C++ > Language > C++ Language Standard**: Set to C++17 or later.
   - **C/C++ > Preprocessor > Preprocessor Definitions**: Add `_USE_MATH_DEFINES` if needed for math constants.
   - **Linker > Input > Additional Dependencies**:
     - For all: `Ws2_32.lib` (for Winsock2 in MPI-lite).
     - For `lu_mpi.cpp`: `msmpi.lib` (MS-MPI library).
   - **C/C++ > Language > OpenMP Support**: Set to "Yes" (`/openmp`).
   - **Linker > System > SubSystem**: Console for all executables.

3. **Build Configurations**:
   - Build separate executables for each driver/worker:
     - `lu_driver.exe`: Compile `lu_driver.cpp` + `lu_common.cpp` + `lu_net.h`.
     - `lu_driver1.exe`: Compile `lu_driver1.cpp` + `lu_common.cpp` + `lu_net.h`.
     - `lu_worker.exe`: Compile `lu_worker.cpp` + `lu_common.cpp` + `lu_net.h`.
     - `lu_mpi.exe`: Compile `lu_mpi.cpp` (includes all necessary code inline).

4. **MS-MPI Setup** (for `lu_mpi.exe`):
   - Install MS-MPI from Microsoft.
   - Ensure `msmpi.lib` is linked.
   - Include paths for MPI headers (e.g., `C:\Program Files (x86)\Microsoft SDKs\MPI\Include`).

5. **Build Command** (if using command line):
   - Use `cl.exe` (Visual Studio compiler):
     ```
     cl /std:c++17 /openmp /EHsc lu_driver.cpp lu_common.cpp /link Ws2_32.lib /out:lu_driver.exe
     ```
     - Repeat for each executable, adjusting sources.

## Running the Code

### Prerequisites
- Windows machines with the executables.
- For MPI-lite: Network connectivity between driver and workers.
- For MPI: MS-MPI installed and configured.

### Matrix Modes and Parameters
- **DiagDominant (dd)**: Default; \(A_{ii} += 2 \cdot \sum |A_{ij}| + N\).
- **WeakDiagDominant (weakdd)**: \(A_{ii} += \alpha \cdot \sum |A_{ij}| + \beta \cdot N\).
- **Random (rand)**: No diagonal boost.
- **NearSingular (near_singular)**: \(A = u \cdot v^T + \epsilon \cdot \text{noise} + \beta \cdot N \cdot I\).

Parameters:
- `--alpha A`: Diagonal boost factor (default varies by mode).
- `--beta B`: Diagonal shift (default varies).
- `--eps E`: Noise magnitude for near-singular (default 1e-3).
- `--seed S`: Random seed (default 123456789ULL).
- `--threads T`: OpenMP threads (default: system default; 0 means auto).
- `--timing`: Enable detailed timing output.
- `--verify`: Compute and print residual norms.
- `--csv PATH`: Append results to CSV file.

### MPI-Lite Version (lu_driver.exe + lu_worker.exe)

1. **Prepare Hosts File**:
   - Create `hosts.txt` with one "IP:port" per line (e.g., `192.168.1.100:5000`).
   - Empty lines and `#comments` are ignored.

2. **Start Workers**:
   - On each worker machine, run:
     ```
     lu_worker.exe --bind 0.0.0.0:5000 --threads T
     ```
     - Workers listen for connections and can handle multiple sessions.

3. **Run Driver**:
   - On the driver machine:
     ```
     lu_driver.exe N --hosts hosts.txt [options]
     ```
     - `N`: Matrix size (e.g., 2000).
     - Options: `--matrix dd`, `--alpha 0.5`, `--timing`, `--verify`, `--csv results.csv`.

   Example:
     ```
     lu_driver.exe 2000 --hosts hosts.txt --matrix weakdd --alpha 0.2 --timing --verify
     ```

4. **Shutdown**:
   - Use `--keep-workers 1` on driver to keep workers running for multiple runs.
   - Workers will accept new connections after a run.

### MPI-Lite with Serial Baseline (lu_driver1.exe)

- Similar to above, but if `hosts.txt` is empty or missing, it runs a serial LU factorization locally.
- Useful for single-machine testing or baselines.

### Standard MPI Version (lu_mpi.exe)

1. **Compile with MS-MPI**.
2. **Run with mpiexec**:
   ```
   mpiexec -n P lu_mpi.exe N [options]
   ```
   - `P`: Number of processes (MPI ranks).
   - `N`: Matrix size.

   Example:
     ```
     mpiexec -n 4 lu_mpi.exe 2000 --matrix rand --timing --verify --csv results.csv
     ```

## Output and Verification

- **Console Output**: Matrix info, factorization time, checksum, relative residual.
- **Timing Details** (if `--timing`): Breakdowns for pivot scan, swaps, broadcasts, elimination, etc.
- **Verification** (if `--verify`): Computes \(\|A x - b\|_\infty\), \(\|b\|_\infty\), relative residual.
- **CSV Output**: One line per run with all metrics.

Checksum is a weighted sum of the first 100 elements of \(x\) for quick validation.

## Examples

1. **Basic MPI-Lite Run**:
   - Workers: `lu_worker.exe --bind 0.0.0.0:5000`
   - Driver: `lu_driver.exe 1000 --hosts hosts.txt --timing`

2. **Serial Baseline**:
   - `lu_driver1.exe 1000 --timing --verify` (no hosts file)

3. **MPI Run**:
   - `mpiexec -n 2 lu_mpi.exe 1000 --matrix near_singular --eps 1e-4 --timing --verify`

4. **Benchmarking**:
   - `lu_driver.exe 2000 --hosts hosts.txt --matrix dd --timing --verify --csv bench.csv`

## Troubleshooting

- **Networking Issues**: Ensure ports are open, IPs are correct. Workers must be reachable from driver.
- **Compilation Errors**: Check VS settings for OpenMP and Winsock2.
- **MPI Errors**: Ensure MS-MPI is installed and paths are set.
- **Performance**: For large N, ensure sufficient memory (O(N^2) per process).
- **Verification Failures**: Check for NaN/Inf; may indicate numerical instability.

## License and Credits

This project is derived from existing LU factorization code, adapted for parallel execution. No specific license; use at your own risk.

For questions, refer to the code comments or contact the maintainer.