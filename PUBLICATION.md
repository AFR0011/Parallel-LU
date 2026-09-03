# Publication and evidence notes

Parallel-LU is published as a completed MSc systems-coursework artifact. Its
maintained claim is deliberately narrow: small synthetic serial and two-worker
MPI-lite runs build and pass automated correctness contracts on Windows.

## Supported public claims

- deterministic dense matrix generation;
- partial-pivot LU/Gaussian elimination;
- explicit serial and custom TCP driver/worker paths;
- block-row distribution and cross-worker row exchange;
- OpenMP local elimination;
- versioned, capped, timeout-bounded native-POD frames for a homogeneous trusted
  environment;
- non-finite and relative-residual pass/fail verification;
- warning-clean MSVC build and Windows CI;
- timing, communication counters, checksums, and optional generated CSV output.

## Claims not established

- universal or production-scale speedup;
- secure use on untrusted networks;
- portable interoperability across architectures or operating systems;
- worker-failure recovery or malicious-peer resistance;
- Linux/OpenMPI support;
- numerical equivalence to a trusted BLAS/LAPACK implementation across a broad
  condition-number range;
- successful build or execution of the optional `lu_mpi.cpp` MS-MPI path.

The MS-MPI source is implemented but unverified in the coursework release. The
archived duplicate coordinator is not a supported target.

## Benchmark reporting rule

Any future performance result must record the exact commit, compiler and flags,
matrix mode/size/seed, process and OpenMP thread counts, machine specifications,
network topology, repeated-run protocol, raw outputs, and aggregation method.
Until such evidence is reviewed, this repository does not claim a universal
speedup and publishes no benchmark plot.

## Release verification

The reviewed coursework release requires:

1. CMake configure and `/W4 /permissive- /WX` build;
2. common numerical/CLI/frame contract tests;
3. deterministic serial success and singular-case failure;
4. two-worker loopback agreement with cross-worker swaps;
5. publication-boundary and credential scans;
6. green public GitHub Actions for the release commit;
7. an exact-SHA fresh public clone rerun.
