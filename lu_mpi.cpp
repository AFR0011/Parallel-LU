// lu_detailed.cpp (modified)
// Distributed LU / Gaussian Elimination with partial pivoting
// - MPI across machines (distributed memory)
// - OpenMP inside each process (shared memory)
// - 1D block-row distribution
//
// Additions (Steps 1-6):
//  1) Matrix generation modes (dd/weakdd/rand/near_singular) with parameters
//  2) Pivot + swap diagnostics (pivot magnitudes, swap counts, pivot-owner histogram)
//  3) Critical-rank timing (additive breakdown) + load imbalance
//  4) Expanded comm timing (pivot scan vs MPI_Allreduce, MPI_Bcast, MPI_Sendrecv)
//  5) Stronger verification (residual + NaN/Inf sanity checks)
//  6) CSV logging (one line per run)
//
// Run examples (MS-MPI):
//   mpiexec -n 2 Parallel-LU-C++.exe 2000 --matrix dd --timing --verify
//   mpiexec -n 2 Parallel-LU-C++.exe 2000 --matrix rand --timing --verify
//   mpiexec -n 2 Parallel-LU-C++.exe 2000 --matrix weakdd --alpha 0.2 --beta 0.0 --timing --verify
//   mpiexec -n 2 Parallel-LU-C++.exe 2000 --matrix near_singular --eps 1e-3 --beta 1e-3 --timing --verify
//   mpiexec -n 2 Parallel-LU-C++.exe 2000 --matrix rand --timing --verify --csv results.csv

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// ---------------------- Options / flags ----------------------

enum class MatrixMode {
   DiagDominant,
   WeakDiagDominant,
   Random,
   NearSingular
};

static const char* mode_name(MatrixMode m) {
   switch (m) {
   case MatrixMode::DiagDominant:     return "dd";
   case MatrixMode::WeakDiagDominant: return "weakdd";
   case MatrixMode::Random:           return "rand";
   case MatrixMode::NearSingular:     return "near_singular";
   default:                           return "unknown";
   }
}

static bool parse_mode(const std::string& s, MatrixMode& out) {
   if (s == "dd" || s == "diag" || s == "diagd") { out = MatrixMode::DiagDominant; return true; }
   if (s == "weakdd" || s == "weak") { out = MatrixMode::WeakDiagDominant; return true; }
   if (s == "rand" || s == "random") { out = MatrixMode::Random; return true; }
   if (s == "near_singular" || s == "ns" || s == "nearsing") { out = MatrixMode::NearSingular; return true; }
   return false;
}

struct Options {
   int n = 0;
   uint64_t seed = 123456789ULL;

   MatrixMode mode = MatrixMode::DiagDominant;

   // Diagonal adjustment (dd/weakdd; and diagonal shift for near_singular):
   //   A_ii += alpha * sum_j |A_ij| + beta * N
   bool alpha_set = false;
   bool beta_set = false;
   double alpha = 0.0;
   double beta = 0.0;

   // Near-singular noise magnitude:
   bool eps_set = false;
   double eps = 1e-3;

   bool timing = false;
   bool verify = false;
   std::string csv_path; // if non-empty, append results to CSV
};

static void print_usage(const char* prog, int rank) {
   if (rank != 0) return;
   std::cerr
       << "Usage:\n"
       << "  " << prog << " N [seed] [--seed S] [--matrix MODE] [--alpha A] [--beta B] [--eps E]\n"
       << "         [--timing] [--verify] [--csv PATH] [--help]\n\n"
       << "Matrix modes:\n"
       << "  dd            : A_ii += 2*sum|row| + 1*N   (default)\n"
       << "  weakdd        : A_ii += alpha*sum|row| + beta*N (defaults alpha=0.2,beta=0)\n"
       << "  rand          : no diagonal boost\n"
       << "  near_singular : A = u*v^T + eps*noise, plus optional beta*N diagonal shift\n\n"
       << "Examples:\n"
       << "  " << prog << " 2000 --timing --verify\n"
       << "  " << prog << " 2000 --matrix rand --timing --verify\n"
       << "  " << prog << " 2000 --matrix weakdd --alpha 0.2 --beta 0.0 --timing --verify\n"
       << "  " << prog << " 2000 --matrix near_singular --eps 1e-3 --beta 1e-3 --timing --verify\n"
       << "  " << prog << " 2000 --matrix rand --timing --verify --csv results.csv\n";
}

static bool is_flag(const char* s) { return s && s[0] == '-'; }

static bool parse_options(int argc, char** argv, int rank, Options& opt) {
   if (argc < 2) { print_usage(argv[0], rank); return false; }

   opt.n = std::stoi(argv[1]);
   if (opt.n <= 0) {
       if (rank == 0) std::cerr << "Error: N must be > 0\n";
       return false;
   }

   int i = 2;
   // Backward-compatible positional seed: argv[2] if not a flag
   if (i < argc && !is_flag(argv[i])) {
       opt.seed = static_cast<uint64_t>(std::stoull(argv[i]));
       ++i;
   }

   for (; i < argc; ++i) {
       const char* a = argv[i];

       if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
           print_usage(argv[0], rank);
           return false;
       }
       else if (!std::strcmp(a, "--timing")) {
           opt.timing = true;
       }
       else if (!std::strcmp(a, "--verify")) {
           opt.verify = true;
       }
       else if (!std::strcmp(a, "--seed") || !std::strcmp(a, "-s")) {
           if (i + 1 >= argc) { if (rank == 0) std::cerr << "Error: --seed requires a value\n"; return false; }
           opt.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
       }
       else if (!std::strcmp(a, "--matrix") || !std::strcmp(a, "--mode")) {
           if (i + 1 >= argc) { if (rank == 0) std::cerr << "Error: --matrix requires a value\n"; return false; }
           MatrixMode m;
           if (!parse_mode(argv[++i], m)) {
               if (rank == 0) std::cerr << "Error: unknown matrix mode: " << argv[i] << "\n";
               return false;
           }
           opt.mode = m;
       }
       else if (!std::strcmp(a, "--alpha")) {
           if (i + 1 >= argc) { if (rank == 0) std::cerr << "Error: --alpha requires a value\n"; return false; }
           opt.alpha = std::stod(argv[++i]);
           opt.alpha_set = true;
       }
       else if (!std::strcmp(a, "--beta")) {
           if (i + 1 >= argc) { if (rank == 0) std::cerr << "Error: --beta requires a value\n"; return false; }
           opt.beta = std::stod(argv[++i]);
           opt.beta_set = true;
       }
       else if (!std::strcmp(a, "--eps")) {
           if (i + 1 >= argc) { if (rank == 0) std::cerr << "Error: --eps requires a value\n"; return false; }
           opt.eps = std::stod(argv[++i]);
           opt.eps_set = true;
       }
       else if (!std::strcmp(a, "--csv")) {
           if (i + 1 >= argc) { if (rank == 0) std::cerr << "Error: --csv requires a path\n"; return false; }
           opt.csv_path = argv[++i];
       }
       else {
           if (rank == 0) std::cerr << "Error: unknown argument: " << a << "\n";
           print_usage(argv[0], rank);
           return false;
       }
   }

   // Mode defaults unless user overrides.
   if (opt.mode == MatrixMode::DiagDominant) {
       if (!opt.alpha_set) opt.alpha = 2.0;
       if (!opt.beta_set)  opt.beta = 1.0;
   }
   else if (opt.mode == MatrixMode::WeakDiagDominant) {
       if (!opt.alpha_set) opt.alpha = 0.2;
       if (!opt.beta_set)  opt.beta = 0.0;
   }
   else if (opt.mode == MatrixMode::Random) {
       if (!opt.alpha_set) opt.alpha = 0.0;
       if (!opt.beta_set)  opt.beta = 0.0;
   }
   else if (opt.mode == MatrixMode::NearSingular) {
       if (!opt.alpha_set) opt.alpha = 0.0;
       if (!opt.beta_set)  opt.beta = 1e-3; // diagonal shift
       if (!opt.eps_set)   opt.eps = 1e-3;
   }

   return true;
}

// ---------------------- Deterministic pseudo-random ----------------------

static inline uint64_t splitmix64(uint64_t x) {
   x += 0x9e3779b97f4a7c15ULL;
   x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
   x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
   return x ^ (x >> 31);
}

static inline double u01_from_u64(uint64_t x) {
   const uint64_t mant = (x >> 11) | 1ULL;
   return (mant & ((1ULL << 53) - 1)) * (1.0 / (double)(1ULL << 53));
}

static inline double val_ij(uint64_t seed, int i, int j) {
   uint64_t x = seed;
   x ^= (uint64_t)(i + 1) * 0xD2B74407B1CE6E93ULL;
   x ^= (uint64_t)(j + 1) * 0xCA5A826395121157ULL;
   double u = u01_from_u64(splitmix64(x));
   return 2.0 * u - 1.0; // [-1,1)
}

// ---------------------- Matrix generator (Step 1) ----------------------

struct MatrixSpec {
   MatrixMode mode;
   double alpha; // diag boost via alpha*sumabs
   double beta;  // diag shift via beta*N
   double eps;   // near_singular noise
};

static inline double base_Aij(uint64_t seed, const MatrixSpec& ms, int i, int j) {
   if (ms.mode == MatrixMode::NearSingular) {
       const uint64_t su = seed ^ 0x9A73B52D1F6EB7C1ULL;
       const uint64_t sv = seed ^ 0xC3A5C85C97CB3127ULL;
       double u = val_ij(su, i, 0);
       double v = val_ij(sv, 0, j);
       double noise = val_ij(seed, i, j);
       return u * v + ms.eps * noise;
   }
   return val_ij(seed, i, j);
}

static inline double diag_adjust(int n, const MatrixSpec& ms, double row_sum_abs) {
   if (ms.mode == MatrixMode::DiagDominant || ms.mode == MatrixMode::WeakDiagDominant) {
       return ms.alpha * row_sum_abs + ms.beta * (double)n;
   }
   if (ms.mode == MatrixMode::NearSingular) {
       return ms.beta * (double)n; // diagonal shift
   }
   return 0.0;
}

static void generate_row(
   int n,
   uint64_t seed,
   const MatrixSpec& ms,
   int i,
   double* row_out,   // length n
   double& b_out,
   double& row_sum_abs_out
) {
   row_sum_abs_out = 0.0;
   for (int j = 0; j < n; ++j) {
       double v = base_Aij(seed, ms, i, j);
       row_out[j] = v;
       row_sum_abs_out += std::abs(v);
   }
   row_out[i] += diag_adjust(n, ms, row_sum_abs_out);

   const uint64_t bseed = seed ^ 0xABCDEF1234567890ULL;
   b_out = val_ij(bseed, i, 0) * 10.0;
}

// ---------------------- Block row distribution helpers ----------------------

struct Dist {
   int n;
   int rank, size;
   int row0;
   int rows;
   std::vector<int> counts;
   std::vector<int> displs;

   Dist(int n_, int rank_, int size_) : n(n_), rank(rank_), size(size_) {
       counts.resize(size);
       displs.resize(size);

       int base = n / size;
       int rem = n % size;
       int off = 0;
       for (int r = 0; r < size; ++r) {
           counts[r] = base + (r < rem ? 1 : 0);
           displs[r] = off;
           off += counts[r];
       }
       row0 = displs[rank];
       rows = counts[rank];
   }

   int owner_of_row(int global_i) const {
       for (int r = 0; r < size; ++r) {
           int start = displs[r];
           int end = start + counts[r];
           if (global_i >= start && global_i < end) return r;
       }
       return -1;
   }

   int local_index(int global_i) const { return global_i - row0; }
};

static inline double& Aat(std::vector<double>& A, int n, int li, int j) {
   return A[(size_t)li * (size_t)n + (size_t)j];
}
static inline const double& Aat(const std::vector<double>& A, int n, int li, int j) {
   return A[(size_t)li * (size_t)n + (size_t)j];
}

// ---------------------- Forward/Backward substitution on root ----------------------

static void solve_on_root_inplace_LU(
   int n,
   const std::vector<double>& LU,
   const std::vector<double>& b,
   std::vector<double>& x_out
) {
   x_out.assign(n, 0.0);
   std::vector<double> y(n, 0.0);

   for (int i = 0; i < n; ++i) {
       double sum = b[(size_t)i];
       const double* row = &LU[(size_t)i * (size_t)n];
       for (int j = 0; j < i; ++j) sum -= row[(size_t)j] * y[(size_t)j];
       y[(size_t)i] = sum;
   }

   for (int i = n - 1; i >= 0; --i) {
       const double* row = &LU[(size_t)i * (size_t)n];
       double sum = y[(size_t)i];
       for (int j = i + 1; j < n; ++j) sum -= row[(size_t)j] * x_out[(size_t)j];
       x_out[(size_t)i] = sum / row[(size_t)i];
   }
}

// ---------------------- Verification helpers (Step 5) ----------------------

static void compute_residual_inf_norms(
   int n,
   uint64_t seed,
   const MatrixSpec& ms,
   const std::vector<double>& x,
   double& resid_inf_out,
   double& b_inf_out,
   double& rel_resid_inf_out
) {
   double resid_inf = 0.0;
   double b_inf = 0.0;
   std::vector<double> row((size_t)n);

   for (int i = 0; i < n; ++i) {
       double bi = 0.0;
       double rowsum = 0.0;
       generate_row(n, seed, ms, i, row.data(), bi, rowsum);

       double dot = 0.0;
       for (int j = 0; j < n; ++j) dot += row[(size_t)j] * x[(size_t)j];

       double ri = dot - bi;
       resid_inf = std::max(resid_inf, std::abs(ri));
       b_inf = std::max(b_inf, std::abs(bi));
   }

   resid_inf_out = resid_inf;
   b_inf_out = b_inf;
   rel_resid_inf_out = (b_inf > 0.0) ? (resid_inf / b_inf) : resid_inf;
}

static void sanity_counts(const std::vector<double>& v, uint64_t& nan_count, uint64_t& inf_count) {
   nan_count = 0;
   inf_count = 0;
   for (double x : v) {
       if (std::isnan(x)) nan_count++;
       else if (!std::isfinite(x)) inf_count++;
   }
}

static std::string now_iso8601_utc() {
   using clock = std::chrono::system_clock;
   auto t = clock::to_time_t(clock::now());
   std::tm tm{};
#if defined(_WIN32)
   gmtime_s(&tm, &t);
#else
   gmtime_r(&t, &tm);
#endif
   char buf[32];
   std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
       tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
       tm.tm_hour, tm.tm_min, tm.tm_sec);
   return std::string(buf);
}

static bool file_is_empty_or_missing(const std::string& path) {
   std::ifstream in(path, std::ios::binary);
   if (!in) return true;
   in.seekg(0, std::ios::end);
   return in.tellg() <= 0;
}

static std::string csv_escape(const std::string& s) {
   bool need = false;
   for (char c : s) {
       if (c == ',' || c == '"' || c == '\n' || c == '\r') { need = true; break; }
   }
   if (!need) return s;
   std::string out = "\"";
   for (char c : s) out += (c == '"' ? std::string("\"\"") : std::string(1, c));
   out += "\"";
   return out;
}

// ---------------------- Main ----------------------

int main(int argc, char** argv) {
   MPI_Init(&argc, &argv);

   int rank = 0, size = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   Options opt;
   if (!parse_options(argc, argv, rank, opt)) {
       MPI_Finalize();
       return 1;
   }

   const int n = opt.n;
   const uint64_t seed = opt.seed;
   MatrixSpec ms{ opt.mode, opt.alpha, opt.beta, opt.eps };

   // Host names (MPI standard; works cross-platform)
   char pname[MPI_MAX_PROCESSOR_NAME];
   int pname_len = 0;
   MPI_Get_processor_name(pname, &pname_len);
   std::string my_host(pname, pname_len);

   // Gather hostnames to root
   std::vector<int> host_lens, host_displs;
   std::vector<char> host_buf;
   int my_len = (int)my_host.size();

   if (rank == 0) host_lens.resize((size_t)size, 0);
   MPI_Gather(&my_len, 1, MPI_INT, rank == 0 ? host_lens.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

   if (rank == 0) {
       host_displs.resize((size_t)size, 0);
       int total = 0;
       for (int r = 0; r < size; ++r) { host_displs[(size_t)r] = total; total += host_lens[(size_t)r]; }
       host_buf.resize((size_t)std::max(total, 1));
   }

   MPI_Gatherv(
       (void*)my_host.data(), my_len, MPI_CHAR,
       rank == 0 ? host_buf.data() : nullptr,
       rank == 0 ? host_lens.data() : nullptr,
       rank == 0 ? host_displs.data() : nullptr,
       MPI_CHAR,
       0, MPI_COMM_WORLD
   );

   std::vector<std::string> hosts;
   if (rank == 0) {
       hosts.resize((size_t)size);
       for (int r = 0; r < size; ++r) {
           hosts[(size_t)r] = std::string(&host_buf[(size_t)host_displs[(size_t)r]], (size_t)host_lens[(size_t)r]);
       }
   }

   Dist dist(n, rank, size);

   // Local storage
   std::vector<double> A((size_t)dist.rows * (size_t)n);
   std::vector<double> b_local((size_t)dist.rows);

   // Generate A and b deterministically (distributed)
   {
       std::vector<double> row((size_t)n);
       for (int li = 0; li < dist.rows; ++li) {
           int gi = dist.row0 + li;
           double bi = 0.0, rowsum = 0.0;
           generate_row(n, seed, ms, gi, row.data(), bi, rowsum);
           for (int j = 0; j < n; ++j) Aat(A, n, li, j) = row[(size_t)j];
           b_local[(size_t)li] = bi;
       }
   }

   // Timing buckets (local, additive per rank)
   double t_pivot_scan = 0.0;
   double t_allreduce = 0.0;
   double t_swap_local = 0.0;
   double t_swap_comm = 0.0; // sendrecv time
   double t_bcast = 0.0;
   double t_elim = 0.0;

   // Communication counters (Step 4)
   uint64_t allreduce_calls = 0, bcast_calls = 0, sendrecv_calls = 0;
   uint64_t bytes_bcast_sent = 0; // estimated bytes sent by bcast roots
   uint64_t bytes_swap_sent = 0; // bytes sent by swap participants (sum is total sent)

   // Diagnostics (Step 2): count once per k on rank 0
   double min_pivot = std::numeric_limits<double>::infinity();
   double max_pivot = 0.0;
   uint64_t small_pivots = 0; // pivots <= 1e-12 or non-finite

   uint64_t swaps_total = 0;
   uint64_t swaps_same_rank = 0;
   uint64_t swaps_cross_rank = 0;

   std::vector<uint64_t> pivot_owner_hist; // only meaningful on rank 0
   if (rank == 0) pivot_owner_hist.assign((size_t)size, 0);

   // Pivot row buffer
   std::vector<double> pivot_row((size_t)n);

   MPI_Barrier(MPI_COMM_WORLD);
   double t0 = MPI_Wtime();

   for (int k = 0; k < n; ++k) {
       // ---- Pivot scan (local) ----
       double scan0 = MPI_Wtime();

       double local_max = -1.0;
       int local_piv_gi = -1;

       int start_g = std::max(k, dist.row0);
       int end_g = dist.row0 + dist.rows;
       if (start_g < end_g) {
           int start_li = start_g - dist.row0;
           for (int li = start_li; li < dist.rows; ++li) {
               double v = std::abs(Aat(A, n, li, k));
               if (v > local_max) {
                   local_max = v;
                   local_piv_gi = dist.row0 + li;
               }
           }
       }

       double scan1 = MPI_Wtime();
       t_pivot_scan += (scan1 - scan0);

       // ---- Allreduce MAXLOC for global pivot ----
       double ar0 = MPI_Wtime();
       struct { double val; int idx; } in{ local_max, local_piv_gi }, out;
       MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
       double ar1 = MPI_Wtime();

       t_allreduce += (ar1 - ar0);
       allreduce_calls++;

       const double piv_abs = out.val;
       const int piv_gi = out.idx;

       if (piv_gi < 0 || piv_abs <= 0.0 || !std::isfinite(piv_abs)) {
           if (rank == 0) std::cerr << "Matrix appears singular or pivot failed at k=" << k << "\n";
           MPI_Abort(MPI_COMM_WORLD, 2);
       }

       int owner_p = dist.owner_of_row(piv_gi);
       int owner_k = dist.owner_of_row(k);

       if (rank == 0) {
           if (owner_p >= 0 && owner_p < size) pivot_owner_hist[(size_t)owner_p]++;
           if (piv_gi != k) {
               swaps_total++;
               if (owner_k == owner_p) swaps_same_rank++;
               else swaps_cross_rank++;
           }
       }

       // ---- Swap (if needed) ----
       if (piv_gi != k) {
           if (owner_k == owner_p) {
               double sl0 = MPI_Wtime();
               if (rank == owner_k) {
                   int lk = dist.local_index(k);
                   int lp = dist.local_index(piv_gi);
                   for (int j = 0; j < n; ++j) std::swap(Aat(A, n, lk, j), Aat(A, n, lp, j));
                   std::swap(b_local[(size_t)lk], b_local[(size_t)lp]);
               }
               double sl1 = MPI_Wtime();
               t_swap_local += (sl1 - sl0);
           }
           else {
               // Cross-rank swap via Sendrecv
               std::vector<double> rowbuf((size_t)n);
               double bbuf = 0.0;

               if (rank == owner_k || rank == owner_p) {
                   bytes_swap_sent += (uint64_t)n * (uint64_t)sizeof(double) + (uint64_t)sizeof(double);
               }

               double sc0 = MPI_Wtime();
               if (rank == owner_k) {
                   int lk = dist.local_index(k);
                   for (int j = 0; j < n; ++j) rowbuf[(size_t)j] = Aat(A, n, lk, j);
                   bbuf = b_local[(size_t)lk];

                   MPI_Sendrecv(rowbuf.data(), n, MPI_DOUBLE, owner_p, 100,
                       pivot_row.data(), n, MPI_DOUBLE, owner_p, 101,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                   MPI_Sendrecv(&bbuf, 1, MPI_DOUBLE, owner_p, 200,
                       &b_local[(size_t)lk], 1, MPI_DOUBLE, owner_p, 201,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                   for (int j = 0; j < n; ++j) Aat(A, n, lk, j) = pivot_row[(size_t)j];
                   sendrecv_calls += 2;
               }
               else if (rank == owner_p) {
                   int lp = dist.local_index(piv_gi);
                   for (int j = 0; j < n; ++j) pivot_row[(size_t)j] = Aat(A, n, lp, j);
                   double bp = b_local[(size_t)lp];

                   MPI_Sendrecv(pivot_row.data(), n, MPI_DOUBLE, owner_k, 101,
                       rowbuf.data(), n, MPI_DOUBLE, owner_k, 100,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                   MPI_Sendrecv(&bp, 1, MPI_DOUBLE, owner_k, 201,
                       &b_local[(size_t)lp], 1, MPI_DOUBLE, owner_k, 200,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                   for (int j = 0; j < n; ++j) Aat(A, n, lp, j) = rowbuf[(size_t)j];
                   sendrecv_calls += 2;
               }
               double sc1 = MPI_Wtime();
               t_swap_comm += (sc1 - sc0);
           }
       }

       // ---- Broadcast pivot row ----
       double bc0 = MPI_Wtime();
       if (rank == owner_k) {
           int lk = dist.local_index(k);
           for (int j = 0; j < n; ++j) pivot_row[(size_t)j] = Aat(A, n, lk, j);
           bytes_bcast_sent += (uint64_t)(size > 0 ? (size - 1) : 0) * (uint64_t)n * (uint64_t)sizeof(double);
       }
       MPI_Bcast(pivot_row.data(), n, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);
       double bc1 = MPI_Wtime();
       t_bcast += (bc1 - bc0);
       bcast_calls++;

       // Pivot diagnostics (rank 0 only, once per k)
       if (rank == 0) {
           double pivot = pivot_row[(size_t)k];
           double ap = std::abs(pivot);
           min_pivot = std::min(min_pivot, ap);
           max_pivot = std::max(max_pivot, ap);
           if (!(ap > 1e-12) || !std::isfinite(ap)) small_pivots++;
       }

       const double pivot = pivot_row[(size_t)k];
       if (std::abs(pivot) < 1e-300 || !std::isfinite(pivot)) {
           if (rank == 0) std::cerr << "Numerical pivot breakdown at k=" << k << "\n";
           MPI_Abort(MPI_COMM_WORLD, 3);
       }

       // ---- Elimination update ----
       double el0 = MPI_Wtime();

       const int local_start_g = std::max(k + 1, dist.row0);
       const int local_end_g = dist.row0 + dist.rows;

       if (local_start_g < local_end_g) {
           const int li0 = local_start_g - dist.row0;
#pragma omp parallel for schedule(static)
           for (int li = li0; li < dist.rows; ++li) {
               double aik = Aat(A, n, li, k) / pivot;
               Aat(A, n, li, k) = aik; // store L factor

               double* rowp = &A[(size_t)li * (size_t)n];
       for (int j = k + 1; j < n; ++j) {
                   rowp[(size_t)j] -= aik * pivot_row[(size_t)j];
               }
           }
       }

       double el1 = MPI_Wtime();
       t_elim += (el1 - el0);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   double t1 = MPI_Wtime();
   double t_total_fact = t1 - t0;

   // ---------------------- Timing interpretability (Step 3) ----------------------
   // Find critical rank (max total time) and use that rank's bucket times for additive breakdown.
   struct { double val; int idx; } tin{ t_total_fact, rank }, tout;
   MPI_Allreduce(&tin, &tout, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
   const int critical_rank = tout.idx;
   const double T_fact_max = tout.val;

   double sum_total = 0.0;
   MPI_Reduce(&t_total_fact, &sum_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   double avg_total = 0.0;
   if (rank == 0) avg_total = sum_total / (double)size;

   struct Buckets {
       double pivot_scan, allreduce, swap_local, swap_comm, bcast, elim, total;
   } bkt_local{ t_pivot_scan, t_allreduce, t_swap_local, t_swap_comm, t_bcast, t_elim, t_total_fact };

   std::vector<Buckets> all_buckets;
   if (rank == 0) all_buckets.resize((size_t)size);

   MPI_Gather(&bkt_local, (int)sizeof(Buckets), MPI_BYTE,
       rank == 0 ? all_buckets.data() : nullptr, (int)sizeof(Buckets), MPI_BYTE,
       0, MPI_COMM_WORLD);

   Buckets crit{};
   if (rank == 0 && critical_rank >= 0 && critical_rank < size) crit = all_buckets[(size_t)critical_rank];

   // ---------------------- Reduce comm counters (Step 4) ----------------------
   uint64_t allreduce_calls_sum = 0, bcast_calls_sum = 0, sendrecv_calls_sum = 0;
   MPI_Reduce(&allreduce_calls, &allreduce_calls_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&bcast_calls, &bcast_calls_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&sendrecv_calls, &sendrecv_calls_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

   uint64_t bytes_bcast_sent_sum = 0, bytes_swap_sent_sum = 0;
   MPI_Reduce(&bytes_bcast_sent, &bytes_bcast_sent_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&bytes_swap_sent, &bytes_swap_sent_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

   // ---------------------- Gather LU and b to root for solve (timed) ----------------------
   std::vector<int> recvcountsA(size), displsA(size);
   std::vector<int> recvcountsB(size), displsB(size);
   for (int r = 0; r < size; ++r) {
       recvcountsB[r] = dist.counts[r];
       displsB[r] = dist.displs[r];
       recvcountsA[r] = dist.counts[r] * n;
       displsA[r] = dist.displs[r] * n;
   }

   std::vector<double> LU_full, b_full;
   if (rank == 0) {
       LU_full.resize((size_t)n * (size_t)n);
       b_full.resize((size_t)n);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   double tg0 = MPI_Wtime();

   MPI_Gatherv(A.data(), (int)(dist.rows * n), MPI_DOUBLE,
       rank == 0 ? LU_full.data() : nullptr,
       recvcountsA.data(), displsA.data(), MPI_DOUBLE,
       0, MPI_COMM_WORLD);

   MPI_Gatherv(b_local.data(), dist.rows, MPI_DOUBLE,
       rank == 0 ? b_full.data() : nullptr,
       recvcountsB.data(), displsB.data(), MPI_DOUBLE,
       0, MPI_COMM_WORLD);

   MPI_Barrier(MPI_COMM_WORLD);
   double tg1 = MPI_Wtime();
   double t_gather_local = tg1 - tg0;
   double t_gather_max = 0.0;
   MPI_Reduce(&t_gather_local, &t_gather_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

   // Solve time on root
   double t_solve_local = 0.0;
   std::vector<double> x;
   if (rank == 0) {
       double ts0 = MPI_Wtime();
       solve_on_root_inplace_LU(n, LU_full, b_full, x);
       double ts1 = MPI_Wtime();
       t_solve_local = ts1 - ts0;
   }

   double t_solve_max = 0.0;
   MPI_Reduce(&t_solve_local, &t_solve_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

   // ---------------------- Verification (Step 5) ----------------------
   double resid_inf = std::numeric_limits<double>::quiet_NaN();
   double b_inf = std::numeric_limits<double>::quiet_NaN();
   double resid_rel = std::numeric_limits<double>::quiet_NaN();
   uint64_t lu_nan = 0, lu_inf = 0, x_nan = 0, x_inf = 0;

   if (rank == 0 && opt.verify) {
       compute_residual_inf_norms(n, seed, ms, x, resid_inf, b_inf, resid_rel);
       sanity_counts(LU_full, lu_nan, lu_inf);
       sanity_counts(x, x_nan, x_inf);
   }

   // ---------------------- Explainability output (root) ----------------------
   if (rank == 0) {
       std::cout << "N=" << n
           << "  ranks=" << size
           << "  omp_threads=" << omp_get_max_threads()
           << "  seed=" << seed
           << "  matrix=" << mode_name(ms.mode)
           << "  alpha=" << std::setprecision(6) << ms.alpha
           << "  beta=" << std::setprecision(6) << ms.beta
           << "  eps=" << std::setprecision(6) << ms.eps
           << "\n";

       std::cout << "Rank->host mapping:\n";
       for (int r = 0; r < size; ++r) {
           std::cout << "  rank " << r << ": " << hosts[(size_t)r] << "\n";
       }
       bool all_same_host = true;
       for (int r = 1; r < size; ++r) if (hosts[(size_t)r] != hosts[0]) { all_same_host = false; break; }
       if (all_same_host && size > 1) {
           std::cout << "WARNING: all ranks are on the same host. This is MPI parallelism, not multi-computer distribution.\n";
       }

       std::cout << "System solved: A x = b (dense), A and b generated deterministically from seed.\n";
       if (ms.mode == MatrixMode::DiagDominant || ms.mode == MatrixMode::WeakDiagDominant) {
           std::cout << "Generation: base A_ij in [-1,1), then A_ii += alpha*sum_j|A_ij| + beta*N.\n";
       }
       else if (ms.mode == MatrixMode::Random) {
           std::cout << "Generation: A_ij in [-1,1) with no diagonal boosting.\n";
       }
       else if (ms.mode == MatrixMode::NearSingular) {
           std::cout << "Generation: A = u*v^T + eps*noise, plus diagonal shift beta*N.\n";
       }
       std::cout << "Algorithm: distributed LU/GE with partial pivoting; row-block distribution; "
           "pivot via MPI_Allreduce(MAXLOC), pivot-row MPI_Bcast, OpenMP elimination.\n";

       std::cout << "LU factorization wall time (critical rank): " << std::setprecision(6) << T_fact_max << " s\n";
       std::cout << "Load imbalance max/avg: " << std::setprecision(6)
           << (avg_total > 0.0 ? (T_fact_max / avg_total) : 0.0)
           << " (avg=" << std::setprecision(6) << avg_total << " s)\n";

       double checksum = 0.0;
       for (int i = 0; i < std::min(n, 100); ++i) checksum += x[(size_t)i] * (i + 1);
       std::cout << "x checksum (first 100 weighted): " << std::setprecision(12) << checksum << "\n";

       std::cout << "\nPivot diagnostics:\n";
       std::cout << "  min |pivot|: " << std::setprecision(12) << min_pivot << "\n";
       std::cout << "  max |pivot|: " << std::setprecision(12) << max_pivot << "\n";
       std::cout << "  # pivots <= 1e-12 or non-finite: " << small_pivots << "\n";

       std::cout << "\nSwap diagnostics:\n";
       std::cout << "  swaps total (piv_gi != k): " << swaps_total << "\n";
       std::cout << "  swaps within same rank:    " << swaps_same_rank << "\n";
       std::cout << "  swaps cross-rank:          " << swaps_cross_rank << "\n";

       std::cout << "\nPivot owner histogram (which rank owned selected pivot rows):\n";
       for (int r = 0; r < size; ++r) std::cout << "  rank " << r << ": " << pivot_owner_hist[(size_t)r] << "\n";

       if (opt.timing) {
           auto pct = [&](double t) -> double { return (crit.total > 0.0) ? (100.0 * t / crit.total) : 0.0; };

           std::cout << "\nTiming breakdown (critical rank " << critical_rank << ", additive, seconds):\n";
           std::cout << "  pivot scan:     " << std::setprecision(6) << crit.pivot_scan
               << "  (" << std::setprecision(3) << pct(crit.pivot_scan) << "%)\n";
           std::cout << "  allreduce:      " << std::setprecision(6) << crit.allreduce
               << "  (" << std::setprecision(3) << pct(crit.allreduce) << "%)\n";
           std::cout << "  swap local:     " << std::setprecision(6) << crit.swap_local
               << "  (" << std::setprecision(3) << pct(crit.swap_local) << "%)\n";
           std::cout << "  swap comm:      " << std::setprecision(6) << crit.swap_comm
               << "  (" << std::setprecision(3) << pct(crit.swap_comm) << "%)\n";
           std::cout << "  bcast:          " << std::setprecision(6) << crit.bcast
               << "  (" << std::setprecision(3) << pct(crit.bcast) << "%)\n";
           std::cout << "  elimination:    " << std::setprecision(6) << crit.elim
               << "  (" << std::setprecision(3) << pct(crit.elim) << "%)\n";
           std::cout << "  total:          " << std::setprecision(6) << crit.total << "\n";

           std::cout << "\nPost-factorization:\n";
           std::cout << "  gather (LU,b) max: " << std::setprecision(6) << t_gather_max << " s\n";
           std::cout << "  solve (root)     : " << std::setprecision(6) << t_solve_max << " s\n";

           std::cout << "\nCommunication counters (estimates):\n";
           std::cout << "  allreduce calls (sum across ranks): " << allreduce_calls_sum << "\n";
           std::cout << "  bcast calls (sum across ranks):     " << bcast_calls_sum << "\n";
           std::cout << "  sendrecv calls (sum across ranks):  " << sendrecv_calls_sum << "\n";
           std::cout << "  bcast bytes sent (sum of roots):    " << bytes_bcast_sent_sum << " bytes\n";
           std::cout << "  swap bytes sent (sum participants): " << bytes_swap_sent_sum << " bytes\n";
       }

       if (opt.verify) {
           std::cout << "\nVerification:\n";
           std::cout << "  ||Ax-b||_inf: " << std::setprecision(12) << resid_inf << "\n";
           std::cout << "  ||b||_inf:    " << std::setprecision(12) << b_inf << "\n";
           std::cout << "  rel residual: " << std::setprecision(12) << resid_rel << "\n";
           std::cout << "  sanity: LU nan=" << lu_nan << " inf=" << lu_inf
               << ", x nan=" << x_nan << " inf=" << x_inf << "\n";
       }

       // ---------------------- CSV logging (Step 6) ----------------------
       if (!opt.csv_path.empty()) {
           bool need_header = file_is_empty_or_missing(opt.csv_path);
           std::ofstream out(opt.csv_path, std::ios::app);
           if (!out) {
               std::cerr << "ERROR: could not open CSV for append: " << opt.csv_path << "\n";
           }
           else {
               if (need_header) {
                   out << "timestamp_utc,hosts,N,ranks,omp_threads,seed,matrix,alpha,beta,eps,"
                       << "T_fact_max,T_fact_avg,imbalance,"
                       << "T_pivot_scan,T_allreduce,T_swap_local,T_swap_comm,T_bcast,T_elim,"
                       << "T_gather_max,T_solve_root,"
                       << "min_pivot,max_pivot,small_pivots,swaps_total,swaps_same_rank,swaps_cross_rank,"
                       << "bytes_bcast_sent,bytes_swap_sent,allreduce_calls,bcast_calls,sendrecv_calls,"
                       << "resid_inf,resid_rel,lu_nan,lu_inf,x_nan,x_inf,critical_rank\n";
               }

               std::ostringstream host_join;
               for (int r = 0; r < size; ++r) {
                   if (r) host_join << "|";
                   host_join << hosts[(size_t)r];
               }

               out << csv_escape(now_iso8601_utc()) << ","
                   << csv_escape(host_join.str()) << ","
                   << n << ","
                   << size << ","
                   << omp_get_max_threads() << ","
                   << seed << ","
                   << mode_name(ms.mode) << ","
                   << std::setprecision(17) << ms.alpha << ","
                   << std::setprecision(17) << ms.beta << ","
                   << std::setprecision(17) << ms.eps << ","
                   << std::setprecision(17) << T_fact_max << ","
                   << std::setprecision(17) << avg_total << ","
                   << std::setprecision(17) << (avg_total > 0.0 ? (T_fact_max / avg_total) : 0.0) << ","
                   << std::setprecision(17) << crit.pivot_scan << ","
                   << std::setprecision(17) << crit.allreduce << ","
                   << std::setprecision(17) << crit.swap_local << ","
                   << std::setprecision(17) << crit.swap_comm << ","
                   << std::setprecision(17) << crit.bcast << ","
                   << std::setprecision(17) << crit.elim << ","
                   << std::setprecision(17) << t_gather_max << ","
                   << std::setprecision(17) << t_solve_max << ","
                   << std::setprecision(17) << min_pivot << ","
                   << std::setprecision(17) << max_pivot << ","
                   << small_pivots << ","
                   << swaps_total << ","
                   << swaps_same_rank << ","
                   << swaps_cross_rank << ","
                   << bytes_bcast_sent_sum << ","
                   << bytes_swap_sent_sum << ","
                   << allreduce_calls_sum << ","
                   << bcast_calls_sum << ","
                   << sendrecv_calls_sum << ","
                   << std::setprecision(17) << (opt.verify ? resid_inf : std::numeric_limits<double>::quiet_NaN()) << ","
                   << std::setprecision(17) << (opt.verify ? resid_rel : std::numeric_limits<double>::quiet_NaN()) << ","
                   << (opt.verify ? lu_nan : 0) << ","
                   << (opt.verify ? lu_inf : 0) << ","
                   << (opt.verify ? x_nan : 0) << ","
                   << (opt.verify ? x_inf : 0) << ","
                   << critical_rank
                   << "\n";
           }
       }
   }

   MPI_Finalize();
   return 0;
}
