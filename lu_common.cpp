// lu_common.cpp
#include "lu_common.h"

#include <sstream>
#include <cstring>
#include <iostream>

// ---------------------- Mode helpers ----------------------

const char* mode_name(MatrixMode m) {
    switch (m) {
    case MatrixMode::DiagDominant:     return "dd";
    case MatrixMode::WeakDiagDominant: return "weakdd";
    case MatrixMode::Random:           return "rand";
    case MatrixMode::NearSingular:     return "near_singular";
    default:                           return "unknown";
    }
}

bool parse_mode(const std::string& s, MatrixMode& out) {
    if (s == "dd" || s == "diag" || s == "diagd") { out = MatrixMode::DiagDominant; return true; }
    if (s == "weakdd" || s == "weak") { out = MatrixMode::WeakDiagDominant; return true; }
    if (s == "rand" || s == "random") { out = MatrixMode::Random; return true; }
    if (s == "near_singular" || s == "ns" || s == "nearsing") { out = MatrixMode::NearSingular; return true; }
    return false;
}

// ---------------------- Deterministic RNG helpers ----------------------

uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double u01_from_u64(uint64_t u) {
    // map to [0,1)
    const uint64_t mant = (u >> 11) | 0x3FF0000000000000ULL;
    double d;
    std::memcpy(&d, &mant, sizeof(double));
    return d - 1.0;
}

double val_ij(uint64_t seed, int i, int j) {
    uint64_t x = seed;
    x ^= (uint64_t)(i + 1) * 0xD2B74407B1CE6E93ULL;
    x ^= (uint64_t)(j + 1) * 0xCA5A826395121157ULL;
    double u = u01_from_u64(splitmix64(x));
    return 2.0 * u - 1.0; // [-1,1)
}

// ---------------------- Matrix generator ----------------------

double base_Aij(uint64_t seed, const MatrixSpec& ms, int i, int j) {
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

double diag_adjust(int n, const MatrixSpec& ms, double row_sum_abs) {
    if (ms.mode == MatrixMode::DiagDominant || ms.mode == MatrixMode::WeakDiagDominant) {
        return ms.alpha * row_sum_abs + ms.beta * (double)n;
    }
    if (ms.mode == MatrixMode::NearSingular) {
        return ms.beta * (double)n; // diagonal shift
    }
    return 0.0;
}

void generate_row(
    int n,
    uint64_t seed,
    const MatrixSpec& ms,
    int i,
    double* row_out,
    double& b_out,
    double& row_sum_abs_out
) {
    row_sum_abs_out = 0.0;
    for (int j = 0; j < n; ++j) {
        double v = base_Aij(seed, ms, i, j);
        row_out[j] = v;
        row_sum_abs_out += std::abs(v);
    }

    // diagonal dominance/shift tweaks
    double diag_add = diag_adjust(n, ms, row_sum_abs_out);
    row_out[i] += diag_add;

    // b_i = f(seed^C, i) * 10  (as in lu_final.cpp)
    const uint64_t sb = seed ^ 0x4F1BBCDCB7A2B3D5ULL;
    b_out = val_ij(sb, i, 0) * 10.0;
}

void compute_residual_inf_norms(
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
        double row_sum_abs = 0.0;
        generate_row(n, seed, ms, i, row.data(), bi, row_sum_abs);

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

// ---------------------- Dist ----------------------

Dist::Dist(int n_, int rank_, int size_) : n(n_), rank(rank_), size(size_) {
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

    rows = counts[rank];
    row0 = displs[rank];
}

int Dist::owner_of_row(int global_i) const {
    for (int r = 0; r < size; ++r) {
        int start = displs[r];
        int end = start + counts[r];
        if (global_i >= start && global_i < end) return r;
    }
    return -1;
}

// ---------------------- Solve (LU already contains the swaps applied by factorization) ----------------------

void solve_on_root_inplace_LU(
    int n,
    const std::vector<double>& LU,
    const std::vector<double>& b,
    std::vector<double>& x_out
) {
    x_out.assign((size_t)n, 0.0);
    std::vector<double> y((size_t)n, 0.0);

    // Forward substitution Ly = b (L has 1s on diag; lower part of LU stores L)
    for (int i = 0; i < n; ++i) {
        double sum = b[(size_t)i];
        const double* row = &LU[(size_t)i * (size_t)n];
        for (int j = 0; j < i; ++j) sum -= row[(size_t)j] * y[(size_t)j];
        y[(size_t)i] = sum; // diag is 1
    }

    // Back substitution Ux = y
    for (int i = n - 1; i >= 0; --i) {
        const double* row = &LU[(size_t)i * (size_t)n];
        double sum = y[(size_t)i];
        for (int j = i + 1; j < n; ++j) sum -= row[(size_t)j] * x_out[(size_t)j];
        double diag = row[(size_t)i];
        if (diag == 0.0 || !std::isfinite(diag)) {
            throw std::runtime_error(
                "Invalid diagonal during back substitution at row " +
                std::to_string(i)
            );
        }
        x_out[(size_t)i] = sum / diag;
    }
}

double checksum_weighted_first100(const std::vector<double>& x) {
    const int m = (int)std::min<size_t>(100, x.size());
    double s = 0.0;
    for (int i = 0; i < m; ++i) s += (double)(i + 1) * x[(size_t)i];
    return s;
}

size_t count_non_finite(const std::vector<double>& values) {
    return (size_t)std::count_if(
        values.begin(), values.end(), [](double value) { return !std::isfinite(value); }
    );
}

// ---------------------- Arg parsing ----------------------

static bool is_flag(const char* s) {
    return s && s[0] == '-' && s[1] == '-';
}

bool parse_driver_args(int argc, char** argv, RunOptions& opt, std::string& err) {
    if (argc < 2) {
        err = "Usage: lu_driver.exe N (--serial | --hosts hosts.txt) [seed] "
              "[--matrix dd|weakdd|rand|near_singular] [--alpha a] [--beta b] "
              "[--eps e] [--threads T] [--timing] [--verify] "
              "[--residual-tol R] [--csv path]";
        return false;
    }

    try {
        opt.n = std::stoi(argv[1]);
    } catch (...) {
        err = "Error: N must be an integer.";
        return false;
    }
    if (opt.n <= 0 || opt.n > kMaxMatrixN) {
        err = "Error: N must be between 1 and " + std::to_string(kMaxMatrixN) + ".";
        return false;
    }

    int i = 2;
    try {
        if (i < argc && !is_flag(argv[i])) {
            opt.seed = (uint64_t)std::stoull(argv[i]);
            ++i;
        }
    } catch (const std::exception&) {
        err = "Error: seed must be an unsigned integer.";
        return false;
    }

    try {
        for (; i < argc; ++i) {
            std::string a = argv[i];
            auto need = [&](const char* name) -> const char* {
                if (i + 1 >= argc) {
                    err = std::string("Error: missing value for ") + name;
                    return nullptr;
                }
                return argv[++i];
            };

            if (a == "--serial") {
                opt.serial = true;
            } else if (a == "--hosts") {
                const char* v = need("--hosts"); if (!v) return false;
                opt.hosts_path = v;
            } else if (a == "--matrix") {
                const char* v = need("--matrix"); if (!v) return false;
                MatrixMode m;
                if (!parse_mode(v, m)) { err = "Error: bad --matrix value."; return false; }
                opt.mode = m;
            } else if (a == "--alpha") {
                const char* v = need("--alpha"); if (!v) return false;
                opt.alpha = std::stod(v); opt.alpha_set = true;
            } else if (a == "--beta") {
                const char* v = need("--beta"); if (!v) return false;
                opt.beta = std::stod(v); opt.beta_set = true;
            } else if (a == "--eps") {
                const char* v = need("--eps"); if (!v) return false;
                opt.eps = std::stod(v); opt.eps_set = true;
            } else if (a == "--threads") {
                const char* v = need("--threads"); if (!v) return false;
                opt.threads = std::stoi(v);
            } else if (a == "--timing") {
                opt.timing = true;
            } else if (a == "--verify") {
                opt.verify = true;
            } else if (a == "--residual-tol") {
                const char* v = need("--residual-tol"); if (!v) return false;
                opt.residual_tol = std::stod(v);
            } else if (a == "--csv") {
                const char* v = need("--csv"); if (!v) return false;
                opt.csv_path = v;
            } else if (a == "--seed") {
                const char* v = need("--seed"); if (!v) return false;
                opt.seed = (uint64_t)std::stoull(v);
            } else if (a == "--keep-workers") {
                opt.keep_workers = 1;
            } else if (a == "--help" || a == "-h") {
                err = "Usage: lu_driver.exe N (--serial | --hosts hosts.txt) [options]";
                return false;
            } else {
                err = "Error: unknown option: " + a;
                return false;
            }
        }
    } catch (const std::exception&) {
        err = "Error: invalid numeric option value.";
        return false;
    }

    // defaults aligned with lu_final.cpp
    if (!opt.alpha_set) {
        if (opt.mode == MatrixMode::DiagDominant) opt.alpha = 2.0;
        else if (opt.mode == MatrixMode::WeakDiagDominant) opt.alpha = 0.2;
        else opt.alpha = 0.0;
    }
    if (!opt.beta_set) {
        if (opt.mode == MatrixMode::DiagDominant) opt.beta = 1.0;
        else opt.beta = 0.0;
    }
    if (!opt.eps_set) opt.eps = 1e-3;

    if (!std::isfinite(opt.alpha) || !std::isfinite(opt.beta) ||
        !std::isfinite(opt.eps) || opt.eps < 0.0) {
        err = "Error: alpha/beta must be finite and eps must be finite and >= 0.";
        return false;
    }
    if (opt.threads < 0 || opt.threads > kMaxThreads) {
        err = "Error: --threads must be between 0 and " + std::to_string(kMaxThreads) + ".";
        return false;
    }
    if (!std::isfinite(opt.residual_tol) || opt.residual_tol <= 0.0) {
        err = "Error: --residual-tol must be finite and > 0.";
        return false;
    }
    if (opt.serial && !opt.hosts_path.empty()) {
        err = "Error: choose either --serial or --hosts, not both.";
        return false;
    }
    if (!opt.serial && opt.hosts_path.empty()) {
        err = "Error: choose --serial or provide --hosts for lu_driver.exe.";
        return false;
    }
    return true;
}

bool parse_worker_args(int argc, char** argv, WorkerOptions& opt, std::string& err) {
    // Usage: lu_worker.exe [--listen ip:port] [--threads T]
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto need = [&](const char* name) -> const char* {
                if (i + 1 >= argc) { err = std::string("Error: missing value for ") + name; return nullptr; }
                return argv[++i];
            };

            if (a == "--listen") {
                const char* v = need("--listen"); if (!v) return false;
                std::string s = v;
                auto pos = s.find(':');
                if (pos == std::string::npos || pos == 0 || pos != s.rfind(':')) {
                    err = "Error: --listen must be hostname-or-IPv4:port";
                    return false;
                }
                opt.bind_ip = s.substr(0, pos);
                size_t consumed = 0;
                int port = std::stoi(s.substr(pos + 1), &consumed);
                if (consumed != s.size() - pos - 1 || port < 1 || port > 65535) {
                    err = "Error: --listen requires a non-empty IPv4/hostname and port 1-65535";
                    return false;
                }
                opt.port = (uint16_t)port;
            } else if (a == "--threads") {
                const char* v = need("--threads"); if (!v) return false;
                opt.threads = std::stoi(v);
            } else if (a == "--help" || a == "-h") {
                err = "Usage: lu_worker.exe [--listen ip:port] [--threads T]";
                return false;
            } else {
                err = "Error: unknown option: " + a;
                return false;
            }
        }
    } catch (const std::exception&) {
        err = "Error: invalid numeric worker option value.";
        return false;
    }
    if (opt.threads < 0 || opt.threads > kMaxThreads) {
        err = "Error: --threads must be between 0 and " + std::to_string(kMaxThreads) + ".";
        return false;
    }
    return true;
}
