// HISTORICAL MPI-LITE DRIVER VARIANT - NOT BUILT OR SUPPORTED.
//
// This duplicate coordinator is preserved for project history. It drifted from
// the canonical lu_driver.cpp and does not implement that driver's serial path.
// Use the root-level lu_driver.cpp and lu_worker.cpp targets instead.
//
// lu_driver.cpp - MPI-lite driver/orchestrator
//
// Usage:
//   lu_driver.exe N --hosts hosts.txt [seed] [--matrix ...] [--alpha a] [--beta b] [--eps e]
//                 [--threads T] [--timing] [--verify] [--csv out.csv]
//
// hosts.txt: one "IP:port" (or hostname:port) per line, empty lines and #comments allowed.
//
// Run model in lab:
//   - Start lu_worker.exe on each machine (same port).
//   - Run lu_driver.exe on the main machine.

#include "lu_common.h"
#include "lu_net.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <iomanip>
#include <chrono>
#include <omp.h>

struct WorkerConn {
    std::string host;
    uint16_t port = 0;
    SOCKET s = INVALID_SOCKET;
};

static std::vector<std::string> read_hosts_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Failed to open hosts file: " + path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) {
        // strip comments
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        // trim
        auto is_ws = [](unsigned char c) { return std::isspace(c) != 0; };
        while (!line.empty() && is_ws((unsigned char)line.front())) line.erase(line.begin());
        while (!line.empty() && is_ws((unsigned char)line.back())) line.pop_back();
        if (line.empty()) continue;
        out.push_back(line);
    }
    return out;
}

static void expect_ok(SOCKET s) {
    net::Msg t; std::vector<uint8_t> payload;
    net::recv_msg(s, t, payload);
    if (t == net::Msg::OK) return;
    if (t == net::Msg::ERR) {
        net::ByteReader r(payload);
        uint32_t len = r.pod<uint32_t>();
        std::string msg(len, '\0');
        if (len) r.bytes(msg.data(), len);
        throw std::runtime_error("Worker ERR: " + msg);
    }
    throw std::runtime_error("Unexpected reply type from worker.");
}

static std::vector<uint8_t> rpc(SOCKET s, net::Msg req_type, const std::vector<uint8_t>& req_payload, net::Msg expected_reply) {
    net::send_msg(s, req_type, req_payload);
    net::Msg rt; std::vector<uint8_t> rp;
    net::recv_msg(s, rt, rp);
    if (rt == expected_reply) return rp;
    if (rt == net::Msg::ERR) {
        net::ByteReader r(rp);
        uint32_t len = r.pod<uint32_t>();
        std::string msg(len, '\0');
        if (len) r.bytes(msg.data(), len);
        throw std::runtime_error("Worker ERR: " + msg);
    }
    throw std::runtime_error("Unexpected reply type from worker.");
}

static MatrixSpec build_spec(const RunOptions& opt) {
    MatrixSpec ms;
    ms.mode = opt.mode;
    ms.alpha = opt.alpha;
    ms.beta = opt.beta;
    ms.eps = opt.eps;
    return ms;
}

static void csv_append_line(const std::string& path, const std::string& line) {
    if (path.empty()) return;
    std::ofstream f(path, std::ios::app);
    if (!f) throw std::runtime_error("Failed to open CSV for append: " + path);
    f << line << "\n";
}

static bool csv_exists(const std::string& path) {
    if (path.empty()) return false;
    std::ifstream f(path);
    return (bool)f;
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);

    RunOptions opt;
    std::string err;
    if (!parse_driver_args(argc, argv, opt, err)) {
        std::cerr << err << "\n";
        return 2;
    }

    const int n = opt.n;
    const MatrixSpec ms = build_spec(opt);

    try {
        net::WSAInit wsa;

        // Load hosts and connect
        auto hosts = read_hosts_file(opt.hosts_path);
        if (hosts.size() < 1) {
            throw std::runtime_error("hosts.txt must contain at least 1 worker.");
        }
        const int p = (int)hosts.size();

        std::vector<WorkerConn> conns((size_t)p);
        for (int r = 0; r < p; ++r) {
            auto hp = net::parse_hostport(hosts[(size_t)r]);
            if (hp.second == 0) throw std::runtime_error("Host entry must be host:port, got: " + hosts[(size_t)r]);
            conns[(size_t)r].host = hp.first;
            conns[(size_t)r].port = hp.second;
            conns[(size_t)r].s = net::connect_tcp(hp.first, hp.second);
        }

        // Distribution (same as original Dist)
        std::vector<int> counts((size_t)p), displs((size_t)p);
        {
            int base = n / p;
            int rem = n % p;
            int off = 0;
            for (int r = 0; r < p; ++r) {
                counts[(size_t)r] = base + (r < rem ? 1 : 0);
                displs[(size_t)r] = off;
                off += counts[(size_t)r];
            }
        }
        auto owner_of_row = [&](int gi) -> int {
            for (int r = 0; r < p; ++r) {
                int start = displs[(size_t)r];
                int end = start + counts[(size_t)r];
                if (gi >= start && gi < end) return r;
            }
            return -1;
            };

        // INIT all workers
        for (int r = 0; r < p; ++r) {
            net::ByteWriter w;
            w.pod((int32_t)n);
            w.pod((int32_t)r);
            w.pod((int32_t)p);
            w.pod((int32_t)displs[(size_t)r]); // row0
            w.pod((int32_t)counts[(size_t)r]); // rows
            w.pod((uint64_t)opt.seed);
            w.pod((int32_t)opt.mode);
            w.pod((double)ms.alpha);
            w.pod((double)ms.beta);
            w.pod((double)ms.eps);
            w.pod((int32_t)opt.threads);
            net::send_msg(conns[(size_t)r].s, net::Msg::INIT, w.buf);
        }
        for (int r = 0; r < p; ++r) expect_ok(conns[(size_t)r].s);

        std::cout << "N=" << n << " workers=" << p << " threads=" << (opt.threads > 0 ? opt.threads : 0)
            << " seed=" << opt.seed
            << " matrix=" << mode_name(opt.mode)
            << " alpha=" << ms.alpha
            << " beta=" << ms.beta
            << " eps=" << ms.eps
            << "\n";
        std::cout << "System: A x = b (dense), generated deterministically from seed.\n";
        if (opt.mode == MatrixMode::Random) {
            std::cout << "A_ij in [-1,1), no diagonal boost (Random mode).\n";
        }
        else if (opt.mode == MatrixMode::NearSingular) {
            std::cout << "A_ij in [-1,1), near-singular structure with eps noise; diagonal shift uses beta*N.\n";
        }
        else {
            std::cout << "A_ij in [-1,1), diagonal adjusted as: A_ii += alpha*sum_j|A_ij| + beta*N.\n";
        }

        // CSV header if needed
        if (!opt.csv_path.empty() && !csv_exists(opt.csv_path)) {
            csv_append_line(opt.csv_path,
                "N,p,threads,seed,matrix,alpha,beta,eps,"
                "T_total,T_pivot,T_swap,T_getpivot,T_bcast_send,T_elim,T_gather,T_solve,"
                "swap_total,swap_cross,swap_bytes,bcast_bytes_sent,"
                "rel_resid,checksum,critical_fact");
        }

        // Timing (driver-side)
        double t_pivot = 0.0;
        double t_swap = 0.0;
        double t_getpivot = 0.0;   // GET_PIVOT_TAIL RPC time
        double t_bcast_send = 0.0; // send+wait for pivot tail broadcast
        double t_elim = 0.0;
        double t_gather = 0.0;
        double t_solve = 0.0;

        // Extra counters
        uint64_t swap_total = 0;
        uint64_t swap_cross = 0;
        uint64_t swap_bytes = 0;            // estimated bytes moved due to cross-worker swaps
        uint64_t bcast_bytes_sent = 0;      // estimated driver->workers bytes for pivot tail broadcasts

        double t0_all = omp_get_wtime();

        // Temp buffers for pivot tail broadcast
        std::vector<double> tail((size_t)n);

        for (int k = 0; k < n; ++k) {
            // Pivot scan (gather local maxima)
            double t0 = omp_get_wtime();
            for (int r = 0; r < p; ++r) {
                net::ByteWriter w; w.pod((int32_t)k);
                net::send_msg(conns[(size_t)r].s, net::Msg::PIVOT_SCAN, w.buf);
            }
            double best = 0.0;
            int best_gi = -1;

            for (int r = 0; r < p; ++r) {
                net::Msg rt; std::vector<uint8_t> rp;
                net::recv_msg(conns[(size_t)r].s, rt, rp);
                if (rt == net::Msg::ERR) {
                    net::ByteReader rr(rp);
                    uint32_t len = rr.pod<uint32_t>();
                    std::string msg(len, '\0');
                    if (len) rr.bytes(msg.data(), len);
                    throw std::runtime_error("Worker ERR during pivot scan: " + msg);
                }
                if (rt != net::Msg::PIVOT_SCAN) throw std::runtime_error("Unexpected pivot scan reply type");
                net::ByteReader rr(rp);
                double loc = rr.pod<double>();
                int gi = rr.pod<int32_t>();
                if (loc > best) { best = loc; best_gi = gi; }
            }
            double t1 = omp_get_wtime();
            t_pivot += (t1 - t0);

            if (best_gi < 0 || best == 0.0) {
                throw std::runtime_error("Matrix appears singular at k=" + std::to_string(k) + " (best pivot=0).");
            }

            // Swap if needed (swap rows k and best_gi; b is swapped as well by workers)
            t0 = omp_get_wtime();
            if (best_gi != k) {
                ++swap_total;
                int owner_k = owner_of_row(k);
                int owner_p = owner_of_row(best_gi);
                if (owner_k < 0 || owner_p < 0) throw std::runtime_error("owner_of_row failed");

                if (owner_k == owner_p) {
                    net::ByteWriter w; w.pod((int32_t)k); w.pod((int32_t)best_gi);
                    net::send_msg(conns[(size_t)owner_k].s, net::Msg::LOCAL_SWAP, w.buf);
                    expect_ok(conns[(size_t)owner_k].s);
                }
                else {
                    ++swap_cross;
                    // Estimate traffic: 2x GET_ROW (worker->driver) + 2x PUT_ROW (driver->worker)
                    // Payload per row message: int32 gi + double b + n doubles
                    const uint64_t row_payload = (uint64_t)sizeof(int32_t) + (uint64_t)sizeof(double) + (uint64_t)n * (uint64_t)sizeof(double);
                    swap_bytes += 4ULL * row_payload;

                    // GET_ROW both
                    net::ByteWriter wk; wk.pod((int32_t)k);
                    auto rk = rpc(conns[(size_t)owner_k].s, net::Msg::GET_ROW, wk.buf, net::Msg::GET_ROW);
                    net::ByteReader rrk(rk);
                    (void)rrk.pod<int32_t>(); // gi
                    double bk = rrk.pod<double>();
                    std::vector<double> rowk((size_t)n);
                    rrk.bytes(rowk.data(), (size_t)n * sizeof(double));

                    net::ByteWriter wp; wp.pod((int32_t)best_gi);
                    auto rp = rpc(conns[(size_t)owner_p].s, net::Msg::GET_ROW, wp.buf, net::Msg::GET_ROW);
                    net::ByteReader rrp(rp);
                    (void)rrp.pod<int32_t>();
                    double bp = rrp.pod<double>();
                    std::vector<double> rowp((size_t)n);
                    rrp.bytes(rowp.data(), (size_t)n * sizeof(double));

                    // PUT swapped
                    net::ByteWriter pk; pk.pod((int32_t)k); pk.pod(bp); pk.bytes(rowp.data(), (size_t)n * sizeof(double));
                    net::send_msg(conns[(size_t)owner_k].s, net::Msg::PUT_ROW, pk.buf);
                    expect_ok(conns[(size_t)owner_k].s);

                    net::ByteWriter pp; pp.pod((int32_t)best_gi); pp.pod(bk); pp.bytes(rowk.data(), (size_t)n * sizeof(double));
                    net::send_msg(conns[(size_t)owner_p].s, net::Msg::PUT_ROW, pp.buf);
                    expect_ok(conns[(size_t)owner_p].s);
                }
            }
            t1 = omp_get_wtime();
            t_swap += (t1 - t0);

            // Get pivot tail from owner of row k (after swap, row k is owned by same owner_of_row(k))
            int owner_k = owner_of_row(k);

            // GET_PIVOT_TAIL (timed separately)
            t0 = omp_get_wtime();
            {
                net::ByteWriter w; w.pod((int32_t)k);
                auto rp = rpc(conns[(size_t)owner_k].s, net::Msg::GET_PIVOT_TAIL, w.buf, net::Msg::GET_PIVOT_TAIL);
                net::ByteReader r(rp);
                int kk = r.pod<int32_t>();
                double pivot = r.pod<double>();
                int tail_len = r.pod<int32_t>();
                if (kk != k) throw std::runtime_error("GET_PIVOT_TAIL: k mismatch");
                if (tail_len != n - (k + 1)) throw std::runtime_error("GET_PIVOT_TAIL: tail_len mismatch");
                if (tail_len > 0) r.bytes(&tail[(size_t)k + 1], (size_t)tail_len * sizeof(double));

                t1 = omp_get_wtime();
                t_getpivot += (t1 - t0);

                // Broadcast pivot tail to all workers (timed separately)
                double tb0 = omp_get_wtime();

                net::ByteWriter wb;
                wb.pod((int32_t)k);
                wb.pod((double)pivot);
                wb.pod((int32_t)tail_len);
                if (tail_len > 0) wb.bytes(&tail[(size_t)k + 1], (size_t)tail_len * sizeof(double));

                // Estimate bytes sent by driver to workers for this broadcast
                // Payload: k(int32) + pivot(double) + tail_len(int32) + tail_len*doubles
                const uint64_t bcast_payload =
                    (uint64_t)sizeof(int32_t) +
                    (uint64_t)sizeof(double) +
                    (uint64_t)sizeof(int32_t) +
                    (uint64_t)tail_len * (uint64_t)sizeof(double);
                bcast_bytes_sent += (uint64_t)p * bcast_payload;

                for (int rnk = 0; rnk < p; ++rnk) {
                    net::send_msg(conns[(size_t)rnk].s, net::Msg::BCAST_PIVOT_TAIL, wb.buf);
                }
                for (int rnk = 0; rnk < p; ++rnk) expect_ok(conns[(size_t)rnk].s);

                double tb1 = omp_get_wtime();
                t_bcast_send += (tb1 - tb0);
            }

            // Eliminate on all workers
            t0 = omp_get_wtime();
            for (int rnk = 0; rnk < p; ++rnk) {
                net::ByteWriter we; we.pod((int32_t)k);
                net::send_msg(conns[(size_t)rnk].s, net::Msg::ELIMINATE, we.buf);
            }
            for (int rnk = 0; rnk < p; ++rnk) expect_ok(conns[(size_t)rnk].s);
            t1 = omp_get_wtime();
            t_elim += (t1 - t0);
        }

        double t_fact = omp_get_wtime() - t0_all;

        // Gather LU and b to driver
        double tg0 = omp_get_wtime();
        std::vector<double> LU((size_t)n * (size_t)n);
        std::vector<double> b_full((size_t)n);

        for (int r = 0; r < p; ++r) {
            auto rp = rpc(conns[(size_t)r].s, net::Msg::GET_BLOCK, {}, net::Msg::GET_BLOCK);
            net::ByteReader rr(rp);
            int row0 = rr.pod<int32_t>();
            int rows = rr.pod<int32_t>();
            if (row0 != displs[(size_t)r] || rows != counts[(size_t)r]) {
                throw std::runtime_error("GET_BLOCK: distribution mismatch from worker " + std::to_string(r));
            }
            // LU block
            rr.bytes(&LU[(size_t)row0 * (size_t)n], (size_t)rows * (size_t)n * sizeof(double));
            // b block
            rr.bytes(&b_full[(size_t)row0], (size_t)rows * sizeof(double));
        }
        double tg1 = omp_get_wtime();
        t_gather += (tg1 - tg0);

        // Solve on driver
        double ts0 = omp_get_wtime();
        std::vector<double> x;
        solve_on_root_inplace_LU(n, LU, b_full, x);
        double ts1 = omp_get_wtime();
        t_solve += (ts1 - ts0);

        double chk = checksum_weighted_first100(x);

        std::cout << "LU factorization wall time (driver): " << std::setprecision(6) << t_fact << " s\n";
        std::cout << "x checksum (first 100 weighted): " << std::setprecision(12) << chk << "\n";

        if (opt.timing) {
            std::cout << "\nTiming breakdown (driver, seconds):\n";
            std::cout << "  pivot (gather maxloc):  " << t_pivot << "\n";
            std::cout << "  swap (row exchanges):   " << t_swap << "\n";
            std::cout << "  getpivot (rpc tail):    " << t_getpivot << "\n";
            std::cout << "  bcast (send+wait):      " << t_bcast_send << "\n";
            std::cout << "  bcast total:            " << (t_getpivot + t_bcast_send) << "\n";
            std::cout << "  elim (coord+compute):   " << t_elim << "\n";
            std::cout << "Post-factorization:\n";
            std::cout << "  gather (LU,b):          " << t_gather << "\n";
            std::cout << "  solve (driver):         " << t_solve << "\n";

            std::cout << "\nSwap/comm counters (driver estimates):\n";
            std::cout << "  swaps_total:            " << swap_total << "\n";
            std::cout << "  swaps_cross_worker:     " << swap_cross << "\n";
            std::cout << "  swap_bytes (est):       " << swap_bytes << "\n";
            std::cout << "  bcast_bytes_sent (est): " << bcast_bytes_sent << "\n";
        }

        double rel = 0.0;
        if (opt.verify) {
            double rinf = 0.0, binf = 0.0, rrel = 0.0;
            compute_residual_inf_norms(n, opt.seed, ms, x, rinf, binf, rrel);
            rel = rrel;
            std::cout << "\nVerification (re-generated A,b on driver):\n";
            std::cout << "  ||Ax-b||_inf: " << std::setprecision(12) << rinf << "\n";
            std::cout << "  ||b||_inf:    " << std::setprecision(12) << binf << "\n";
            std::cout << "  rel residual: " << std::setprecision(12) << rrel << "\n";
        }

        // Pull worker stats (optional but useful)
        double max_worker = 0.0;
        double scan_max = 0.0, recv_max = 0.0, elim_max = 0.0;
        uint64_t bytes_in_sum = 0, bytes_out_sum = 0;
        if (opt.timing) {
            std::cout << "\nWorker timing (seconds):\n";
            for (int r = 0; r < p; ++r) {
                auto rp = rpc(conns[(size_t)r].s, net::Msg::GET_STATS, {}, net::Msg::GET_STATS);
                net::ByteReader rr(rp);
                double sscan = rr.pod<double>();
                double srecv = rr.pod<double>();
                double selim = rr.pod<double>();
                uint64_t bin = rr.pod<uint64_t>();
                uint64_t bout = rr.pod<uint64_t>();
                double tot = sscan + srecv + selim;
                max_worker = std::max(max_worker, tot);
                scan_max = std::max(scan_max, sscan);
                recv_max = std::max(recv_max, srecv);
                elim_max = std::max(elim_max, selim);
                bytes_in_sum += bin;
                bytes_out_sum += bout;
                std::cout << "  worker[" << r << "] scan=" << sscan << " recv=" << srecv << " elim=" << selim
                    << "  bytes_in=" << bin << " bytes_out=" << bout << "\n";
            }

            // Phase-level critical path estimates (best-effort)
            // Driver timings include waiting for workers. Worker timings are pure worker-side work.
            // Reporting both helps explain where time goes.
            const double crit_pivot = std::max(t_pivot, scan_max);
            const double crit_swap = t_swap; // worker swap time not tracked separately
            const double crit_bcast = std::max(t_bcast_send, recv_max) + t_getpivot; // getpivot is driver-only
            const double crit_elim = std::max(t_elim, elim_max);
            std::cout << "\nCritical-path estimates (seconds):\n";
            std::cout << "  pivot  crit=" << crit_pivot << " (driver=" << t_pivot << ", workers_scan_max=" << scan_max << ")\n";
            std::cout << "  swap   crit=" << crit_swap << " (driver=" << t_swap << ")\n";
            std::cout << "  bcast  crit=" << crit_bcast << " (getpivot=" << t_getpivot << ", driver_send=" << t_bcast_send
                << ", workers_recv_max=" << recv_max << ")\n";
            std::cout << "  elim   crit=" << crit_elim << " (driver=" << t_elim << ", workers_elim_max=" << elim_max << ")\n";
            std::cout << "\nWorker traffic summary (counters from workers):\n";
            std::cout << "  bytes_in_sum=" << bytes_in_sum << " bytes_out_sum=" << bytes_out_sum << "\n";
        }

        // CSV line
        if (!opt.csv_path.empty()) {
            std::ostringstream line;
            line << n << "," << p << "," << opt.threads << "," << opt.seed << "," << mode_name(opt.mode) << ","
                << ms.alpha << "," << ms.beta << "," << ms.eps << ","
                << t_fact << "," << t_pivot << "," << t_swap << "," << t_getpivot << "," << t_bcast_send << "," << t_elim << ","
                << t_gather << "," << t_solve << ","
                << swap_total << "," << swap_cross << "," << swap_bytes << "," << bcast_bytes_sent << ","
                << rel << "," << chk << ",";

            // critical-path factorization estimate (max(driver, max_worker))
            double critical_fact = std::max(t_fact, max_worker);
            line << critical_fact;
            csv_append_line(opt.csv_path, line.str());
        }

        // Shutdown workers
        if (!opt.keep_workers) {
            for (int r = 0; r < p; ++r) net::send_msg(conns[(size_t)r].s, net::Msg::SHUTDOWN, {});
            for (int r = 0; r < p; ++r) expect_ok(conns[(size_t)r].s);
        }
        else {
			std::cout << "\nNote: workers kept alive (--keep-workers); remember to shut them down manually.\n";
        }
        for (auto& c : conns) net::closesocket_safe(c.s);

        // For report: an estimate of critical-path factorization time is max(driver, max(worker)).
        if (opt.timing) {
            double critical = std::max(t_fact, max_worker);
            std::cout << "\nEstimated critical-path factorization time: " << critical << " s\n";
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "lu_driver fatal: " << e.what() << "\n";
        return 1;
    }
}
