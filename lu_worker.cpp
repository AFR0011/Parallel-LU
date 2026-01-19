// lu_worker.cpp - MPI-lite worker process
//
// Usage (run on every compute node):
//   lu_worker.exe --listen 0.0.0.0:5000
//
// The worker holds a block of rows [row0, row0+rows) and performs OpenMP elimination steps
// on command from the driver.
//
// IMPORTANT:
//  - With --keep-workers on the driver, the driver will close the TCP connection after a run.
//    That is normal. The worker must treat "peer closed connection" as "session ended" and
//    return to accept() for the next run.

#include "lu_common.h"
#include "lu_net.h"

#include <iostream>
#include <fstream>
#include <omp.h>

static inline double& Aat(std::vector<double>& A, int n, int li, int j) {
    return A[(size_t)li * (size_t)n + (size_t)j];
}
static inline const double& Aat(const std::vector<double>& A, int n, int li, int j) {
    return A[(size_t)li * (size_t)n + (size_t)j];
}

struct WorkerState {
    int n = 0;
    int rank = 0;
    int p = 1;
    int row0 = 0;
    int rows = 0;

    uint64_t seed = 0;
    MatrixSpec ms{};
    int threads = 0;

    std::vector<double> A; // rows*n
    std::vector<double> b; // rows

    // Current pivot info for step k:
    int cur_k = -1;
    double cur_pivot = 0.0;
    std::vector<double> pivot_tail; // size n, only [k+1..n-1] valid

    // Timing counters
    double t_scan = 0.0;
    double t_elim = 0.0;
    double t_recv_pivot = 0.0;
    uint64_t bytes_in = 0;
    uint64_t bytes_out = 0;

    void init_storage() {
        A.assign((size_t)rows * (size_t)n, 0.0);
        b.assign((size_t)rows, 0.0);
        pivot_tail.assign((size_t)n, 0.0);

        std::vector<double> row((size_t)n);
        for (int li = 0; li < rows; ++li) {
            int gi = row0 + li;
            double bi = 0.0, sumabs = 0.0;
            generate_row(n, seed, ms, gi, row.data(), bi, sumabs);
            std::copy(row.begin(), row.end(), &A[(size_t)li * (size_t)n]);
            b[(size_t)li] = bi;
        }
    }
};

static void send_ok(SOCKET s) {
    net::send_msg(s, net::Msg::OK, {});
}
static void send_err(SOCKET s, const std::string& msg) {
    net::ByteWriter w;
    uint32_t len = (uint32_t)msg.size();
    w.pod(len);
    w.bytes(msg.data(), msg.size());
    net::send_msg(s, net::Msg::ERR, w.buf);
}

static bool handle_message(SOCKET s, WorkerState& st, net::Msg type, const std::vector<uint8_t>& payload) {
    using namespace std;

    try {
        if (type == net::Msg::INIT) {
            net::ByteReader r(payload);
            st.n = r.pod<int32_t>();
            st.rank = r.pod<int32_t>();
            st.p = r.pod<int32_t>();
            st.row0 = r.pod<int32_t>();
            st.rows = r.pod<int32_t>();
            st.seed = r.pod<uint64_t>();
            int32_t mode_i = r.pod<int32_t>();
            st.ms.mode = (MatrixMode)mode_i;
            st.ms.alpha = r.pod<double>();
            st.ms.beta = r.pod<double>();
            st.ms.eps = r.pod<double>();
            st.threads = r.pod<int32_t>();

            if (st.threads > 0) omp_set_num_threads(st.threads);

            st.init_storage();
            send_ok(s);
            return true;
        }

        if (st.n <= 0) {
            send_err(s, "Worker not initialized (missing INIT).");
            return true;
        }

        if (type == net::Msg::PIVOT_SCAN) {
            net::ByteReader r(payload);
            int k = r.pod<int32_t>();

            double t0 = omp_get_wtime();
            double best = 0.0;
            int best_gi = -1;

            int start_g = std::max(k, st.row0);
            int end_g = st.row0 + st.rows;
            if (start_g < end_g) {
                int li0 = start_g - st.row0;
                for (int li = li0; li < st.rows; ++li) {
                    int gi = st.row0 + li;
                    double v = std::abs(Aat(st.A, st.n, li, k));
                    if (v > best) { best = v; best_gi = gi; }
                }
            }

            double t1 = omp_get_wtime();
            st.t_scan += (t1 - t0);

            net::ByteWriter w;
            w.pod(best);
            w.pod((int32_t)best_gi);
            st.bytes_out += sizeof(uint32_t) + 4 + w.buf.size();
            net::send_msg(s, net::Msg::PIVOT_SCAN, w.buf);
            return true;
        }

        if (type == net::Msg::LOCAL_SWAP) {
            net::ByteReader r(payload);
            int gi1 = r.pod<int32_t>();
            int gi2 = r.pod<int32_t>();
            int li1 = gi1 - st.row0;
            int li2 = gi2 - st.row0;
            if (li1 < 0 || li1 >= st.rows || li2 < 0 || li2 >= st.rows) {
                send_err(s, "LOCAL_SWAP rows not owned by this worker.");
                return true;
            }
            for (int j = 0; j < st.n; ++j) std::swap(Aat(st.A, st.n, li1, j), Aat(st.A, st.n, li2, j));
            std::swap(st.b[(size_t)li1], st.b[(size_t)li2]);
            send_ok(s);
            return true;
        }

        if (type == net::Msg::GET_ROW) {
            net::ByteReader r(payload);
            int gi = r.pod<int32_t>();
            int li = gi - st.row0;
            if (li < 0 || li >= st.rows) {
                send_err(s, "GET_ROW row not owned by this worker.");
                return true;
            }
            net::ByteWriter w;
            w.pod((int32_t)gi);
            w.pod(st.b[(size_t)li]);
            w.bytes(&st.A[(size_t)li * (size_t)st.n], (size_t)st.n * sizeof(double));
            st.bytes_out += sizeof(uint32_t) + 4 + w.buf.size();
            net::send_msg(s, net::Msg::GET_ROW, w.buf);
            return true;
        }

        if (type == net::Msg::PUT_ROW) {
            net::ByteReader r(payload);
            int gi = r.pod<int32_t>();
            double bi = r.pod<double>();
            int li = gi - st.row0;
            if (li < 0 || li >= st.rows) {
                send_err(s, "PUT_ROW row not owned by this worker.");
                return true;
            }
            st.b[(size_t)li] = bi;
            r.bytes(&st.A[(size_t)li * (size_t)st.n], (size_t)st.n * sizeof(double));
            send_ok(s);
            return true;
        }

        if (type == net::Msg::GET_PIVOT_TAIL) {
            net::ByteReader r(payload);
            int k = r.pod<int32_t>();
            int li = k - st.row0; // pivot row index equals global k
            if (li < 0 || li >= st.rows) {
                send_err(s, "GET_PIVOT_TAIL: pivot row not owned by this worker.");
                return true;
            }
            double pivot = Aat(st.A, st.n, li, k);
            int tail_len = st.n - (k + 1);

            net::ByteWriter w;
            w.pod((int32_t)k);
            w.pod(pivot);
            w.pod((int32_t)tail_len);
            if (tail_len > 0) {
                w.bytes(&st.A[(size_t)li * (size_t)st.n + (size_t)(k + 1)], (size_t)tail_len * sizeof(double));
            }
            st.bytes_out += sizeof(uint32_t) + 4 + w.buf.size();
            net::send_msg(s, net::Msg::GET_PIVOT_TAIL, w.buf);
            return true;
        }

        if (type == net::Msg::BCAST_PIVOT_TAIL) {
            double t0 = omp_get_wtime();
            net::ByteReader r(payload);
            st.cur_k = r.pod<int32_t>();
            st.cur_pivot = r.pod<double>();
            int tail_len = r.pod<int32_t>();
            if (st.cur_k < 0 || st.cur_k >= st.n) {
                send_err(s, "BCAST_PIVOT_TAIL: bad k");
                return true;
            }
            if (tail_len != st.n - (st.cur_k + 1)) {
                send_err(s, "BCAST_PIVOT_TAIL: tail_len mismatch");
                return true;
            }
            if (tail_len > 0) {
                r.bytes(&st.pivot_tail[(size_t)st.cur_k + 1], (size_t)tail_len * sizeof(double));
            }
            double t1 = omp_get_wtime();
            st.t_recv_pivot += (t1 - t0);
            send_ok(s);
            return true;
        }

        if (type == net::Msg::ELIMINATE) {
            net::ByteReader r(payload);
            int k = r.pod<int32_t>();
            if (k != st.cur_k) {
                send_err(s, "ELIMINATE: k does not match last BCAST_PIVOT_TAIL");
                return true;
            }
            double pivot = st.cur_pivot;
            if (pivot == 0.0) {
                send_err(s, "ELIMINATE: pivot is zero");
                return true;
            }

            double t0 = omp_get_wtime();
            int start_g = std::max(k + 1, st.row0);
            int end_g = st.row0 + st.rows;
            if (start_g < end_g) {
                int li0 = start_g - st.row0;
#pragma omp parallel for schedule(static)
                for (int li = li0; li < st.rows; ++li) {
                    double aik = Aat(st.A, st.n, li, k) / pivot;
                    Aat(st.A, st.n, li, k) = aik; // store L

                    double* rowp = &st.A[(size_t)li * (size_t)st.n];
#pragma omp simd
                    for (int j = k + 1; j < st.n; ++j) {
                        rowp[(size_t)j] -= aik * st.pivot_tail[(size_t)j];
                    }
                }
            }
            double t1 = omp_get_wtime();
            st.t_elim += (t1 - t0);
            send_ok(s);
            return true;
        }

        if (type == net::Msg::GET_BLOCK) {
            net::ByteWriter w;
            w.pod((int32_t)st.row0);
            w.pod((int32_t)st.rows);
            w.bytes(st.A.data(), st.A.size() * sizeof(double));
            w.bytes(st.b.data(), st.b.size() * sizeof(double));
            st.bytes_out += sizeof(uint32_t) + 4 + w.buf.size();
            net::send_msg(s, net::Msg::GET_BLOCK, w.buf);
            return true;
        }

        if (type == net::Msg::GET_STATS) {
            net::ByteWriter w;
            w.pod((double)st.t_scan);
            w.pod((double)st.t_recv_pivot);
            w.pod((double)st.t_elim);
            w.pod((uint64_t)st.bytes_in);
            w.pod((uint64_t)st.bytes_out);
            net::send_msg(s, net::Msg::GET_STATS, w.buf);
            return true;
        }

        if (type == net::Msg::SHUTDOWN) {
            send_ok(s);
            return false; // end session and exit program
        }

        send_err(s, "Unknown message type.");
        return true;
    }
    catch (const std::exception& e) {
        send_err(s, std::string("Exception: ") + e.what());
        return true;
    }
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);

    WorkerOptions wopt;
    std::string err;
    if (!parse_worker_args(argc, argv, wopt, err)) {
        std::cerr << err << "\n";
        return 2;
    }

    try {
        net::WSAInit wsa;

        std::cout << "lu_worker: listening on " << wopt.bind_ip << ":" << wopt.port << "\n";
        std::cout << "lu_worker: ready (will accept multiple driver sessions).\n";

        for (;;) {
            SOCKET s = INVALID_SOCKET;
            try {
                s = net::listen_and_accept(wopt.bind_ip, wopt.port);
                std::cout << "lu_worker: driver connected.\n";

                WorkerState st;
                if (wopt.threads > 0) st.threads = wopt.threads;

                bool shutdown_requested = false;

                for (;;) {
                    net::Msg type;
                    std::vector<uint8_t> payload;

                    try {
                        net::recv_msg(s, type, payload);
                    }
                    catch (const std::exception& e) {
                        // Normal end-of-session when driver closes connection in --keep-workers mode.
                        std::string msg = e.what();
                        if (msg.find("peer closed connection") != std::string::npos) {
                            std::cout << "lu_worker: driver disconnected (session ended).\n";
                            break; // back to accept()
                        }
                        throw; // real recv error
                    }

                    st.bytes_in += sizeof(uint32_t) + 4 + payload.size();

                    bool cont = handle_message(s, st, type, payload);
                    if (!cont) {
                        // Only SHUTDOWN returns false.
                        shutdown_requested = true;
                        break;
                    }
                }

                net::closesocket_safe(s);
                s = INVALID_SOCKET;

                if (shutdown_requested) {
                    std::cout << "lu_worker: shutdown.\n";
                    return 0;
                }

                // otherwise continue accepting new sessions
            }
            catch (const std::exception& e) {
                if (s != INVALID_SOCKET) net::closesocket_safe(s);
                std::cerr << "lu_worker session error: " << e.what() << "\n";
                // keep worker alive; go back to accept
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "lu_worker fatal: " << e.what() << "\n";
        return 1;
    }
}
