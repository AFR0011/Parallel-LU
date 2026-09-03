#include "lu_common.h"
#include "lu_net.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool parse_driver(std::vector<std::string> args, RunOptions& options, std::string& error) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) argv.push_back(arg.data());
    return parse_driver_args((int)argv.size(), argv.data(), options, error);
}

bool parse_worker(std::vector<std::string> args, WorkerOptions& options, std::string& error) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) argv.push_back(arg.data());
    return parse_worker_args((int)argv.size(), argv.data(), options, error);
}

} // namespace

int main() {
    try {
        MatrixMode mode{};
        require(parse_mode("near_singular", mode), "near_singular mode should parse");
        require(mode == MatrixMode::NearSingular, "parsed matrix mode mismatch");

        const MatrixSpec spec{ MatrixMode::DiagDominant, 2.0, 1.0, 1e-3 };
        std::vector<double> row_a(8), row_b(8);
        double rhs_a = 0.0, rhs_b = 0.0, sum_a = 0.0, sum_b = 0.0;
        generate_row(8, 42, spec, 3, row_a.data(), rhs_a, sum_a);
        generate_row(8, 42, spec, 3, row_b.data(), rhs_b, sum_b);
        require(row_a == row_b && rhs_a == rhs_b, "matrix generation must be deterministic");

        const std::vector<double> lu = { 4.0, 3.0, 0.5, -0.5 };
        const std::vector<double> rhs = { 10.0, 4.0 };
        std::vector<double> solution;
        solve_on_root_inplace_LU(2, lu, rhs, solution);
        require(std::abs(solution[0] - 1.0) < 1e-12, "known solve x[0] mismatch");
        require(std::abs(solution[1] - 2.0) < 1e-12, "known solve x[1] mismatch");
        require(count_non_finite(solution) == 0, "known solution should be finite");

        bool invalid_diagonal_rejected = false;
        try {
            solve_on_root_inplace_LU(1, { 0.0 }, { 1.0 }, solution);
        } catch (const std::runtime_error&) {
            invalid_diagonal_rejected = true;
        }
        require(invalid_diagonal_rejected, "zero diagonal must be rejected");

        RunOptions serial_options;
        std::string error;
        require(
            parse_driver({ "lu_driver", "32", "--serial", "--verify" }, serial_options, error),
            "explicit serial arguments should parse"
        );
        require(serial_options.serial && serial_options.verify, "serial flags were not retained");

        RunOptions ambiguous_options;
        error.clear();
        require(
            !parse_driver(
                { "lu_driver", "32", "--serial", "--hosts", "hosts.txt" },
                ambiguous_options,
                error
            ),
            "serial and hosts must be mutually exclusive"
        );

        WorkerOptions worker_options;
        error.clear();
        require(
            !parse_worker(
                { "lu_worker", "--listen", "127.0.0.1:70000" },
                worker_options,
                error
            ),
            "out-of-range worker port must be rejected"
        );
        WorkerOptions trailing_port_options;
        error.clear();
        require(
            !parse_worker(
                { "lu_worker", "--listen", "127.0.0.1:5000junk" },
                trailing_port_options,
                error
            ),
            "worker port with trailing characters must be rejected"
        );

        require(
            net::parse_hostport("127.0.0.1:5000").second == 5000,
            "valid host port should parse"
        );
        bool bad_host_rejected = false;
        try {
            (void)net::parse_hostport("127.0.0.1:70000");
        } catch (const std::runtime_error&) {
            bad_host_rejected = true;
        }
        require(bad_host_rejected, "out-of-range host port must be rejected");

        bool oversized_payload_rejected = false;
        try {
            net::validate_payload_size(net::kMaxFrameBytes);
        } catch (const std::runtime_error&) {
            oversized_payload_rejected = true;
        }
        require(oversized_payload_rejected, "oversized payload must be rejected");

        std::cout << "Common numerical, CLI, and frame contracts passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Contract test failed: " << error.what() << "\n";
        return 1;
    }
}
