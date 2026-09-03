"""Synthetic serial and two-worker smoke checks for the Windows MPI-lite path."""

from __future__ import annotations

import argparse
import math
import os
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path


def run(command: list[str], expected: int = 0, timeout: int = 60) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != expected:
        raise AssertionError(
            f"Expected exit {expected}, got {completed.returncode}: {' '.join(command)}\n{output}"
        )
    return output


def metric(output: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}\s*:\s*([-+0-9.eE]+)", output)
    if not match:
        raise AssertionError(f"Missing metric '{label}' in output:\n{output}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise AssertionError(f"Metric '{label}' is non-finite: {value}")
    return value


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_for_listener(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"Worker did not listen on port {port}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()

    driver = str(Path(args.driver).resolve())
    worker = str(Path(args.worker).resolve())
    common = [
        "32",
        "--matrix",
        "rand",
        "--threads",
        "1",
        "--timing",
        "--verify",
    ]

    serial_output = run([driver, *common, "--serial"])
    assert "status:       PASSED" in serial_output
    serial_checksum = metric(serial_output, "x checksum (first 100 weighted)")
    assert metric(serial_output, "rel residual") <= 1e-10

    with tempfile.TemporaryDirectory(prefix="parallel-lu-csv-") as csv_dir:
        csv_path = Path(csv_dir) / "serial.csv"
        run(
            [
                driver,
                "8",
                "--serial",
                "--matrix",
                "dd",
                "--threads",
                "1",
                "--verify",
                "--csv",
                str(csv_path),
            ]
        )
        csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert len(csv_lines) == 2
        assert "residual_tol,verification_passed" in csv_lines[0]
        assert csv_lines[1].split(",")[-3] == "1"

    singular_output = run(
        [
            driver,
            "16",
            "--serial",
            "--matrix",
            "near_singular",
            "--eps",
            "0",
            "--beta",
            "0",
            "--threads",
            "1",
            "--verify",
        ],
        expected=3,
    )
    assert "status:       FAILED" in singular_output

    run([driver, "9000", "--serial"], expected=2)
    run([worker, "--listen", "127.0.0.1:70000"], expected=2)

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with tempfile.TemporaryDirectory(prefix="parallel-lu-smoke-") as temp_dir:
        temp = Path(temp_dir)
        ports = [free_port(), free_port()]
        while ports[1] == ports[0]:
            ports[1] = free_port()
        logs = [open(temp / f"worker-{index}.log", "w+", encoding="utf-8") for index in range(2)]
        workers: list[subprocess.Popen[str]] = []
        try:
            for port, log in zip(ports, logs, strict=True):
                workers.append(
                    subprocess.Popen(
                        [worker, "--listen", f"127.0.0.1:{port}", "--threads", "1"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        creationflags=creation_flags,
                    )
                )
            for port in ports:
                wait_for_listener(port)
            time.sleep(0.3)

            hosts = temp / "hosts.txt"
            hosts.write_text(
                "\n".join(f"127.0.0.1:{port}" for port in ports) + "\n",
                encoding="utf-8",
            )
            distributed_output = run([driver, *common, "--hosts", str(hosts)])
            assert "status:       PASSED" in distributed_output
            distributed_checksum = metric(
                distributed_output, "x checksum (first 100 weighted)"
            )
            assert math.isclose(
                serial_checksum, distributed_checksum, rel_tol=1e-12, abs_tol=1e-12
            )
            assert metric(distributed_output, "rel residual") <= 1e-10
            assert metric(distributed_output, "swaps_cross_worker") > 0

            for process in workers:
                process.wait(timeout=10)
                assert process.returncode == 0
        finally:
            for process in workers:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            for log in logs:
                log.close()

    print("Serial pass/fail and two-worker loopback agreement checks passed.")


if __name__ == "__main__":
    if os.name != "nt":
        raise SystemExit("This maintained integration check requires Windows.")
    main()
