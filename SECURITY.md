# Security policy and network boundary

Parallel-LU is educational coursework, not a network service for production or
untrusted environments.

The custom MPI-lite transport has no authentication, authorization, or
encryption. It serializes native in-memory values and therefore requires
compatible builds on a trusted, homogeneous Windows lab network. The default
worker binding is loopback (`127.0.0.1`). Do not expose a worker to the public
internet or a network containing untrusted peers.

The maintained code caps frames at 64 MiB, marks protocol version 1, validates
ports and run dimensions, and sets 30-second connected-socket I/O timeouts. These
controls bound common mistakes and resource abuse; they do not make the protocol
secure against a hostile participant or provide fault tolerance.

Please report a vulnerability privately through GitHub's security-advisory
interface rather than opening a public issue containing exploit details.
