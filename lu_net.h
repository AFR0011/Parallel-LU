#pragma once
// lu_net.h - minimal Winsock2 helpers + length-prefixed messaging for MPI-lite.
//
// Build note (Visual Studio):
//   - Add Ws2_32.lib to Linker -> Input -> Additional Dependencies
//   - Include dirs should contain this header.

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "Ws2_32.lib")
#else
  #error "This MPI-lite networking layer is implemented for Windows (Winsock2)."
#endif

#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <iostream>
#include <limits>

namespace net {

constexpr uint16_t kProtocolVersion = 1;
constexpr uint32_t kMaxFrameBytes = 64u * 1024u * 1024u;
constexpr DWORD kSocketTimeoutMs = 30000;

struct WSAInit {
    WSAInit() {
        WSADATA wsa;
        int rc = WSAStartup(MAKEWORD(2, 2), &wsa);
        if (rc != 0) throw std::runtime_error("WSAStartup failed: " + std::to_string(rc));
    }
    ~WSAInit() { WSACleanup(); }
};

inline std::string last_error(const char* where) {
    int e = WSAGetLastError();
    std::ostringstream oss;
    oss << where << " failed (WSAGetLastError=" << e << ")";
    return oss.str();
}

inline void closesocket_safe(SOCKET s) {
    if (s != INVALID_SOCKET) closesocket(s);
}

inline void send_all(SOCKET s, const void* data, size_t n) {
    const char* p = (const char*)data;
    while (n > 0) {
        int sent = ::send(s, p, (int)std::min<size_t>(n, 1u<<30), 0);
        if (sent == SOCKET_ERROR) throw std::runtime_error(last_error("send"));
        if (sent == 0) throw std::runtime_error("send failed: zero bytes written");
        p += sent;
        n -= (size_t)sent;
    }
}

inline void recv_all(SOCKET s, void* data, size_t n) {
    char* p = (char*)data;
    while (n > 0) {
        int recvd = ::recv(s, p, (int)std::min<size_t>(n, 1u<<30), 0);
        if (recvd == 0) throw std::runtime_error("recv failed: peer closed connection");
        if (recvd == SOCKET_ERROR) throw std::runtime_error(last_error("recv"));
        p += recvd;
        n -= (size_t)recvd;
    }
}

inline void set_nodelay(SOCKET s, bool on=true) {
    int flag = on ? 1 : 0;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&flag, sizeof(flag));
}

inline void set_io_timeouts(SOCKET s, DWORD timeout_ms = kSocketTimeoutMs) {
    if (setsockopt(
            s, SOL_SOCKET, SO_RCVTIMEO,
            reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms)
        ) == SOCKET_ERROR) {
        throw std::runtime_error(last_error("setsockopt(SO_RCVTIMEO)"));
    }
    if (setsockopt(
            s, SOL_SOCKET, SO_SNDTIMEO,
            reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms)
        ) == SOCKET_ERROR) {
        throw std::runtime_error(last_error("setsockopt(SO_SNDTIMEO)"));
    }
}

inline SOCKET connect_tcp(const std::string& host, uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%u", (unsigned)port);

    addrinfo* res = nullptr;
    int rc = getaddrinfo(host.c_str(), port_str, &hints, &res);
    if (rc != 0 || !res) throw std::runtime_error("getaddrinfo failed for " + host + ":" + port_str);

    SOCKET s = INVALID_SOCKET;
    for (addrinfo* p = res; p; p = p->ai_next) {
        s = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (s == INVALID_SOCKET) continue;
        if (::connect(s, p->ai_addr, (int)p->ai_addrlen) == 0) break;
        closesocket_safe(s);
        s = INVALID_SOCKET;
    }
    freeaddrinfo(res);
    if (s == INVALID_SOCKET) throw std::runtime_error("connect failed to " + host + ":" + port_str);
    set_nodelay(s, true);
    set_io_timeouts(s);
    return s;
}

inline SOCKET listen_and_accept(const std::string& bind_ip, uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%u", (unsigned)port);

    addrinfo* res = nullptr;
    int rc = getaddrinfo(bind_ip.empty() ? nullptr : bind_ip.c_str(), port_str, &hints, &res);
    if (rc != 0 || !res) throw std::runtime_error("getaddrinfo(listen) failed");

    SOCKET ls = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (ls == INVALID_SOCKET) { freeaddrinfo(res); throw std::runtime_error(last_error("socket(listen)")); }

    BOOL yes = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));

    if (::bind(ls, res->ai_addr, (int)res->ai_addrlen) == SOCKET_ERROR) {
        freeaddrinfo(res);
        closesocket_safe(ls);
        throw std::runtime_error(last_error("bind"));
    }
    freeaddrinfo(res);

    if (::listen(ls, 1) == SOCKET_ERROR) {
        closesocket_safe(ls);
        throw std::runtime_error(last_error("listen"));
    }

    SOCKET cs = ::accept(ls, nullptr, nullptr);
    closesocket_safe(ls);
    if (cs == INVALID_SOCKET) throw std::runtime_error(last_error("accept"));
    set_nodelay(cs, true);
    set_io_timeouts(cs);
    return cs;
}

// ---------------------- Message packing ----------------------

enum class Msg : uint16_t {
    INIT = 1,
    PIVOT_SCAN = 2,
    GET_ROW = 3,
    PUT_ROW = 4,
    LOCAL_SWAP = 5,
    GET_PIVOT_TAIL = 6,
    BCAST_PIVOT_TAIL = 7,
    ELIMINATE = 8,
    GET_BLOCK = 9,
    GET_STATS = 10,
    SHUTDOWN = 11,

    OK = 200,
    ERR = 500
};

struct ByteWriter {
    std::vector<uint8_t> buf;
    template <typename T>
    void pod(const T& v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        buf.insert(buf.end(), p, p + sizeof(T));
    }
    void bytes(const void* data, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
        buf.insert(buf.end(), p, p + n);
    }
};

struct ByteReader {
    const uint8_t* p;
    const uint8_t* e;
    ByteReader(const std::vector<uint8_t>& b) : p(b.data()), e(b.data() + b.size()) {}
    template <typename T>
    T pod() {
        if ((size_t)(e - p) < sizeof(T)) throw std::runtime_error("ByteReader: underflow");
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    void bytes(void* out, size_t n) {
        if ((size_t)(e - p) < n) throw std::runtime_error("ByteReader: underflow(bytes)");
        std::memcpy(out, p, n);
        p += n;
    }
};

// Frame: uint32 payload_bytes, then uint16 msg, uint16 reserved, then payload.
inline void validate_payload_size(size_t payload_size) {
    if (payload_size > (size_t)kMaxFrameBytes - 4u ||
        payload_size > (size_t)std::numeric_limits<uint32_t>::max() - 4u) {
        throw std::runtime_error("Message payload exceeds the 64 MiB frame limit");
    }
}

inline void send_msg(SOCKET s, Msg type, const std::vector<uint8_t>& payload) {
    validate_payload_size(payload.size());
    uint32_t payload_bytes = (uint32_t)(payload.size() + 4); // msg+reserved included in payload area
    uint16_t t = (uint16_t)type;
    uint16_t rsv = kProtocolVersion;
    send_all(s, &payload_bytes, sizeof(payload_bytes));
    send_all(s, &t, sizeof(t));
    send_all(s, &rsv, sizeof(rsv));
    if (!payload.empty()) send_all(s, payload.data(), payload.size());
}

inline void recv_msg(SOCKET s, Msg& type_out, std::vector<uint8_t>& payload_out) {
    uint32_t payload_bytes = 0;
    recv_all(s, &payload_bytes, sizeof(payload_bytes));
    if (payload_bytes < 4) throw std::runtime_error("Bad frame: payload_bytes<4");
    if (payload_bytes > kMaxFrameBytes) {
        throw std::runtime_error("Bad frame: payload exceeds the 64 MiB limit");
    }
    uint16_t t=0, rsv=0;
    recv_all(s, &t, sizeof(t));
    recv_all(s, &rsv, sizeof(rsv));
    if (rsv != kProtocolVersion) {
        throw std::runtime_error("Bad frame: incompatible protocol version");
    }
    type_out = (Msg)t;
    payload_out.resize(payload_bytes - 4);
    if (!payload_out.empty()) recv_all(s, payload_out.data(), payload_out.size());
}

inline std::pair<std::string,uint16_t> parse_hostport(const std::string& s) {
    auto pos = s.find(':');
    if (pos == std::string::npos || pos == 0 || pos != s.rfind(':')) {
        throw std::runtime_error("Host entry must use hostname-or-IPv4:port: " + s);
    }
    size_t consumed = 0;
    unsigned long port = std::stoul(s.substr(pos + 1), &consumed);
    if (consumed != s.size() - pos - 1 || port < 1 || port > 65535) {
        throw std::runtime_error("Host port must be between 1 and 65535: " + s);
    }
    return {s.substr(0,pos), (uint16_t)port};
}

} // namespace net
