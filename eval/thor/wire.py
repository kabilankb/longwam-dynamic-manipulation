"""Tiny length-prefixed wire format shared by the Thor server and the yamlab6 client.

A message is: 4-byte big-endian header length, a JSON header, then the binary blobs whose
sizes are listed in header["_blobs"]. JSON + raw bytes only (no pickle), so a malformed peer
cannot execute code on the other side.
"""
import json
import socket
import struct


def send_msg(sock: socket.socket, header: dict, blobs=()) -> None:
    blobs = [bytes(b) for b in blobs]
    header = dict(header, _blobs=[len(b) for b in blobs])
    raw = json.dumps(header).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw + b"".join(blobs))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket):
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, n))
    blobs = [_recv_exact(sock, size) for size in header.pop("_blobs", [])]
    return header, blobs
