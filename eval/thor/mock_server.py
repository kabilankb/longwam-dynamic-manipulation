"""Protocol-compatible stand-in for longwam_yam_server.py: returns 'hold current pose' chunks.

Used to test the yamlab6 client without the model:  python mock_server.py --port 8781
"""
import argparse
import socket

import cv2
import numpy as np

from wire import recv_msg, send_msg

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8781)
args = p.parse_args()
lsock = socket.socket()
lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
lsock.bind(("127.0.0.1", args.port))
lsock.listen(1)
print("mock listening", args.port, flush=True)
conn, _ = lsock.accept()
episode, last_state, n_obs = 0, None, 0
try:
    while True:
        h, blobs = recv_msg(conn)
        if h["op"] == "info":
            send_msg(conn, {"ok": True, "info": {"action_horizon": 32, "mock": True}})
        elif h["op"] == "reset":
            episode += 1
            send_msg(conn, {"ok": True, "episode": episode})
        else:
            for o in h["obs"]:
                imgs = {c: cv2.imdecode(np.frombuffer(blobs[i], np.uint8), cv2.IMREAD_COLOR) for c, i in o["images"].items()}
                if n_obs == 0:
                    print("first obs:", {c: im.shape for c, im in imgs.items()}, "state", np.round(o["state"], 3), flush=True)
                    for c, im in imgs.items():
                        cv2.imwrite(f"mock_first_{c}.jpg", im)
                last_state, n_obs = o["state"], n_obs + 1
            r = {"ok": True}
            if h["infer"]:
                r.update(anchor=n_obs - 1, actions=[last_state] * 32, infer_s=0.0)
            send_msg(conn, r)
except ConnectionError:
    print("client closed; observations received:", n_obs, flush=True)
