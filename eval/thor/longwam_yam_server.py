"""Long-WAM-YAM policy server (runs on the Jetson Thor).

Wraps Long-WAM's in-process runtime (longwam.runtime.policy.initialize_policy with the YAM
adapter) behind a TCP socket so a remote simulator can stream observations and get 32-step
action chunks back.

Protocol (see wire.py), one request -> one reply:
  {"op": "info"}                                     -> {"ok", "info": {...}}
  {"op": "reset", "instruction": str}                -> {"ok", "episode": int}
  {"op": "step", "episode": int, "obs": [ {"step": int, "state": [14 floats],
        "images": {"top": i, "left_wrist": j, "right_wrist": k}} ... ],  # i,j,k = blob index (JPEG/PNG)
   "infer": bool}
     -> {"ok", "anchor": int, "actions": [[14 floats] x H], "infer_s": float}   (actions only if infer)

Observations must be sent for every consecutive control step (the model samples every 4th one
for its 48-step history); "infer" predicts from the newest observation in the request.
State/action are in checkpoint units: per arm 6 joints [rad] + gripper [0..1], left then right.

Usage:
  python longwam_yam_server.py --ckpt checkpoints/yam --port 8780
"""
import argparse
import logging
import socket
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from wire import recv_msg, send_msg

CAMERAS = ("top", "left_wrist", "right_wrist")
ACTION_NAMES = [f"{side}_{j}" for side in ("left", "right")
                for j in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper")]

log = logging.getLogger("longwam-yam")


def build_settings(args):
    ckpt = Path(args.ckpt).expanduser().resolve()
    return {
        "robot": "yam",
        "checkpoint": str(ckpt / "model.pt"),
        "model_config": str(Path(args.model_config).expanduser().resolve() if args.model_config else ckpt / "config.yaml"),
        "stats": str(ckpt / "dataset_stats.json"),
        "text_cache": None,                      # encode prompts with the umT5 encoder
        "adapter_options": {"action_names": ACTION_NAMES, "state_names": None},
        "execution": "sync",
        "action_horizon": args.action_horizon,
        "execute_steps": args.execute_steps,
        "streaming_vae": False,
        "hardware": "reference",
        "device": "cuda",
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "policy_options": {},
    }


def decode(blob):
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Server:
    def __init__(self, args):
        from longwam.runtime.settings import normalize_settings
        from longwam.runtime.policy import initialize_policy
        from longwam.runtime.robots import make_adapter

        settings = normalize_settings(build_settings(args))
        t0 = time.time()
        self.handle, self.info = initialize_policy(settings)
        self.info.update(load_s=round(time.time() - t0, 1), action_horizon=settings.action_horizon,
                         execute_steps=settings.execute_steps, cameras=list(CAMERAS),
                         action_names=ACTION_NAMES, num_inference_steps=settings.num_inference_steps)
        self.adapter = make_adapter("yam", {"action_names": ACTION_NAMES})
        self.episode = 0
        self.next_step = 0
        log.info("model ready in %.1fs: %s", time.time() - t0, self.info)

    def warmup(self):
        """One dummy inference so the first real request doesn't pay CUDA/kernel init."""
        self.reset("warm up")
        rng = np.random.default_rng(0)
        obs = {"images": {c: rng.integers(0, 255, (240, 320, 3), np.uint8) for c in CAMERAS},
               "state": np.zeros(14, np.float32)}
        self.handle("observe", {"episode": self.episode, "step": 0, "observation": self.adapter.observation(obs)})
        t0 = time.time()
        self.handle("infer", {"episode": self.episode, "anchor": 0})
        log.info("warm-up inference %.2fs", time.time() - t0)

    def reset(self, instruction):
        self.episode += 1
        self.next_step = 0
        self.handle("reset", {"episode": self.episode, "instruction": instruction})
        return self.episode

    def serve_request(self, header, blobs):
        op = header.get("op")
        if op == "info":
            return {"ok": True, "info": self.info}
        if op == "reset":
            ep = self.reset(str(header["instruction"]))
            log.info("reset episode %d: %r", ep, header["instruction"])
            return {"ok": True, "episode": ep}
        if op != "step":
            raise ValueError(f"unknown op {op!r}")
        if int(header["episode"]) != self.episode:
            raise ValueError("stale episode id")
        t_obs = time.time()
        for o in header["obs"]:
            step = int(o["step"])
            if step != self.next_step:
                raise ValueError(f"expected step {self.next_step}, got {step}")
            raw = {"images": {c: decode(blobs[o["images"][c]]) for c in CAMERAS},
                   "state": np.asarray(o["state"], np.float32)}
            self.handle("observe", {"episode": self.episode, "step": step,
                                    "observation": self.adapter.observation(raw)})
            self.next_step += 1
        reply = {"ok": True, "observe_s": round(time.time() - t_obs, 3)}
        if header.get("infer"):
            t0 = time.time()
            out = self.handle("infer", {"episode": self.episode, "anchor": self.next_step - 1})
            reply.update(anchor=out["anchor"], actions=np.asarray(out["actions"], np.float32).tolist(),
                         infer_s=round(time.time() - t0, 3))
            log.info("ep %d anchor %d: chunk %s in %.2fs", self.episode, out["anchor"],
                     np.asarray(out["actions"]).shape, reply["infer_s"])
        return reply


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/yam")
    p.add_argument("--model_config", default=None,
                   help="Override config.yaml (the released YAM config lacks processor.train_transforms)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8780)
    p.add_argument("--action_horizon", type=int, default=32)
    p.add_argument("--execute_steps", type=int, default=24)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_warmup", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    server = Server(args)
    if not args.no_warmup:
        server.warmup()
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((args.host, args.port))
    lsock.listen(1)
    log.info("listening on %s:%d", args.host, args.port)
    while True:
        conn, addr = lsock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log.info("client %s connected", addr)
        try:
            while True:
                header, blobs = recv_msg(conn)
                try:
                    reply = server.serve_request(header, blobs)
                except Exception as exc:  # report to the client, keep serving
                    log.error("request failed: %s", traceback.format_exc())
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                send_msg(conn, reply)
        except ConnectionError:
            log.info("client %s disconnected", addr)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
