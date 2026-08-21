# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import argparse
import os


FRAMEWORK_DETECTORS = (
    ("rynnbrain", "starVLA.model.framework.RynnBrainOFT", "RynnBrain_OFT"),
    ("gr00t", "starVLA.model.framework.QwenGR00T", "Qwen_GR00T"),
    ("qwenpi", "starVLA.model.framework.QwenPI", "Qwen_PI"),
    ("qwen_pi", "starVLA.model.framework.QwenPI", "Qwen_PI"),
    ("qwenfast", "starVLA.model.framework.QwenFast", "Qwenvl_Fast"),
    ("qwenoft", "starVLA.model.framework.QwenOFT_memory", "Qwenvl_OFT"),
    ("oft", "starVLA.model.framework.QwenOFT_memory", "Qwenvl_OFT"),
)


def resolve_framework_cls(ckpt_path: str):
    ckpt_path_lower = ckpt_path.lower()
    for pattern, module_name, class_name in FRAMEWORK_DETECTORS:
        if pattern in ckpt_path_lower:
            logging.info("Resolved framework %s from checkpoint path pattern `%s`", class_name, pattern)
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)

    logging.info("No checkpoint path pattern matched; falling back to baseframework")
    from starVLA.model.framework.base_framework import baseframework

    return baseframework


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    logging.info("Loading framework class for checkpoint: %s", args.ckpt_path)
    framework_cls = resolve_framework_cls(args.ckpt_path)
    logging.info("Loading model checkpoint")
    vla = framework_cls.from_pretrained(args.ckpt_path, base_vlm_path=args.base_vlm_path)

    logging.info("Importing torch")
    import torch

    if args.use_bf16: # False
        logging.info("Casting model to bfloat16")
        vla = vla.to(torch.bfloat16)
    logging.info("Moving model to CUDA")
    vla = vla.to("cuda").eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--base_vlm_path", type=str, default=None)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout" , type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    debug_value = os.getenv("DEBUG", "").strip().lower()
    if debug_value in {"1", "true", "yes", "y"}:
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)