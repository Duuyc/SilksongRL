import asyncio
import json
import os
import sys

DEFAULT_CONFIG = {
    "transport": "socket_sync",  # "socket_sync", "socket_async"
    "algorithm": "ppo",          # "ppo", "dqn"
    "host": "localhost",
    "port": 8000,
}


def load_config(path: str = "server_config.json"):
    candidates = [path]
    script_dir_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if script_dir_config not in candidates:
        candidates.append(script_dir_config)

    config_path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
    if config_path is None:
        print(f"[Launcher] Config not found at {path}, using defaults.", flush=True)
        return DEFAULT_CONFIG.copy()

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[Launcher] Failed to parse config: {e}. Using defaults.", flush=True)
            return DEFAULT_CONFIG.copy()
    merged = DEFAULT_CONFIG.copy()
    merged.update(cfg)
    print(f"[Launcher] Loaded config: {config_path}", flush=True)
    return merged


def run_socket_sync(host: str, port: int):
    from socket_server import RLSocketServer

    print(f"[Launcher] Starting sync socket server on {host}:{port}", flush=True)
    server = RLSocketServer(host=host, port=port)
    server.start()


def run_socket_async(host: str, port: int):
    from socket_server_async import AsyncRLSocketServer

    print(f"[Launcher] Starting async socket server on {host}:{port}", flush=True)
    server = AsyncRLSocketServer(host=host, port=port)
    asyncio.run(server.start())


def main():
    cfg = load_config()
    transport = cfg.get("transport", "socket_sync").lower()
    algorithm = cfg.get("algorithm", "dqn").lower()
    host = cfg.get("host", "localhost")
    port = int(cfg.get("port", 8000))
    os.environ["SILKSONGRL_ALGORITHM"] = algorithm
    print(f"[Launcher] Algorithm: {algorithm}", flush=True)

    if transport == "socket_sync":
        run_socket_sync(host, port)
    elif transport == "socket_async":
        run_socket_async(host, port)
    else:
        print(f"[Launcher] Unknown transport '{transport}'. Expected one of socket_sync|socket_async.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

