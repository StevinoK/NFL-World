"""Open a dashboard in the browser, starting its Streamlit server first if it isn't already running.

The desktop shortcuts call this with pythonw, so there is no console window; each server keeps running in the
background after its browser tab closes, and clicking the shortcut again reuses it. Server output goes to
nflsim/outputs/dashboard_logs.

    pythonw nflsim/dashboard/launch.py hub
"""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Name: (script, port); ports match .claude/launch.json.
DASHBOARDS = {
    "hub": ("nflsim/dashboard/hub.py", 8516),
    "simulator": ("nflsim/dashboard/app.py", 8517),
    "dfs": ("nflsim/dfs/dashboard.py", 8518),
    "ridge": ("nflsim/ridge/explorer.py", 8519),
    "players": ("player_value/explorer.py", 8520),
}
STARTUP_SECONDS = 180


def listening(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(0.5)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def ensure_running(name: str) -> str:
    script, port = DASHBOARDS[name]
    if not listening(port):
        logs = REPO / "nflsim/outputs/dashboard_logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{name}.log"
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
        python = Path(sys.executable).with_name("python.exe")
        with open(log_path, "w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [str(python), "-m", "streamlit", "run", str(REPO / script), "--server.port", str(port), "--server.address", "localhost",
                 "--server.headless", "true", "--server.fileWatcherType", "none"],
                cwd=REPO, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        deadline = time.monotonic() + STARTUP_SECONDS
        while not listening(port):
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"The {name} dashboard did not start. See {log_path}")
            time.sleep(0.5)
    return f"http://localhost:{port}"


def main():
    try:
        webbrowser.open(ensure_running(sys.argv[1]))
    except (IndexError, KeyError, OSError, RuntimeError) as error:
        ctypes.windll.user32.MessageBoxW(None, str(error), "NFL dashboard", 0x10)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
