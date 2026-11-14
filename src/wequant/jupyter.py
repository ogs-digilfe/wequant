"""Jupyter Labのプロセスと接続先を管理する。"""

import os
import re
from pathlib import Path
from queue import Empty, Queue
import secrets
import signal
import socket
import subprocess
import sys
from threading import Thread
import time
from collections.abc import Callable


def get_lan_ip() -> str:
    """パケットを送らず、既定経路で選ばれるIPv4を取得する。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            address = sock.getsockname()[0]
    except OSError as exc:
        raise ValueError("LAN IPを取得できません。--hostnameを指定してください。") from exc
    if address.startswith("127.") or address == "0.0.0.0":
        raise ValueError("LAN IPを取得できません。--hostnameを指定してください。")
    return address


class JupyterLabServer:
    """専用トークンで起動し、起動ログを確認してからURLを通知する。"""

    def run(self, *, port: int, hostname: str, on_ready: Callable[[str], None]) -> None:
        notebooks = Path(__file__).resolve().parents[2] / "notebooks"
        if not notebooks.is_dir():
            raise ValueError("リポジトリのnotebooks/が見つかりません。")
        token = secrets.token_urlsafe(32)
        env = os.environ.copy()
        env["JUPYTER_TOKEN"] = token
        command = [
            "uv", "run", "--offline", "--no-sync", "jupyter", "lab",
            "--ip=0.0.0.0", f"--port={port}", "--no-browser",
            "--ServerApp.port_retries=0", "--ServerApp.base_url=/",
            "--ServerApp.default_url=/lab", "--ServerApp.open_browser=False",
            "--ServerApp.custom_display_url=", "--ServerApp.log_level=INFO",
        ]
        process = subprocess.Popen(
            command, cwd=notebooks, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        lines: Queue[str | None] = Queue()

        def read_output() -> None:
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        reader = Thread(target=read_output, daemon=True)
        reader.start()
        ready = False
        deadline = time.monotonic() + 60
        try:
            while True:
                if not ready and time.monotonic() >= deadline:
                    raise RuntimeError("Jupyter Labの起動確認がタイムアウトしました。")
                try:
                    line = lines.get(timeout=0.2)
                except Empty:
                    continue
                if line is None:
                    break
                # トークン付き起動URLはCLIからのみ通知する。
                if not ready and re.search(
                    rf"http://[^\s/]+:{port}/lab\?token=(?:{re.escape(token)}|\.\.\.)(?:\s|$)", line,
                ):
                    on_ready(f"http://{hostname}:{port}/lab?token={token}")
                    ready = True
                sys.stderr.write(line.replace(token, "[redacted]"))
            code = process.wait()
            if code != 0 or not ready:
                raise RuntimeError(
                    f"Jupyter Labを正常に実行できませんでした（終了コード {code}）。"
                    f"ポート {port} の使用状況と起動ログを確認してください。"
                )
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            reader.join(timeout=1)
            process.stdout.close()
