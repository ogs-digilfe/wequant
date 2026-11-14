"""Jupyterのホスト決定と実行Task。"""

from collections.abc import Callable

from wequant.jupyter import JupyterLabServer, get_lan_ip


def resolve_jupyter_host_task(hostname: str | None) -> str:
    return hostname if hostname is not None else get_lan_ip()


def run_jupyter_lab_task(*, port: int, hostname: str, on_ready: Callable[[str], None]) -> None:
    JupyterLabServer().run(port=port, hostname=hostname, on_ready=on_ready)
