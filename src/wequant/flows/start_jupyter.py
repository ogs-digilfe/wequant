"""Jupyter Lab起動Flow。"""

from collections.abc import Callable

from wequant.tasks.start_jupyter import resolve_jupyter_host_task, run_jupyter_lab_task


def start_jupyter_flow(
    *, port: int = 8888, hostname: str | None = None, on_ready: Callable[[str], None],
) -> None:
    host = resolve_jupyter_host_task(hostname)
    run_jupyter_lab_task(port=port, hostname=host, on_ready=on_ready)
