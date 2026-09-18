"""Deliverサーバーから管理対象データを取得するFlow。"""

from dataclasses import dataclass
from typing import Callable, Iterable

from wequant.tasks.download_file import DownloadFileTask, list_downloadable_files


@dataclass(frozen=True)
class DownloadDataResult:
    """データダウンロードFlowの実行結果。"""

    downloaded_files: tuple[str, ...]


def download_data_flow(
    *,
    list_files: Callable[[], Iterable[str]] = list_downloadable_files,
    download_file: Callable[[str], str] | None = None,
) -> DownloadDataResult:
    """管理対象ファイルを順番にダウンロードする。"""
    download = download_file or DownloadFileTask()
    downloaded_files = tuple(download(filename) for filename in list_files())
    return DownloadDataResult(downloaded_files=downloaded_files)
