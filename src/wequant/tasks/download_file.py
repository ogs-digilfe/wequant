"""管理対象データのダウンロードに関するTask。"""

from collections.abc import Callable
from typing import Protocol

from wequant.api import Client
from wequant.data_files import DOWNLOADABLE_FILES


class DownloadClient(Protocol):
    """ダウンロードTaskが利用するクライアントの契約。"""

    def download(self, filename: str) -> None:
        """指定されたファイルをダウンロードする。"""


def list_downloadable_files() -> tuple[str, ...]:
    """管理対象のファイル名を返す。"""
    return tuple(DOWNLOADABLE_FILES)


class DownloadFileTask:
    """同じクライアントを再利用してファイルを1件ずつダウンロードするTask。"""

    def __init__(
        self,
        client_factory: Callable[[], DownloadClient] = Client,
    ) -> None:
        self._client_factory = client_factory
        self._client: DownloadClient | None = None

    def __call__(self, filename: str) -> str:
        if self._client is None:
            self._client = self._client_factory()

        self._client.download(filename)
        return filename
