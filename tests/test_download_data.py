from unittest import TestCase
from unittest.mock import Mock

from wequant.data_files import DOWNLOADABLE_FILES
from wequant.flows.download_data import DownloadDataResult, download_data_flow
from wequant.tasks.download_file import DownloadFileTask, list_downloadable_files


class DownloadDataFlowTests(TestCase):
    def test_coordinates_tasks_and_returns_structured_result(self):
        download_file = Mock(side_effect=lambda filename: filename)

        result = download_data_flow(
            list_files=lambda: ("first.parquet", "second.parquet"),
            download_file=download_file,
        )

        self.assertEqual(
            result,
            DownloadDataResult(
                downloaded_files=("first.parquet", "second.parquet")
            ),
        )
        self.assertEqual(
            [call.args for call in download_file.call_args_list],
            [("first.parquet",), ("second.parquet",)],
        )


class DownloadFileTaskTests(TestCase):
    def test_lists_the_central_downloadable_files_definition(self):
        self.assertEqual(list_downloadable_files(), tuple(DOWNLOADABLE_FILES))

    def test_reuses_one_client_to_download_multiple_files(self):
        client = Mock()
        client_factory = Mock(return_value=client)
        task = DownloadFileTask(client_factory=client_factory)

        first_result = task("first.parquet")
        second_result = task("second.parquet")

        self.assertEqual(first_result, "first.parquet")
        self.assertEqual(second_result, "second.parquet")
        client_factory.assert_called_once_with()
        self.assertEqual(
            [call.args for call in client.download.call_args_list],
            [("first.parquet",), ("second.parquet",)],
        )
