from unittest import TestCase
from unittest.mock import call, patch

from wequant.commands.download_data import DOWNLOADABLE_FILES, download_data


class DownloadDataTests(TestCase):
    @patch("wequant.commands.download_data.Client")
    def test_downloads_each_configured_file(self, client_class):
        client = client_class.return_value

        download_data()

        client_class.assert_called_once_with()
        self.assertEqual(
            client.download.call_args_list,
            [call(filename) for filename in DOWNLOADABLE_FILES],
        )
