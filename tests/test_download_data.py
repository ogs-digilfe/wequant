from unittest import TestCase
from unittest.mock import call, patch

from wequant.commands.download_data import DOWNLOADABLE_FILES, download_data
from wequant.data_files import DOWNLOADABLE_FILES as CENTRAL_DOWNLOADABLE_FILES
from wequant.data_processing import DOWNLOADABLE_FILES as PROCESSING_DOWNLOADABLE_FILES


class DownloadDataTests(TestCase):
    def test_uses_central_downloadable_files_definition(self):
        self.assertIs(DOWNLOADABLE_FILES, CENTRAL_DOWNLOADABLE_FILES)
        self.assertIs(PROCESSING_DOWNLOADABLE_FILES, CENTRAL_DOWNLOADABLE_FILES)

    @patch("wequant.commands.download_data.Client")
    def test_downloads_each_configured_file(self, client_class):
        client = client_class.return_value

        download_data()

        client_class.assert_called_once_with()
        self.assertEqual(
            client.download.call_args_list,
            [call(filename) for filename in DOWNLOADABLE_FILES],
        )
