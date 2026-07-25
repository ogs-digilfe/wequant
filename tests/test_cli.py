from unittest import TestCase
from unittest.mock import patch

from typer.testing import CliRunner

from wequant.cli import app


class CliTests(TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_get_app_name(self):
        result = self.runner.invoke(app, ["get-app-name"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout.strip(), "wequant")

    def test_describe(self):
        result = self.runner.invoke(app, ["describe"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout.strip(), "analysis tool for stock market")

    @patch("wequant.cli.download_data")
    def test_dl_pq_delegates_without_network_access(self, download_data):
        result = self.runner.invoke(app, ["dl-pq"])

        self.assertEqual(result.exit_code, 0)
        download_data.assert_called_once_with()
        self.assertIn("Download complete.", result.stdout)
