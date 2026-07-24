import os
from pathlib import Path
import sys
from unittest import TestCase
from unittest.mock import patch


LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
sys.path.insert(0, str(LIB_DIR))

from config import ConfigurationError, load_deliver_settings


class LoadDeliverSettingsTests(TestCase):
    def test_loads_settings_from_environment(self):
        environment = {
            "WEQUANT_DELIVER_BASE_URL": "https://deliver.example.invalid/",
            "WEQUANT_DELIVER_USERNAME": "example-user",
            "WEQUANT_DELIVER_PASSWORD": "example-password",
        }

        with patch.dict(os.environ, environment, clear=True):
            settings = load_deliver_settings(Path("/does/not/exist"))

        self.assertEqual(settings.base_url, "https://deliver.example.invalid")
        self.assertEqual(settings.username, "example-user")
        self.assertEqual(settings.password, "example-password")

    def test_reports_missing_variable_names_without_values(self):
        environment = {
            "WEQUANT_DELIVER_USERNAME": "example-user",
        }

        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(ConfigurationError) as raised:
                load_deliver_settings(Path("/does/not/exist"))

        message = str(raised.exception)
        self.assertIn("WEQUANT_DELIVER_BASE_URL", message)
        self.assertIn("WEQUANT_DELIVER_PASSWORD", message)
        self.assertNotIn("example-user", message)
