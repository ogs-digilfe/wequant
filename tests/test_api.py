from unittest import TestCase
from unittest.mock import Mock, patch

from wequant.api import Client
from wequant.config import DeliverSettings


class ClientAuthenticationTests(TestCase):
    @patch("wequant.api.requests.post")
    @patch("wequant.api.load_deliver_settings")
    def test_initializes_authorization_header(self, load_settings, post):
        load_settings.return_value = DeliverSettings(
            base_url="https://deliver.example.invalid",
            username="example-user",
            password="example-password",
        )
        response = Mock(status_code=200)
        response.json.return_value = {"access_token": "example-token"}
        post.return_value = response

        client = Client()

        post.assert_called_once_with(
            "https://deliver.example.invalid/token",
            data={
                "username": "example-user",
                "password": "example-password",
            },
        )
        self.assertEqual(
            client.headers,
            {"Authorization": "Bearer example-token"},
        )

    @patch("wequant.api.requests.post")
    @patch("wequant.api.load_deliver_settings")
    def test_raises_on_authentication_failure(self, load_settings, post):
        load_settings.return_value = DeliverSettings(
            base_url="https://deliver.example.invalid",
            username="example-user",
            password="example-password",
        )
        post.return_value = Mock(status_code=401)

        with self.assertRaisesRegex(ValueError, "status_code=401"):
            Client()
