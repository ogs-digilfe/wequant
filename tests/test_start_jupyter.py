from io import StringIO
import subprocess
from unittest import TestCase
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from wequant.cli import app
from wequant.flows.start_jupyter import start_jupyter_flow
from wequant.jupyter import JupyterLabServer, get_lan_ip
from wequant.tasks.start_jupyter import resolve_jupyter_host_task


class StartJupyterTests(TestCase):
    @patch("wequant.cli.start_jupyter_flow")
    def test_cli_defaults_and_url(self, flow):
        flow.side_effect = lambda **kw: kw["on_ready"]("http://lan:8888/lab?token=test")
        result = CliRunner().invoke(app, ["start-jupyter"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.stdout.strip(), "http://lan:8888/lab?token=test")
        self.assertEqual(flow.call_args.kwargs["port"], 8888)
        self.assertIsNone(flow.call_args.kwargs["hostname"])

    @patch("wequant.cli.start_jupyter_flow")
    def test_cli_options_and_error(self, flow):
        flow.side_effect = RuntimeError("port in use")
        result = CliRunner().invoke(app, ["start-jupyter", "--port", "8890", "--hostname", "my-server"])
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("port in use", result.stderr)
        self.assertEqual(flow.call_args.kwargs["hostname"], "my-server")
        self.assertEqual(flow.call_args.kwargs["port"], 8890)

    @patch("wequant.cli.start_jupyter_flow")
    def test_invalid_options(self, flow):
        for args in (["--port", "0"], ["--port", "65536"], ["--hostname", "http://host"]):
            with self.subTest(args=args):
                self.assertEqual(CliRunner().invoke(app, ["start-jupyter", *args]).exit_code, 2)
        flow.assert_not_called()

    @patch("wequant.flows.start_jupyter.run_jupyter_lab_task")
    @patch("wequant.flows.start_jupyter.resolve_jupyter_host_task", return_value="192.168.1.5")
    def test_flow(self, resolve, run):
        callback = Mock()
        start_jupyter_flow(on_ready=callback)
        resolve.assert_called_once_with(None)
        run.assert_called_once_with(port=8888, hostname="192.168.1.5", on_ready=callback)

    @patch("wequant.tasks.start_jupyter.get_lan_ip", return_value="192.168.1.5")
    def test_hostname_bypasses_detection(self, detect):
        self.assertEqual(resolve_jupyter_host_task("server"), "server")
        detect.assert_not_called()
        self.assertEqual(resolve_jupyter_host_task(None), "192.168.1.5")

    @patch("wequant.jupyter.socket.socket")
    def test_lan_detection(self, socket_factory):
        sock = socket_factory.return_value.__enter__.return_value
        sock.getsockname.return_value = ("192.168.1.5", 12345)
        self.assertEqual(get_lan_ip(), "192.168.1.5")
        sock.send.assert_not_called()
        sock.getsockname.return_value = ("127.0.0.1", 12345)
        with self.assertRaises(ValueError):
            get_lan_ip()

    @patch("wequant.jupyter.secrets.token_urlsafe", return_value="test-token")
    @patch("wequant.jupyter.subprocess.Popen")
    def test_process_ready_and_fixed_port(self, popen, token):
        process = popen.return_value
        process.stdout = StringIO("http://localhost:8890/lab?token=...\n")
        process.wait.return_value = 0
        process.poll.return_value = 0
        callback = Mock()
        with patch("wequant.jupyter.sys.stderr", new_callable=StringIO) as log:
            JupyterLabServer().run(port=8890, hostname="my-server", on_ready=callback)
            self.assertNotIn("test-token", log.getvalue())
        callback.assert_called_once_with("http://my-server:8890/lab?token=test-token")
        args, kwargs = popen.call_args
        self.assertIn("--ServerApp.port_retries=0", args[0])
        self.assertIn("--port=8890", args[0])
        self.assertEqual(kwargs["cwd"].name, "notebooks")
        self.assertEqual(kwargs["env"]["JUPYTER_TOKEN"], "test-token")

    @patch("wequant.jupyter.subprocess.Popen")
    def test_failed_start_does_not_print_url(self, popen):
        process = popen.return_value
        process.stdout = StringIO("port is already in use\n")
        process.wait.return_value = 1
        process.poll.return_value = 1
        callback = Mock()
        with patch("wequant.jupyter.sys.stderr", new_callable=StringIO):
            with self.assertRaisesRegex(RuntimeError, "8888"):
                JupyterLabServer().run(port=8888, hostname="server", on_ready=callback)
        callback.assert_not_called()

    @patch("wequant.jupyter.os.killpg")
    @patch("wequant.jupyter.subprocess.Popen")
    def test_interrupt_stops_child(self, popen, killpg):
        process = popen.return_value
        process.stdout = StringIO("")
        process.poll.return_value = None
        with patch("wequant.jupyter.Queue.get", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                JupyterLabServer().run(port=8888, hostname="server", on_ready=Mock())
        killpg.assert_called_once()
        process.wait.assert_called_with(timeout=5)

    @patch("wequant.jupyter.os.killpg")
    @patch("wequant.jupyter.subprocess.Popen")
    def test_start_timeout_kills_unresponsive_child(self, popen, killpg):
        process = popen.return_value
        process.stdout = StringIO("")
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("jupyter", 5), 0]
        callback = Mock()
        with patch("wequant.jupyter.time.monotonic", side_effect=[0, 61]):
            with self.assertRaisesRegex(RuntimeError, "タイムアウト"):
                JupyterLabServer().run(port=8888, hostname="server", on_ready=callback)
        callback.assert_not_called()
        self.assertEqual(killpg.call_count, 2)
