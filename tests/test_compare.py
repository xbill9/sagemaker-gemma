"""compare.py's http mode: same request body and fields as sm.invoke, logs from docker."""

import json
import sys
import threading
import unittest
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare

SEEN: list[dict] = []


class FakeVllm(BaseHTTPRequestHandler):
    def do_POST(self):
        SEEN.append(
            {"path": self.path, "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))}
        )
        out = json.dumps(
            {
                "model": "m",
                "choices": [{"message": {"content": "948"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 16},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


class HttpModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeVllm)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        SEEN.clear()

    def test_invoke_sends_sm_invoke_body_and_returns_its_fields(self):
        ep = {"name": "x", "url": self.url}
        out = compare.invoke(ep, "hello", 16, fixed=True)
        self.assertEqual(SEEN[0]["path"], "/v1/chat/completions")
        self.assertEqual(
            SEEN[0]["body"],
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        )
        self.assertEqual(out["text"], "948")
        self.assertEqual(out["completion_tokens"], 16)
        self.assertIn("wall_seconds", out)

    def test_load_facts_and_throughput_read_docker_logs(self):
        log = "\n".join(
            [
                "INFO 09-29 18:00:06 [model_runner.py:428] Model loading took 7.26 GiB memory and 51.9 seconds",
                "INFO 09-29 18:02:02 [kv_cache_utils.py:2395] GPU KV cache size: 929,454 tokens, "
                "Maximum concurrency for 8,192 tokens per request: 113.46x",
                "INFO 09-29 18:04:00 [loggers.py:315] Avg generation throughput: 100.5 tokens/s",
                "INFO 09-29 18:04:10 [loggers.py:315] Avg generation throughput: 400.0 tokens/s",
                "INFO 09-29 18:09:00 [loggers.py:315] Avg generation throughput: 999.0 tokens/s",
            ]
        )
        ep = {"name": "x", "url": self.url, "container": "vllm"}
        with mock.patch.object(compare, "docker_log", lambda c: log):
            facts = compare.load_facts(ep)
            year = datetime.now(UTC).year
            t0 = int(datetime(year, 9, 29, 18, 3, tzinfo=UTC).timestamp() * 1000)
            t1 = int(datetime(year, 9, 29, 18, 5, tzinfo=UTC).timestamp() * 1000)
            srv = compare.server_throughput(ep, t0, t1)
        self.assertEqual(facts["weights_gib"], "7.26")
        self.assertEqual(facts["kv_cache_tokens"], "929,454")
        self.assertEqual(srv, {"lines": 2, "peak_tokens_per_second": 400.0})

    def test_load_skips_the_aws_login_refresh_for_http(self):
        ep = {"name": "x", "url": self.url}
        with (
            mock.patch.object(compare, "CONCURRENCY", [1]),
            mock.patch.object(compare, "LOAD_REPEATS", 1),
            mock.patch.object(compare.time, "sleep", lambda s: None),
            mock.patch.object(compare.sm, "aws", side_effect=AssertionError("aws called")),
        ):
            out = compare.load([ep])
        self.assertIn("1", out["x"])


if __name__ == "__main__":
    unittest.main()
