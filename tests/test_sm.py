"""Offline tests for sm.py: subprocess.run is replaced, so no aws CLI or credentials are used."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sm


class FakeAws:
    """Records aws CLI invocations and answers them from a handler."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def __call__(self, cmd, capture_output, text, env, check):
        self.calls.append((cmd, env))
        rc, out, err = self.handler(cmd)
        return subprocess.CompletedProcess(cmd, rc, out, err)


def ok(obj):
    return 0, json.dumps(obj), ""


class SmTests(unittest.TestCase):
    def fake(self, handler):
        f = FakeAws(handler)
        patcher = mock.patch.object(sm.subprocess, "run", f)
        patcher.start()
        self.addCleanup(patcher.stop)
        return f

    def test_aws_adds_json_output_and_region_and_drops_static_keys(self):
        self.enterContext(mock.patch.dict(os.environ, {"AWS_SESSION_TOKEN": "stale"}))
        f = self.fake(lambda cmd: ok({"Account": "123"}))
        sm.aws("sts", "get-caller-identity", region="us-west-2")
        cmd, env = f.calls[0]
        self.assertEqual(
            cmd,
            [
                "aws",
                "sts",
                "get-caller-identity",
                "--output",
                "json",
                "--region",
                "us-west-2",
            ],
        )
        self.assertNotIn("AWS_SESSION_TOKEN", env)
        self.assertEqual(env["AWS_PAGER"], "")

    def test_expired_session_names_aws_login(self):
        self.fake(lambda cmd: (255, "", "Your session has expired. Please reauthenticate"))
        with self.assertRaisesRegex(sm.AwsError, "aws login"):
            sm.aws("sts", "get-caller-identity")

    def test_latest_vllm_image_picks_highest_version_not_newest_push(self):
        images = [
            {
                "tags": ["0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1"],
                "pushed": "2026-09-23",
            },
            {
                "tags": ["0.27.1-gpu-py312-cu130-ubuntu22.04-sagemaker-v1.3"],
                "pushed": "2026-09-24",
            },
            {
                "tags": ["0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1-soci"],
                "pushed": "2026-09-25",
            },
            {"tags": ["omni-sagemaker-cuda-v1.7"], "pushed": "2026-09-25"},
            {
                "tags": ["0.30-gpu-py312-cu130-ubuntu24.04-sagemaker-v1"],
                "pushed": "2026-09-23",
            },
            {"tags": None, "pushed": "2026-09-25"},
        ]
        self.fake(lambda cmd: ok(images))
        uri = sm.latest_vllm_image("us-east-1")
        self.assertTrue(uri.endswith("vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1"))

    def test_status_not_found(self):
        self.fake(
            lambda cmd: (
                254,
                "",
                "An error occurred (ValidationException): Could not find endpoint",
            )
        )
        self.assertEqual(sm.status("nope"), {"endpoint": "nope", "status": "NotFound"})

    def test_list_endpoints_counts_in_code(self):
        eps = [
            {"name": "gemma-a", "status": "InService"},
            {"name": "gemma-b", "status": "Creating"},
            {"name": "gemma-c", "status": "InService"},
        ]
        self.fake(lambda cmd: ok(eps))
        out = sm.list_endpoints("gemma")
        self.assertEqual(out["count"], 3)
        self.assertEqual(out["by_status"], {"InService": 2, "Creating": 1})
        self.assertEqual(out["filter"]["name_contains"], "gemma")

    def test_invoke_sends_chat_body_and_parses_reply(self):
        sent = {}

        def handler(cmd):
            body_path = cmd[cmd.index("--body") + 1].removeprefix("fileb://")
            with open(body_path) as fh:
                sent.update(json.load(fh))
            out_path = cmd[cmd.index("--output") - 1]
            reply = {
                "model": "google/gemma-4-E2B-it",
                "choices": [{"message": {"content": "Hi."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }
            with open(out_path, "w") as fh:
                json.dump(reply, fh)
            return 0, json.dumps({"ContentType": "application/json"}), ""

        self.fake(handler)
        out = sm.invoke("hello", max_tokens=16, system="be brief")
        self.assertEqual(
            sent["messages"],
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertEqual(sent["max_tokens"], 16)
        self.assertEqual(out["text"], "Hi.")
        self.assertEqual(out["completion_tokens"], 3)

    def test_deploy_issues_three_create_calls(self):
        self.enterContext(mock.patch.object(sm, "ensure_role", lambda: "arn:aws:iam::1:role/r"))
        f = self.fake(lambda cmd: ok({}))
        out = sm.deploy(endpoint_name="gemma-x", image_uri="img:1", model_id="google/gemma-4-E4B-it")
        verbs = [c[0][2] for c in f.calls]
        self.assertEqual(verbs, ["create-model", "create-endpoint-config", "create-endpoint"])
        container = json.loads(f.calls[0][0][f.calls[0][0].index("--primary-container") + 1])
        self.assertEqual(container["Environment"]["SM_VLLM_MODEL"], "google/gemma-4-E4B-it")
        self.assertEqual(out["status"], "Creating")

    def test_destroy_tolerates_missing_pieces(self):
        def handler(cmd):
            if cmd[2] == "delete-endpoint":
                return 254, "", "Could not find endpoint"
            return 0, "", ""

        self.fake(handler)
        out = sm.destroy("gemma-x")
        self.assertEqual(
            out["results"],
            {
                "endpoint": "not found",
                "endpoint-config": "deleted",
                "model": "deleted",
            },
        )

    def test_deploy_with_instance_pools_sets_priorities(self):
        self.enterContext(mock.patch.object(sm, "ensure_role", lambda: "arn:aws:iam::1:role/r"))
        f = self.fake(lambda cmd: ok({}))
        sm.deploy(image_uri="img:1", instance_pools=["ml.g6.xlarge", "ml.g6.2xlarge"])
        cmd = f.calls[1][0]
        variant = json.loads(cmd[cmd.index("--production-variants") + 1])[0]
        self.assertNotIn("InstanceType", variant)
        self.assertEqual(
            variant["InstancePools"],
            [
                {"InstanceType": "ml.g6.xlarge", "Priority": 1},
                {"InstanceType": "ml.g6.2xlarge", "Priority": 2},
            ],
        )

    def test_destroy_stops_when_endpoint_is_still_creating(self):
        calls = []

        def handler(cmd):
            calls.append(cmd[2])
            return 254, "", 'ValidationException: Cannot update in-progress endpoint "arn:..."'

        self.fake(handler)
        out = sm.destroy("gemma-x")
        self.assertIn("still Creating", out["results"]["endpoint"])
        self.assertEqual(calls, ["delete-endpoint"])


if __name__ == "__main__":
    unittest.main()
