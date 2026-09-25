"""Offline tests for the MCP server: tool catalog, annotations, and no-AWS tools.

No AWS and no network: tools that would call the aws CLI are exercised with
subprocess.run replaced.
"""

import asyncio
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server
import sm


def run(coro):
    return asyncio.run(coro)


class ToolCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {tool.name: tool for tool in run(server.mcp.list_tools())}

    def test_catalog(self):
        expected = {
            "get_help",
            "get_deployment_config",
            "find_vllm_image",
            "check_quotas",
            "deploy_endpoint",
            "get_endpoint_status",
            "list_endpoints",
            "get_endpoint_logs",
            "verify_model_health",
            "query_model",
            "delete_endpoint",
        }
        self.assertEqual(set(self.tools), expected)

    def test_annotations(self):
        destructive = {name for name, tool in self.tools.items() if tool.annotations.destructive_hint}
        self.assertEqual(destructive, {"delete_endpoint"})
        writes = {name for name, tool in self.tools.items() if not tool.annotations.read_only_hint}
        self.assertEqual(writes, {"deploy_endpoint", "delete_endpoint"})
        for name, tool in self.tools.items():
            self.assertTrue(tool.title, name)
            self.assertTrue(tool.description, name)


class NoAwsToolTests(unittest.TestCase):
    def setUp(self):
        # Any aws CLI call from these tools is a bug.
        patcher = mock.patch.object(sm.subprocess, "run", side_effect=AssertionError("aws called"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_help_shows_config(self):
        text = run(server.get_help())
        self.assertIn(sm.ENDPOINT_NAME, text)
        self.assertIn(sm.MODEL_ID, text)

    def test_deployment_config_has_the_three_create_calls_and_pools(self):
        text = run(server.get_deployment_config(model_id="google/gemma-4-E2B-it-qat-w4a16-ct"))
        for verb in ("create-model", "create-endpoint-config", "create-endpoint "):
            self.assertIn(f"aws sagemaker {verb}", text)
        self.assertIn('"SM_VLLM_MODEL\\":\\"google/gemma-4-E2B-it-qat-w4a16-ct', text)
        self.assertIn('"InstancePools"', text)


class AwsBackedToolTests(unittest.TestCase):
    def fake(self, handler):
        def run_(cmd, capture_output, text, env, check):
            rc, out, err = handler(cmd)
            return subprocess.CompletedProcess(cmd, rc, out, err)

        patcher = mock.patch.object(sm.subprocess, "run", run_)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_status_reports_placed_pool_type(self):
        described = {
            "EndpointStatus": "InService",
            "CreationTime": "t0",
            "LastModifiedTime": "t1",
            "ProductionVariants": [
                {
                    "CurrentInstanceCount": 1,
                    "InstancePools": [
                        {"InstanceType": "ml.g6.xlarge", "CurrentInstanceCount": 1},
                        {"InstanceType": "ml.g6.2xlarge", "CurrentInstanceCount": 0},
                    ],
                }
            ],
        }
        self.fake(lambda cmd: (0, json.dumps(described), ""))
        text = run(server.get_endpoint_status())
        self.assertIn("InService", text)
        self.assertIn("`ml.g6.xlarge`", text)
        self.assertNotIn("ml.g6.2xlarge", text)

    def test_aws_failure_becomes_error_text(self):
        self.fake(lambda cmd: (255, "", "Your session has expired."))
        text = run(server.get_endpoint_status())
        self.assertTrue(text.startswith("❌"))
        self.assertIn("aws login", text)

    def test_quota_grid_counts_regions_in_code(self):
        def handler(cmd):
            region = cmd[cmd.index("--region") + 1]
            if region == "us-west-1":
                return 0, "[]", ""
            rows = [["ml.g6.xlarge for endpoint usage", 1.0], ["ml.g5.xlarge for endpoint usage", 2.0]]
            return 0, json.dumps(rows), ""

        self.fake(handler)
        text = run(server.check_quotas(["ml.g6.xlarge"]))
        self.assertIn("| `ml.g6.xlarge` | 1 | 1 | - | 1 |", text)
        self.assertIn("us-east-1, us-east-2, us-west-2.", text)


if __name__ == "__main__":
    unittest.main()
