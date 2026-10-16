"""Focused tracking checks: no model, GPU, network or W&B dependency."""
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_tracking as tracking


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = MagicMock()
        self.run.id = "test-id"
        self.run.project = "test-project"
        self.run.summary = {}
        self.sdk = types.SimpleNamespace(
            init=MagicMock(return_value=self.run),
            util=types.SimpleNamespace(generate_id=lambda: "test-id"))
        self.env = patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.modules = patch.dict(sys.modules, {"wandb": self.sdk})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(setattr, tracking, "_run", None)

    def start(self, **kwargs):
        tracking.start({"cache": Path("/cache"), "api_key": "never-upload"},
                       Path(self.directory.name) / "results" / "experiment" / "run",
                       model="langflow", job_type="train", **kwargs)

    def test_rank_zero_only_and_disabled(self):
        self.start(rank=1)
        with patch.dict(os.environ, {"WANDB_MODE": "disabled"}):
            self.start()
        self.sdk.init.assert_not_called()

    def test_offline_defaults_config_and_same_step(self):
        @tracking.tracked
        def experiment():
            self.start()
            tracking.log({"train/ce": 3.0}, step=20)
            tracking.log({"val/ce_sc": 2.9}, step=20)
        experiment()
        options = self.sdk.init.call_args.kwargs
        self.assertEqual(options["mode"], "offline")
        self.assertNotIn("api_key", options["config"])
        self.assertEqual(options["config"]["cache"], "/cache")
        self.assertIn("/logs/experiment/run", options["dir"])
        self.assertEqual(self.run.log.call_count, 2)
        for call in self.run.log.call_args_list:
            self.assertEqual(call.args[0]["optimizer_step"], 20)
            self.assertNotIn("step", call.kwargs)
        self.run.finish.assert_called_once_with(exit_code=0)

    def test_eval_never_logs_text_and_failure_finishes(self):
        @tracking.tracked
        def experiment():
            self.start()
            tracking.log_evaluation({"validation": {
                "gen_ppl": 100.0, "generated_texts": ["private text"],
                "reference_texts": ["dataset text"], "dataset_indices": [1]}}, complete=False)
            raise ValueError("original failure")
        with self.assertRaisesRegex(ValueError, "original failure"):
            experiment()
        self.assertEqual(self.run.log.call_args.args[0], {"eval/validation/gen_ppl": 100.0})
        self.assertFalse(self.run.summary["complete"])
        self.run.finish.assert_called_once_with(exit_code=1)


if __name__ == "__main__":
    unittest.main()
