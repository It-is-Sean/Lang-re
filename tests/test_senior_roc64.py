"""Small regressions for the opt-in full ROC64 profile; no training run."""
import unittest
from types import SimpleNamespace

import torch

from training.senior_roc64 import ema_decay, quantiles
from training.trainer import build_parser


class SeniorROC64Tests(unittest.TestCase):
    def test_adaptive_ema(self):
        self.assertAlmostEqual(ema_decay(0.9999, 1), 2 / 11)
        self.assertAlmostEqual(ema_decay(0.9999, 18200), 18201 / 18210)
        self.assertEqual(ema_decay(0.9999, 1000000), 0.9999)

    def test_quantiles_keep_rank_strata(self):
        pieces = []
        for rank in range(4):
            torch.manual_seed(1)
            ctx = SimpleNamespace(rank=rank, world_size=4, device=torch.device("cpu"))
            pieces.append(quantiles(512, ctx))
        q = torch.cat(pieces)
        self.assertTrue(torch.equal((q * 512).long(), torch.arange(512)))
        self.assertTrue(((q >= 1e-5) & (q <= 1-1e-5)).all())

    def test_epoch_checkpoint_setting(self):
        args = build_parser().parse_args([
            "--training_profile", "senior-roc64", "--epochs", "100",
            "--global_batch_size", "512", "--checkpoint_every", "0", "--save_every_fraction", "1"])
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.save_every_fraction, 1)
        self.assertEqual(args.checkpoint_every, 0)
        self.assertEqual(93253 // args.global_batch_size, 182)


if __name__ == "__main__":
    unittest.main()
