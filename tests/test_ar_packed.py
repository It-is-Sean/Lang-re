"""Focused regressions for BOS-only AR generation and next-token supervision."""

import unittest
import torch

from eval_ar import generate_continuations
from train_ar import causal_loss


class ScriptedModel(torch.nn.Module):
    def forward(self, ids):
        logits = torch.full((*ids.shape, 8), -100.0)
        # BOS=EOS=1: an internal EOS must not stop packed sampling.
        logits[:, -1, 1 if ids.shape[1] == 1 else 7] = 100.0
        return logits


class PackedARTests(unittest.TestCase):
    def test_internal_eos_kept_and_last_token_not_forced(self):
        rows = generate_continuations(
            ScriptedModel(), torch.ones(3, 1, dtype=torch.long),
            1, 6, 2, 1.0, 1.0, 0, 42, torch.device('cpu'), stop_at_eos=False)
        self.assertEqual(rows, [[1, 1, 7, 7, 7, 7]] * 3)

    def test_bos_predicts_first_content_and_terminal_eos_is_scored(self):
        ids = torch.tensor([[1, 2, 3, 1]])
        logits = torch.zeros(1, 4, 8, requires_grad=True)
        loss, count = causal_loss(logits, ids, torch.tensor([4]))
        self.assertEqual(count, 3)
        loss.backward()
        self.assertLess(logits.grad[0, 0, 2].item(), 0)
        self.assertLess(logits.grad[0, 2, 1].item(), 0)
        self.assertEqual(logits.grad[0, 3].abs().sum().item(), 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
