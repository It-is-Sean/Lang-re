"""Small CPU regressions for packing, gradient flow, sampling and validation."""

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from langflow import LangFlow, LangFlowConfig
from training.data import PROTOCOL, PackedDataset, pack_documents, read_rocstories, read_tinystories
from training.objective import LangFlowTrainingObjective, validate
from training.runtime import DistributedContext, latest_checkpoint


def tiny_model():
    return LangFlow(LangFlowConfig(
        vocab_size=32, hidden_size=16, cond_dim=8, n_blocks=2, n_heads=2,
        model_length=8, dropout=0.0,
    ))


class PaperPackingTests(unittest.TestCase):
    def test_document_eos_and_block_boundaries_without_document_truncation(self):
        blocks, dropped = pack_documents([[10, 11], list(range(12, 21))], 6, 1, 2)
        np.testing.assert_array_equal(blocks, [
            [1, 10, 11, 2, 12, 2], [1, 13, 14, 15, 16, 2], [1, 17, 18, 19, 20, 2],
        ])
        self.assertEqual(dropped, 1)  # Last document EOS is the incomplete tail.

    def test_split_before_packing_matches_original_rocstories_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stories.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[f"sentence{i}" for i in range(1, 6)])
                writer.writeheader()
                for row in range(20):
                    writer.writerow({f"sentence{i}": f"story{row} part{i}" for i in range(1, 6)})
            stories, indices = read_rocstories(path, 42, 0.2)
            expected = list(range(20))
            random.Random(42).shuffle(expected)
            self.assertEqual(indices["validation"], expected[:4])
            self.assertEqual(indices["train"], expected[4:])
            self.assertFalse(set(indices["train"]) & set(indices["validation"]))
            self.assertTrue(stories["train"][0].endswith("part5"))

    def test_tinystories_documents_and_unterminated_last_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stories.txt"
            path.write_text("one\nline\n<|endoftext|>\ntwo<|endoftext|>three")
            self.assertEqual(list(read_tinystories(path)), ["one\nline", "two", "three"])

    def test_cache_rejects_legacy_and_truncated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = {"fingerprint": {"protocol": PROTOCOL, "sequence_length": 8, "dtype": "uint16"},
                        "splits": {"train": {"blocks": 2}}}
            (path / "metadata.json").write_text(json.dumps(metadata))
            np.zeros(15, dtype=np.uint16).tofile(path / "train.bin")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                PackedDataset(path, "train")
            (path / "metadata.json").write_text(json.dumps({"fingerprint": {}}))
            with self.assertRaisesRegex(ValueError, "not a paper-packed"):
                PackedDataset(path, "train")

    def test_all_tokens_including_special_tokens_contribute_ce_and_gradients(self):
        model = tiny_model()
        objective = LangFlowTrainingObjective(model, self_condition_probability=1.0)
        ids = torch.tensor([[1, 10, 2, 11, 12, 13, 14, 2], [1, 12, 2, 11, 13, 14, 15, 2]])
        q = torch.tensor([0.3, 0.8])
        torch.manual_seed(5)
        loss, ce, _ = objective(ids, q, 0.0)
        self.assertAlmostEqual(ce.item(), np.log(32), places=5)
        loss.backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        self.assertEqual(missing, [])
        self.assertTrue(torch.isfinite(loss))

    def test_sampler_keeps_full_block_and_does_not_force_terminal_eos(self):
        model = tiny_model().eval()
        seen_lengths = []

        def prediction(noisy_embeds, **kwargs):
            batch, length, _ = noisy_embeds.shape
            seen_lengths.append(length)
            logits = torch.full((batch, length, 32), -20.)
            logits[:, :, 7] = 20.
            logits[:, 2, 2] = 40.  # An internal EOS must not stop sampling.
            return logits

        with patch.object(model, "forward", side_effect=prediction):
            ids = model.generate_samples(num_samples=2, seq_length=8, num_steps=4)
        self.assertEqual(seen_lengths, [8, 8, 8, 8])
        self.assertEqual(tuple(ids.shape), (2, 8))
        self.assertTrue((ids[:, -1] == 7).all())
        self.assertTrue((ids[:, 2] == 2).all())

    def test_validation_is_independent_of_learned_proposal_and_preserves_rng(self):
        model = tiny_model().eval()
        data = torch.randint(0, 32, (5, 8))
        loader = DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(99))
        ctx = DistributedContext(0, 0, 1, torch.device("cpu"))
        before = torch.get_rng_state().clone()
        first = validate(model, loader, ctx, "fp32", 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with torch.no_grad():
            model.proposal.loc.add_(5)
            model.proposal.scale.mul_(2)
        second = validate(model, loader, ctx, "fp32", 0)
        self.assertEqual(first, second)

    def test_latest_checkpoint_uses_step_and_prefers_final(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ["step-0000002.pt", "step-0000010.pt", "step-0000010-final.pt", "step-bad.pt"]:
                (path / name).touch()
            self.assertEqual(latest_checkpoint(path).name, "step-0000010-final.pt")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
