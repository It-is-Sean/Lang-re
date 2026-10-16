"""Small CPU checks of GenPPL accounting and evaluation safety guards."""
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from eval_langflow import build_parser, compute_generated_ppl, validate_arguments


class Tokenizer:
    sequences = {"a": [0, 1, 0, 2], "b": [0, 3], "c": [0, 2, 1]}

    def __call__(self, texts, **kwargs):
        sequences = [self.sequences[text] for text in texts]
        width = max(map(len, sequences))
        return {
            "input_ids": torch.tensor([ids + [0] * (width - len(ids)) for ids in sequences]),
            "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences]),
        }


class Scorer:
    def __call__(self, input_ids, attention_mask, labels):
        # Distinct per-token losses, so equal weighting of batches is detectably wrong.
        logits = (input_ids.float()[..., None] + 1) * torch.arange(4).float()
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 4), labels[:, 1:].reshape(-1))
        return SimpleNamespace(loss=loss)


class EvalProtocolTests(unittest.TestCase):
    def test_token_weighting_padding_and_internal_eos(self):
        scores = [compute_generated_ppl(Scorer(), Tokenizer(), ["a", "b", "c"], batch,
                                        64, torch.device("cpu")) for batch in (1, 2, 3)]
        for score in scores:
            self.assertEqual(score["num_scored_tokens"], 6)  # EOS=0 is valid inside a sequence.
            self.assertAlmostEqual(score["nll_per_token"], scores[0]["nll_per_token"], places=6)
            self.assertAlmostEqual(score["gen_ppl"], math.exp(score["nll_per_token"]))

    def test_reject_overwrite_and_short_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "eval"
            args = build_parser().parse_args([
                "--checkpoint", "unused.pt", "--cache-dir", "unused", "--output-dir", str(output)])
            validate_arguments(args)
            args.max_ppl_length = 32
            with self.assertRaisesRegex(ValueError, "entire packed block"):
                validate_arguments(args)
            args.max_ppl_length = 64
            output.mkdir()
            (output / "summary.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                validate_arguments(args)


if __name__ == "__main__":
    unittest.main()
