"""Full ROC64 profile evaluation, using the senior experiment's actual scorer.

Do not use eval_langflow.py for this protocol: it intentionally retains EOS in
packed blocks, whereas Project's story evaluator cuts at the first EOS.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch

from langflow import LangFlow, LangFlowConfig
from training import senior_roc64
from training.runtime import latest_checkpoint
import experiment_tracking as tracking


@tracking.tracked
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-mauve", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.num_samples, args.generation_batch_size) < 1 or args.num_steps < 2:
        raise ValueError("Invalid generation size/step count")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to overwrite evaluation")
    if args.checkpoint.is_dir():
        args.checkpoint = latest_checkpoint(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    metadata = senior_roc64.load_metadata(args.data_dir)
    if checkpoint.get("data_protocol") != senior_roc64.PROTOCOL or checkpoint["data_fingerprint"] != metadata["fingerprint"]:
        raise ValueError("Evaluation data must exactly match the full-story checkpoint")
    model = LangFlow(LangFlowConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["ema"])
    step = checkpoint["step"]
    del checkpoint
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model.to(device).eval()
    sys.path.insert(0, str(args.reference_project))
    from ccflow.data import Data, decode
    from ccflow.evaluate import TextEvaluator
    source_hashes = {name: senior_roc64.sha256(args.reference_project / name)
                     for name in ("ccflow/data.py", "ccflow/evaluate.py", "evalkit/evaluate.py")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracking.start({**vars(args), "data_protocol": senior_roc64.PROTOCOL,
                    "scorer_source_hashes": source_hashes}, args.output_dir,
                   model="langflow", job_type="eval", group_dir=args.checkpoint.parent.parent)
    ids = []
    for index, start in enumerate(range(0, args.num_samples, args.generation_batch_size)):
        torch.manual_seed(args.seed + index)
        batch = model.generate_samples(num_samples=min(args.generation_batch_size, args.num_samples-start),
                                       seq_length=64, num_steps=args.num_steps, device=device)
        ids.append(batch.cpu())
        print(f"Generated {start+len(batch)}/{args.num_samples}", flush=True)
    ids = torch.cat(ids)
    texts = decode(ids)
    with (args.output_dir / "samples.jsonl").open("w") as handle:
        for row, text in zip(ids.tolist(), texts):
            handle.write(json.dumps({"ids": row, "text": text}, ensure_ascii=False) + "\n")
    del model, batch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    evaluator = TextEvaluator(Data(args.data_dir), device, mauve=not args.skip_mauve)
    metrics = evaluator.vs_refs(texts)
    payload = {"complete": True, "step": step, "epoch": step / 182, "weights": "ema",
               "data_protocol": senior_roc64.PROTOCOL, "data_fingerprint": metadata["fingerprint"],
               "scorer_source_hashes": source_hashes, "args": tracking.clean_config(vars(args)),
               "metrics": metrics}
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    (args.output_dir / "summary.json").write_text(json.dumps(metrics, indent=2))
    tracking.log_evaluation({"generation": metrics}, complete=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
