"""Paper Algorithm 1 and fixed-noise validation diagnostics."""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from langflow import LangFlow
from .runtime import precision_context


class LangFlowTrainingObjective(nn.Module):
    def __init__(self, model: LangFlow, self_condition_probability: float):
        super().__init__()
        self.model = model
        self.self_condition_probability = self_condition_probability

    def forward(
        self,
        input_ids: torch.Tensor,
        quantiles: torch.Tensor,
        bias_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Algorithm 1 explicitly stops gradients through sampled gamma.  This
        # lets CE train only the denoiser while the fitting loss trains only the
        # proposal parameters.
        gamma = self.model.proposal(quantiles).detach()
        clean_embeddings = self.model._embed_tokens(input_ids)
        noisy_embeddings = self.model._forward_diffusion(clean_embeddings, gamma)

        self_condition = None
        if self.model.config.self_conditioning and self.self_condition_probability > 0:
            active = torch.rand(input_ids.shape[0], device=input_ids.device) < self.self_condition_probability
            active_indices = active.nonzero(as_tuple=True)[0]
            self_condition = torch.zeros_like(clean_embeddings)
            if active_indices.numel() > 0:
                with torch.no_grad():
                    first_logits = self.model(
                        noisy_embeds=noisy_embeddings.index_select(0, active_indices),
                        timesteps=gamma.index_select(0, active_indices),
                        x_self_cond=None,
                        bias_scale=bias_scale,
                        return_dict=False,
                    )
                    first_probabilities = F.softmax(first_logits.float(), dim=-1)
                    prediction = self.model._embed_tokens(first_probabilities)
                self_condition.index_copy_(0, active_indices, prediction.to(self_condition.dtype))

        logits = self.model(
            noisy_embeds=noisy_embeddings,
            timesteps=gamma,
            x_self_cond=self_condition,
            bias_scale=bias_scale,
            return_dict=False,
        )
        token_loss = F.cross_entropy(
            logits.float().flatten(0, 1),
            input_ids.flatten(),
            reduction="none",
        ).view_as(input_ids)
        per_example_ce = token_loss.mean(dim=1)
        ce_loss = per_example_ce.mean()

        proposal = self.model.proposal
        proposal_scale = (proposal.scale if getattr(self.model.config, "precision_scope", None) == "backbone-bf16"
                          else proposal.scale.clamp_min(1e-4))
        entropy_curve = proposal.entropy * torch.exp(
            -torch.exp(-(gamma - proposal.loc) / proposal_scale)
        )
        scheduler_loss = F.mse_loss(entropy_curve, per_example_ce.detach())
        return ce_loss + scheduler_loss, ce_loss.detach(), scheduler_loss.detach()


@torch.no_grad()
def validate(model, loader, ctx, precision, max_batches):
    """Diagnostic CE on a fixed initial-proposal grid, with and without SC.

    This is not likelihood/PPL. Neither the grid nor the noise draw depends on
    the checkpoint's learned scheduler. No duplicated distributed tail rows.
    """
    model.eval()
    total = torch.zeros(3, device=ctx.device, dtype=torch.float64)
    devices = [ctx.device.index] if ctx.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(17029 + ctx.rank)
        for batch_index, input_ids in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            input_ids = input_ids.to(ctx.device)
            # Validation sampler traverses rank, rank+world_size, ... .
            indices = (batch_index * loader.batch_size + torch.arange(len(input_ids), device=ctx.device))
            indices = indices * ctx.world_size + ctx.rank
            # Low-discrepancy quantiles cover the full noise range even
            # when evaluating only a few batches, and remain checkpoint-stable.
            q = (((indices * 0.6180339887498949) + 0.5) % 1).float().clamp(1e-5, 1-1e-5)
            gamma = model.config.gumbel_loc - model.config.gumbel_scale * torch.log(-torch.log(q))
            noisy = model._forward_diffusion(model._embed_tokens(input_ids), gamma)
            with precision_context(precision, ctx.device):
                logits = model(noisy_embeds=noisy, timesteps=gamma, return_dict=False)
                no_sc = F.cross_entropy(logits.float().flatten(0, 1), input_ids.flatten(), reduction="sum")
                if model.config.self_conditioning:
                    sc = model._embed_tokens(F.softmax(logits.float(), dim=-1))
                    logits = model(noisy_embeds=noisy, timesteps=gamma, x_self_cond=sc, return_dict=False)
                with_sc = F.cross_entropy(logits.float().flatten(0, 1), input_ids.flatten(), reduction="sum")
            total += torch.stack((no_sc.double(), with_sc.double(), total.new_tensor(input_ids.numel())))
    if ctx.world_size > 1:
        dist.all_reduce(total)
    if total[2] == 0:
        raise ValueError("Validation contained no tokens")
    return {"ce_no_sc": (total[0] / total[2]).item(), "ce_sc": (total[1] / total[2]).item()}
