"""
Section 2.2: Ablation of the attention scaling factor.

This experiment compares standard scaled dot-product attention with an unscaled
variant. During the first 1,000 optimization steps, it logs the gradient norms
of Query and Key projection weights for analysis of softmax saturation.
"""

from functools import partial
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

import model as model_module
from dataset import Multi30kDataset, collate_batch
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask
from train import LabelSmoothingLoss, run_epoch, save_checkpoint


BASE_CONFIG = {
    "batch_size": 64,
    "num_epochs": 10,
    "d_model": 512,
    "N": 6,
    "num_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "warmup_steps": 4000,
    "smoothing": 0.1,
    "grad_log_steps": 1000,
}


ORIGINAL_ATTENTION = model_module.scaled_dot_product_attention


def unscaled_dot_product_attention(Q, K, V, mask=None):
    """Attention variant used to isolate the effect of removing sqrt(d_k)."""
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if mask is not None:
        mask = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    attn_w = F.softmax(scores, dim=-1)
    if mask is not None:
        attn_w = attn_w.masked_fill(mask, 0.0)

    return torch.matmul(attn_w, V), attn_w


def set_attention_variant(use_scaling: bool) -> None:
    """Switch the attention function used by MultiHeadAttention."""
    model_module.scaled_dot_product_attention = (
        ORIGINAL_ATTENTION if use_scaling else unscaled_dot_product_attention
    )


def build_loaders(batch_size: int):
    train_data = Multi30kDataset("train")
    val_data = Multi30kDataset("validation")
    pad_idx = train_data.tgt_vocab.pad_idx
    collate = partial(collate_batch, pad_idx=pad_idx)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    return train_data, train_loader, val_loader


def projection_grad_norms(model: Transformer) -> tuple[float, float]:
    """Return aggregate L2 gradient norms for all Query and Key projections."""
    q_norm_sq = 0.0
    k_norm_sq = 0.0

    for module in model.modules():
        if hasattr(module, "w_q") and module.w_q.weight.grad is not None:
            q_norm_sq += module.w_q.weight.grad.detach().norm(2).item() ** 2
        if hasattr(module, "w_k") and module.w_k.weight.grad is not None:
            k_norm_sq += module.w_k.weight.grad.detach().norm(2).item() ** 2

    return math.sqrt(q_norm_sq), math.sqrt(k_norm_sq)


def train_one_epoch_with_grad_logging(
    data_loader,
    model: Transformer,
    loss_fn: LabelSmoothingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    device: str,
    global_step: int,
    max_grad_log_steps: int,
) -> tuple[float, int]:
    """Train for one epoch while logging Query/Key gradient norms early in training."""
    model.train()
    pad_idx = loss_fn.pad_idx
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in data_loader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(
            src,
            tgt_input,
            make_src_mask(src, pad_idx),
            make_tgt_mask(tgt_input, pad_idx),
        )
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()

        global_step += 1
        if global_step <= max_grad_log_steps:
            # Gradient norms are recorded before optimizer.step() changes weights.
            q_grad_norm, k_grad_norm = projection_grad_norms(model)
            wandb.log(
                {
                    "global_step": global_step,
                    "q_grad_norm": q_grad_norm,
                    "k_grad_norm": k_grad_norm,
                    "train_loss_step": loss.item(),
                }
            )

        optimizer.step()
        scheduler.step()

        n_tokens = (tgt_out != pad_idx).sum().item()
        total_loss += loss.item() * max(1, n_tokens)
        total_tokens += n_tokens

    return total_loss / max(1, total_tokens), global_step


def run_condition(condition: str, train_data, train_loader, val_loader, device: str) -> None:
    use_scaling = condition == "scaled"
    set_attention_variant(use_scaling)

    config = dict(BASE_CONFIG)
    config["condition"] = condition
    config["attention_score"] = "QK^T / sqrt(d_k)" if use_scaling else "QK^T"

    wandb.init(
        project="da6401_assignment_3",
        name=f"2_2_{condition}_attention",
        group="2_2_scaling_ablation",
        config=config,
        reinit=True,
    )
    config = wandb.config

    model = Transformer(
        src_vocab_size=len(train_data.src_vocab),
        tgt_vocab_size=len(train_data.tgt_vocab),
        d_model=config.d_model,
        N=config.N,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(
        optimizer,
        d_model=config.d_model,
        warmup_steps=config.warmup_steps,
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_data.tgt_vocab),
        pad_idx=train_data.tgt_vocab.pad_idx,
        smoothing=config.smoothing,
    )

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(config.num_epochs):
        train_loss, global_step = train_one_epoch_with_grad_logging(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            device,
            global_step,
            config.grad_log_steps,
        )
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path=f"checkpoint_2_2_{condition}.pt",
            )

    wandb.finish()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, train_loader, val_loader = build_loaders(BASE_CONFIG["batch_size"])

    try:
        run_condition("scaled", train_data, train_loader, val_loader, device)
        run_condition("unscaled", train_data, train_loader, val_loader, device)
    finally:
        set_attention_variant(True)


if __name__ == "__main__":
    main()
