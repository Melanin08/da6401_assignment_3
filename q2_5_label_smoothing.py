"""
Section 2.5: Decoder sensitivity to label smoothing.

This experiment compares epsilon=0.1 label smoothing with epsilon=0.0 training.
It logs prediction confidence, defined as the softmax probability assigned to
the correct non-padding target token.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

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
}


def build_loaders(batch_size: int):
    """Create train and validation loaders for both smoothing settings."""
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


def prediction_confidence(model, data_loader, pad_idx: int, device: str, max_batches: int = 20) -> float:
    """
    Estimate average probability assigned to the correct target token.

    Padding positions are ignored. A fixed number of validation batches is used
    to keep the diagnostic consistent across both smoothing settings.
    """
    model.eval()
    confidence_sum = 0.0
    token_count = 0

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(
                src,
                tgt_input,
                make_src_mask(src, pad_idx),
                make_tgt_mask(tgt_input, pad_idx),
            )
            probs = F.softmax(logits, dim=-1)
            # Select the probability assigned to the correct token at each position.
            gold_probs = probs.gather(dim=-1, index=tgt_out.unsqueeze(-1)).squeeze(-1)
            non_pad = tgt_out != pad_idx
            confidence_sum += gold_probs[non_pad].sum().item()
            token_count += non_pad.sum().item()

    return confidence_sum / max(1, token_count)


def run_condition(smoothing: float, train_data, train_loader, val_loader, device: str) -> None:
    condition = f"eps_{smoothing:.1f}"
    config = dict(BASE_CONFIG)
    config["smoothing"] = smoothing
    config["condition"] = condition

    wandb.init(
        project="da6401_assignment_3",
        name=f"2_5_label_smoothing_{condition}",
        group="2_5_label_smoothing",
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
    for epoch in range(config.num_epochs):
        # The model architecture is unchanged; only epsilon in the loss changes.
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
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
        confidence = prediction_confidence(
            model,
            val_loader,
            train_data.tgt_vocab.pad_idx,
            device,
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "prediction_confidence": confidence,
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
                path=f"checkpoint_2_5_{condition}.pt",
            )

    wandb.finish()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, train_loader, val_loader = build_loaders(BASE_CONFIG["batch_size"])

    run_condition(0.1, train_data, train_loader, val_loader, device)
    run_condition(0.0, train_data, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
