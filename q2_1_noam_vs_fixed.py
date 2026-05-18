"""
Section 2.1: Necessity of the Noam learning-rate schedule.

This experiment trains the same Transformer architecture under two optimizer
settings: the Noam warmup schedule and a fixed learning rate. The resulting
training-loss and validation-accuracy curves are logged to W&B for comparison.
"""

from functools import partial

import torch
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
    "fixed_lr": 1e-4,
    "smoothing": 0.1,
}


def build_loaders(batch_size: int):
    """Create train and validation loaders using the shared training vocabulary."""
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


def evaluate_token_accuracy(model, data_loader, pad_idx: int, device: str) -> float:
    """
    Compute validation accuracy over non-padding target tokens.

    The decoder input is the shifted target prefix. Accuracy is measured against
    the next-token labels, ignoring positions introduced only for padding.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for src, tgt in data_loader:
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
            pred = logits.argmax(dim=-1)
            non_pad = tgt_out != pad_idx
            correct += ((pred == tgt_out) & non_pad).sum().item()
            total += non_pad.sum().item()

    return correct / max(1, total)


def run_condition(condition: str, train_data, train_loader, val_loader, device: str) -> None:
    config = dict(BASE_CONFIG)
    config["condition"] = condition
    config["scheduler"] = "noam" if condition == "noam" else "fixed_lr"

    wandb.init(
        project="da6401_assignment_3",
        name=f"2_1_{condition}",
        group="2_1_noam_vs_fixed",
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

    # Only the learning-rate policy changes between the two conditions.
    if condition == "noam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(
            optimizer,
            d_model=config.d_model,
            warmup_steps=config.warmup_steps,
        )
    elif condition == "fixed_lr":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.fixed_lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        scheduler = None
    else:
        raise ValueError(f"Unknown condition: {condition}")

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_data.tgt_vocab),
        pad_idx=train_data.tgt_vocab.pad_idx,
        smoothing=config.smoothing,
    )

    best_val_loss = float("inf")
    for epoch in range(config.num_epochs):
        # Log both curves required in the report: training loss and validation accuracy.
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
        train_accuracy = evaluate_token_accuracy(
            model,
            train_loader,
            train_data.tgt_vocab.pad_idx,
            device,
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
        val_accuracy = evaluate_token_accuracy(
            model,
            val_loader,
            train_data.tgt_vocab.pad_idx,
            device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path=f"checkpoint_2_1_{condition}.pt",
            )

    wandb.finish()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, train_loader, val_loader = build_loaders(BASE_CONFIG["batch_size"])

    run_condition("noam", train_data, train_loader, val_loader, device)
    run_condition("fixed_lr", train_data, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
