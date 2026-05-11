"""
Section 2.4: Sinusoidal versus learned positional encodings.

This experiment trains two otherwise identical Transformers: one with the fixed
sinusoidal encoding from the paper and one with learned positional embeddings.
Validation BLEU is logged for direct comparison in the W&B report.
"""

from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from dataset import Multi30kDataset, collate_batch
from lr_scheduler import NoamScheduler
from model import Transformer
from train import LabelSmoothingLoss, evaluate_bleu, run_epoch, save_checkpoint


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
    "max_len": 5000,
    "bleu_max_len": 100,
}


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings for the ablation experiment.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.dropout(x + self.position_embedding(positions))


def build_loaders(batch_size: int):
    """Create loaders shared by the sinusoidal and learned-position runs."""
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


def build_model(condition: str, train_data: Multi30kDataset, config, device: str) -> Transformer:
    """Build the same Transformer, changing only the positional encoding module."""
    model = Transformer(
        src_vocab_size=len(train_data.src_vocab),
        tgt_vocab_size=len(train_data.tgt_vocab),
        d_model=config.d_model,
        N=config.N,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
    )

    if condition == "learned":
        # Replace the fixed sinusoidal module with learned position embeddings.
        model.positional_encoding = LearnedPositionalEncoding(
            d_model=config.d_model,
            dropout=config.dropout,
            max_len=config.max_len,
        )
    elif condition == "sinusoidal":
        pass
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return model.to(device)


def run_condition(condition: str, train_data, train_loader, val_loader, device: str) -> None:
    config = dict(BASE_CONFIG)
    config["condition"] = condition

    wandb.init(
        project="da6401_assignment_3",
        name=f"2_4_{condition}_positional_encoding",
        group="2_4_positional_encoding_ablation",
        config=config,
        reinit=True,
    )
    config = wandb.config

    model = build_model(condition, train_data, config, device)
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

    best_val_bleu = -1.0
    for epoch in range(config.num_epochs):
        # Validation BLEU is the comparison metric requested for this ablation.
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
        val_bleu = evaluate_bleu(
            model,
            val_loader,
            train_data.tgt_vocab,
            device=device,
            max_len=config.bleu_max_len,
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_bleu": val_bleu,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path=f"checkpoint_2_4_{condition}.pt",
            )

    wandb.finish()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, train_loader, val_loader = build_loaders(BASE_CONFIG["batch_size"])

    run_condition("sinusoidal", train_data, train_loader, val_loader, device)
    run_condition("learned", train_data, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
