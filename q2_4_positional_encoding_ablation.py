"""
Section 2.4: Sinusoidal versus learned positional encodings.

This experiment trains two otherwise identical Transformers: one with the fixed
sinusoidal encoding from the paper and one with learned positional embeddings.
Validation BLEU is logged for direct comparison in the W&B report.
"""

from functools import partial

import matplotlib.pyplot as plt
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
    "encoding_plot_positions": 128,
    "encoding_plot_dims": 64,
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


def positional_encoding_matrix(model: Transformer, num_positions: int, num_dims: int) -> torch.Tensor:
    """Return a small position-by-dimension matrix for W&B visualizations."""
    positional_encoding = model.positional_encoding

    if hasattr(positional_encoding, "pe"):
        matrix = positional_encoding.pe.squeeze(0)
    elif hasattr(positional_encoding, "position_embedding"):
        matrix = positional_encoding.position_embedding.weight
    else:
        raise TypeError("Unsupported positional encoding module")

    return matrix[:num_positions, :num_dims].detach().cpu()


def log_positional_encoding_plots(
    model: Transformer,
    condition: str,
    prefix: str,
    num_positions: int,
    num_dims: int,
) -> None:
    """Log heatmap and position-norm plots for the current positional encoding."""
    matrix = positional_encoding_matrix(model, num_positions, num_dims)

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(matrix.numpy(), aspect="auto", cmap="viridis")
    ax.set_title(f"{condition.capitalize()} Positional Encoding ({prefix})")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Position")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Encoding value")
    fig.tight_layout()

    wandb.log(
        {
            f"{prefix}_positional_encoding_heatmap": wandb.Image(fig),
        }
    )
    plt.close(fig)

    # Norm plot
    norms = matrix.norm(dim=1)

    norm_table = wandb.Table(columns=["position", "encoding_norm"])
    for position, norm in enumerate(norms.tolist()):
        norm_table.add_data(position, norm)

    wandb.log(
        {
            f"{prefix}_positional_encoding_norm_table": norm_table,
            f"{prefix}_positional_encoding_norm": wandb.plot.line(
                norm_table,
                "position",
                "encoding_norm",
                title=f"{condition.capitalize()} Positional Encoding Norm ({prefix})",
            ),
        }
    )


def run_condition(condition: str, train_data, train_loader, val_loader, device: str) -> dict:
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

    # Initial positional encoding visualization
    log_positional_encoding_plots(
        model,
        condition,
        "initial",
        config.encoding_plot_positions,
        config.encoding_plot_dims,
    )

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
    best_epoch = -1

    final_train_loss = None
    final_val_loss = None
    final_val_bleu = None

    for epoch in range(config.num_epochs):

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

        # Main plots
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
            best_epoch = epoch

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path=f"checkpoint_2_4_{condition}.pt",
            )

        final_train_loss = train_loss
        final_val_loss = val_loss
        final_val_bleu = val_bleu

        wandb.log(
            {
                "epoch": epoch,
                "best_val_bleu_so_far": best_val_bleu,
                "best_epoch_so_far": best_epoch,
            }
        )

    # Final positional encoding visualization
    log_positional_encoding_plots(
        model,
        condition,
        "final",
        config.encoding_plot_positions,
        config.encoding_plot_dims,
    )

    wandb.summary["best_val_bleu"] = best_val_bleu
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["final_train_loss"] = final_train_loss
    wandb.summary["final_val_loss"] = final_val_loss
    wandb.summary["final_val_bleu"] = final_val_bleu

    wandb.finish()

    return {
        "condition": condition,
        "best_val_bleu": best_val_bleu,
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_val_bleu": final_val_bleu,
    }


def log_summary_plots(results: list[dict]) -> None:
    """Log direct comparisons between the two positional encodings."""

    wandb.init(
        project="da6401_assignment_3",
        name="2_4_positional_encoding_summary",
        group="2_4_positional_encoding_ablation",
        job_type="summary",
        reinit=True,
    )

    summary_table = wandb.Table(
        columns=[
            "condition",
            "best_val_bleu",
            "best_epoch",
            "final_train_loss",
            "final_val_loss",
            "final_val_bleu",
        ]
    )

    for result in results:
        summary_table.add_data(
            result["condition"],
            result["best_val_bleu"],
            result["best_epoch"],
            result["final_train_loss"],
            result["final_val_loss"],
            result["final_val_bleu"],
        )

    wandb.log(
        {
            "summary_table": summary_table,

            # MAIN REQUIRED PLOT
            "best_val_bleu_comparison": wandb.plot.bar(
                summary_table,
                "condition",
                "best_val_bleu",
                title="Best Validation BLEU by Positional Encoding",
            ),

            # SUPPORTING PLOTS
            "final_val_bleu_comparison": wandb.plot.bar(
                summary_table,
                "condition",
                "final_val_bleu",
                title="Final Validation BLEU by Positional Encoding",
            ),

            "final_val_loss_comparison": wandb.plot.bar(
                summary_table,
                "condition",
                "final_val_loss",
                title="Final Validation Loss by Positional Encoding",
            ),

            "final_train_loss_comparison": wandb.plot.bar(
                summary_table,
                "condition",
                "final_train_loss",
                title="Final Training Loss by Positional Encoding",
            ),
        }
    )

    wandb.finish()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data, train_loader, val_loader = build_loaders(BASE_CONFIG["batch_size"])

    results = [
        run_condition("sinusoidal", train_data, train_loader, val_loader, device),
        run_condition("learned", train_data, train_loader, val_loader, device),
    ]

    log_summary_plots(results)


if __name__ == "__main__":
    main()
