"""
Section 2.3: Attention-head visualization.

This script loads a trained checkpoint, extracts attention weights from the last
encoder layer, and logs one heat map per head to W&B. Additional per-head
statistics summarize local attention, adjacent-token attention, and average
attention distance for the report discussion.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb

from dataset import Multi30kDataset
from model import Transformer, make_src_mask


def build_model_from_checkpoint(checkpoint_path: str, device: str) -> Transformer:
    """Rebuild the Transformer architecture recorded inside a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_config" not in checkpoint:
        raise KeyError("Checkpoint must contain model_config to rebuild Transformer")

    model = Transformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def encode_source_sentence(dataset: Multi30kDataset, sentence: str | None):
    """Tokenize a source sentence and prepare labels for attention heat maps."""
    if sentence is None:
        example = dataset.raw_data[0]
        sentence, _ = dataset._get_text_pair(example)

    tokens = dataset._tokenize_de(sentence)
    indices = dataset.src_vocab.encode(tokens)
    labels = ["<sos>"] + tokens + ["<eos>"]
    src = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    return sentence, labels, src


def attention_statistics(attn: torch.Tensor) -> dict[str, float]:
    """
    Summarize one attention matrix for qualitative interpretation.

    The matrix has shape [seq_len, seq_len], where rows are query tokens and
    columns are key tokens.
    """
    seq_len = attn.size(0)
    diag = torch.diagonal(attn).mean().item()
    next_token = torch.diagonal(attn[:, 1:], offset=0).mean().item() if seq_len > 1 else 0.0
    prev_token = torch.diagonal(attn[1:, :], offset=0).mean().item() if seq_len > 1 else 0.0

    distance = torch.arange(seq_len, device=attn.device)
    distance = (distance.unsqueeze(0) - distance.unsqueeze(1)).abs().float()
    expected_distance = (attn * distance).sum(dim=-1).mean().item()

    entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
    return {
        "diagonal_attention": diag,
        "next_token_attention": next_token,
        "previous_token_attention": prev_token,
        "expected_attention_distance": expected_distance,
        "attention_entropy": entropy,
    }


def make_heatmap(attn, labels, title: str):
    """Create a labeled heat map for a single attention head."""
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(attn, cmap="viridis", vmin=0.0, vmax=max(1e-8, float(attn.max())))
    ax.set_title(title)
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def log_last_encoder_attention(
    model: Transformer,
    src: torch.Tensor,
    labels: list[str],
    pad_idx: int,
    device: str,
) -> None:
    src = src.to(device)
    src_mask = make_src_mask(src, pad_idx).to(device)

    with torch.no_grad():
        model.encode(src, src_mask)

    # The MultiHeadAttention module stores weights from its latest forward pass.
    attn = model.encoder.layers[-1].self_attn.attn_weights
    if attn is None:
        raise RuntimeError("No attention weights found on the last encoder layer")

    attn = attn.squeeze(0).detach().cpu()
    for head_idx in range(attn.size(0)):
        # Each head is logged separately so it can be inspected in the report.
        head_attn = attn[head_idx]
        stats = attention_statistics(head_attn)
        fig = make_heatmap(
            head_attn.numpy(),
            labels,
            title=f"Last Encoder Layer - Head {head_idx}",
        )

        log_data = {
            "head": head_idx,
            f"head_{head_idx}_heatmap": wandb.Image(fig),
        }
        log_data.update({f"head_{head_idx}_{key}": value for key, value in stats.items()})
        wandb.log(log_data)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log last encoder layer attention heads to W&B")
    parser.add_argument("--checkpoint", default="checkpoint.pt", help="Path to trained checkpoint")
    parser.add_argument("--sentence", default=None, help="German source sentence to visualize")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(
            f"{args.checkpoint} was not found. Train the main model first with: python train.py"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = Multi30kDataset("validation")
    sentence, labels, src = encode_source_sentence(dataset, args.sentence)
    model = build_model_from_checkpoint(args.checkpoint, device)

    wandb.init(
        project="da6401_assignment_3",
        name="2_3_attention_head_heatmaps",
        group="2_3_attention_rollout",
        config={
            "checkpoint": args.checkpoint,
            "sentence": sentence,
            "num_tokens": len(labels),
        },
    )
    wandb.log({"visualized_sentence": sentence})
    log_last_encoder_attention(model, src, labels, dataset.src_vocab.pad_idx, device)
    wandb.finish()


if __name__ == "__main__":
    main()
