"""
train.py - Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"
"""

from collections import Counter
from functools import partial
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]
            target : shape [batch * tgt_len]

        Returns:
            Scalar loss value.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            # Build the smoothed target distribution by hand so PAD tokens contribute
            # neither probability mass nor loss.
            true_dist = torch.full_like(
                log_probs,
                self.smoothing / max(1, self.vocab_size - 2),
            )
            true_dist[:, self.pad_idx] = 0.0
            target_for_scatter = target.masked_fill(target == self.pad_idx, 0)
            true_dist.scatter_(1, target_for_scatter.unsqueeze(1), self.confidence)
            true_dist[target == self.pad_idx] = 0.0

        loss = F.kl_div(log_probs, true_dist, reduction="sum")
        non_pad = (target != self.pad_idx).sum().clamp_min(1)
        return loss / non_pad


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.
    """
    model.train(is_train)
    pad_idx = getattr(loss_fn, "pad_idx", 1)
    total_loss = 0.0
    total_tokens = 0

    with torch.set_grad_enabled(is_train):
        for src, tgt in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)
            # Teacher forcing: decoder reads every target token except the last
            # and predicts every target token except the first <sos>.
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx)

            if is_train:
                if optimizer is None:
                    raise ValueError("optimizer must be provided during training")
                optimizer.zero_grad(set_to_none=True)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            if is_train:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n_tokens = (tgt_out != pad_idx).sum().item()
            total_loss += loss.item() * max(1, n_tokens)
            total_tokens += n_tokens

    return total_loss / max(1, total_tokens)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.
    """
    was_training = model.training
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            # The decoder mask is rebuilt as the generated prefix grows.
            tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            next_word = torch.argmax(out[:, -1, :], dim=-1).item()
            ys = torch.cat(
                [ys, torch.tensor([[next_word]], dtype=torch.long, device=device)],
                dim=1,
            )
            if next_word == end_symbol:
                break

    model.train(was_training)
    return ys


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
    """
    def token_for(idx: int) -> str:
        if hasattr(tgt_vocab, "lookup_token"):
            return tgt_vocab.lookup_token(idx)
        return tgt_vocab.itos[idx]

    def idx_for(token: str, default: int) -> int:
        if hasattr(tgt_vocab, "stoi"):
            return tgt_vocab.stoi.get(token, default)
        return default

    def strip_specials(indices: list[int]) -> list[str]:
        words = []
        for idx in indices:
            token = token_for(int(idx))
            if token == "<eos>":
                break
            if token not in {"<sos>", "<pad>", "<unk>"}:
                words.append(token)
        return words

    def ngram_counts(tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    pad_idx = idx_for("<pad>", 1)
    sos_idx = idx_for("<sos>", 2)
    eos_idx = idx_for("<eos>", 3)
    model.eval()
    hypotheses = []
    references = []

    with torch.no_grad():
        for src_batch, tgt_batch in test_dataloader:
            for src, tgt in zip(src_batch, tgt_batch):
                src = src.unsqueeze(0).to(device)
                src_mask = make_src_mask(src, pad_idx).to(device)
                pred = greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx, device)
                hypotheses.append(strip_specials(pred.squeeze(0).tolist()))
                references.append(strip_specials(tgt.tolist()))

    if not hypotheses:
        return 0.0

    precisions = []
    for n in range(1, 5):
        clipped = 0
        total = 0
        for hyp, ref in zip(hypotheses, references):
            hyp_counts = ngram_counts(hyp, n)
            ref_counts = ngram_counts(ref, n)
            # Corpus BLEU clips hypothesis n-grams by reference counts.
            clipped += sum(min(count, ref_counts[ngram]) for ngram, count in hyp_counts.items())
            total += sum(hyp_counts.values())
        precisions.append(clipped / total if total else 0.0)

    if min(precisions) == 0.0:
        return 0.0

    hyp_len = sum(len(hyp) for hyp in hypotheses)
    ref_len = sum(len(ref) for ref in references)
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(1, hyp_len))
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return bleu * 100.0


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.
    """
    model_config = getattr(model, "model_config", None)
    if model_config is None:
        model_config = {
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
            "d_model": model.d_model,
            "N": model.N,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "dropout": model.dropout_p,
        }

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model_config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model and optionally optimizer/scheduler state from disk.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"])


def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    import wandb
    from dataset import Multi30kDataset, collate_batch
    from lr_scheduler import NoamScheduler

    config = {
        "batch_size": 64,
        "num_epochs": 10,
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "smoothing": 0.1,
    }
    wandb.init(project="da6401_assignment_3", config=config)
    config = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = Multi30kDataset("train")
    val_data = Multi30kDataset("validation")
    test_data = Multi30kDataset("test")
    pad_idx = train_data.tgt_vocab.pad_idx

    collate = partial(collate_batch, pad_idx=pad_idx)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate)

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
    scheduler = NoamScheduler(optimizer, d_model=config.d_model, warmup_steps=config.warmup_steps)
    loss_fn = LabelSmoothingLoss(len(train_data.tgt_vocab), pad_idx, config.smoothing)

    best_val = float("inf")
    for epoch in range(config.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, False, device)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, "checkpoint.pt")

    bleu = evaluate_bleu(model, test_loader, train_data.tgt_vocab, device)
    wandb.log({"test_bleu": bleu})


if __name__ == "__main__":
    run_training_experiment()
