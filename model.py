"""
Transformer model components for DA6401 Assignment 3.

This file implements the base encoder-decoder Transformer from
"Attention Is All You Need" using PyTorch building blocks. The public
function signatures are kept stable for the assignment autograder.
"""

import copy
import math
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CHECKPOINT_PATH = "checkpoint.pt"
# Replace this with your Google Drive sharing URL or file id before submission.
DEFAULT_CHECKPOINT_URL = os.environ.get(
    "TRANSFORMER_CHECKPOINT_URL",
    "https://drive.google.com/file/d/1imdff6iYoDwl3wPVBHDk4ePxnK0R5HfK/view?usp=sharing",
)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor with shape (..., seq_q, d_k).
        K: Key tensor with shape (..., seq_k, d_k).
        V: Value tensor with shape (..., seq_k, d_v).
        mask: Optional boolean mask broadcastable to (..., seq_q, seq_k).
            True entries are excluded from attention.

    Returns:
        A pair (output, attention_weights).
    """
    d_k = Q.size(-1)
    # Scaling keeps dot products in a range where softmax gradients remain useful.
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(device=scores.device, dtype=torch.bool)
        # Masked keys receive zero probability after softmax.
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    attn_w = F.softmax(scores, dim=-1)
    if mask is not None:
        attn_w = attn_w.masked_fill(mask, 0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Create an encoder padding mask.

    Returns a boolean tensor of shape [batch, 1, 1, src_len].
    True values mark PAD positions.
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Create a decoder mask combining padding and causal masking.

    Returns a boolean tensor of shape [batch, 1, tgt_len, tgt_len].
    True values mark positions that must not be attended to.
    """
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    # Upper-triangular entries prevent positions from attending to future tokens.
    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(1)
    return pad_mask | causal_mask.expand(batch_size, 1, tgt_len, tgt_len)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention implemented with explicit linear projections.

    The module does not use torch.nn.MultiheadAttention. Attention weights from
    the most recent forward pass are stored in self.attn_weights for analysis.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Tensor of shape [batch, seq_q, d_model].
            key: Tensor of shape [batch, seq_k, d_model].
            value: Tensor of shape [batch, seq_k, d_model].
            mask: Optional boolean mask broadcastable to
                [batch, num_heads, seq_q, seq_k].
        """
        batch_size = query.size(0)

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            # [batch, seq, d_model] -> [batch, heads, seq, d_k]
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        q = split_heads(self.w_q(query))
        k = split_heads(self.w_k(key))
        v = split_heads(self.w_v(value))

        _, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        self.attn_weights = attn_weights.detach()
        attn_output = torch.matmul(self.dropout(attn_weights), v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        return self.w_o(attn_output)


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.

    The encoding is stored as a buffer so it moves with the module across
    devices but is not optimized as a trainable parameter.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.dropout = nn.Dropout(dropout)
        # Fixed encodings should move with the model but not receive gradients.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to token embeddings."""
        x = x + self.pe[:, : x.size(1), :].to(dtype=x.dtype)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward network applied independently at each position."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder layer with self-attention and feed-forward sublayers."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        return self.norm2(x + self.dropout2(ff_out))


class DecoderLayer(nn.Module):
    """
    Single decoder layer with masked self-attention, cross-attention, and FFN.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))
        ff_out = self.feed_forward(x)
        return self.norm3(x + self.dropout3(ff_out))


class Encoder(nn.Module):
    """Stack of N encoder layers followed by final layer normalization."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm2.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N decoder layers followed by final layer normalization."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm3.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """Full encoder-decoder Transformer for sequence-to-sequence translation."""

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        checkpoint_url: str = DEFAULT_CHECKPOINT_URL,
        load_weights: Optional[bool] = None,
        max_decode_len: int = 100,
    ) -> None:
        super().__init__()
        autograder_mode = src_vocab_size is None or tgt_vocab_size is None
        if load_weights is None:
            load_weights = autograder_mode

        self.src_vocab = None
        self.tgt_vocab = None
        self.tokenizer_de = None
        self.tokenizer_en = None
        self.max_decode_len = max_decode_len
        self.checkpoint_path = checkpoint_path
        self.checkpoint_url = checkpoint_url

        checkpoint = None
        if load_weights:
            checkpoint = self._read_checkpoint(checkpoint_path, checkpoint_url)
            checkpoint_config = checkpoint.get("model_config", {})
            d_model = checkpoint_config.get("d_model", d_model)
            N = checkpoint_config.get("N", N)
            num_heads = checkpoint_config.get("num_heads", num_heads)
            d_ff = checkpoint_config.get("d_ff", d_ff)
            dropout = checkpoint_config.get("dropout", dropout)
            src_vocab_size = checkpoint_config.get("src_vocab_size", src_vocab_size)
            tgt_vocab_size = checkpoint_config.get("tgt_vocab_size", tgt_vocab_size)

        if autograder_mode:
            self._load_translation_assets()
            src_vocab_size = src_vocab_size or len(self.src_vocab)
            tgt_vocab_size = tgt_vocab_size or len(self.tgt_vocab)

        if src_vocab_size is None or tgt_vocab_size is None:
            raise ValueError("src_vocab_size and tgt_vocab_size are required when vocab assets are not loaded")

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self.model_config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
        }

        self._reset_parameters()
        if checkpoint is not None:
            self.load_state_dict(checkpoint["model_state_dict"])

    def _read_checkpoint(self, checkpoint_path: str, checkpoint_url: str) -> dict:
        path = Path(checkpoint_path)
        if not path.exists():
            if not checkpoint_url:
                raise FileNotFoundError(
                    f"{checkpoint_path} not found. Set DEFAULT_CHECKPOINT_URL in model.py "
                    "to your Google Drive checkpoint URL before submission."
                )
            try:
                import gdown
            except ImportError as exc:
                raise ImportError(
                    "gdown is required to download the trained checkpoint. "
                    "Install it or add it to requirements.txt."
                ) from exc
            gdown.download(checkpoint_url, str(path), quiet=False, fuzzy=True)

        checkpoint = torch.load(path, map_location="cpu")
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint must contain a model_state_dict entry")
        return checkpoint

    def _load_translation_assets(self) -> None:
        if self.src_vocab is not None and self.tgt_vocab is not None:
            return
        from dataset import Multi30kDataset

        dataset = Multi30kDataset("train")
        self.src_vocab = dataset.src_vocab
        self.tgt_vocab = dataset.tgt_vocab
        self.tokenizer_de = dataset.tokenizer_de
        self.tokenizer_en = dataset.tokenizer_en

    def _tokenize_de(self, sentence: str) -> list[str]:
        self._load_translation_assets()
        return [token.lower_ for token in self.tokenizer_de(sentence)]

    def _detokenize_en(self, tokens: list[str]) -> str:
        sentence = " ".join(tokens)
        sentence = re.sub(r"\s+([.,!?;:%])", r"\1", sentence)
        sentence = re.sub(r"\s+(')", r"\1", sentence)
        sentence = sentence.replace("( ", "(").replace(" )", ")")
        return sentence.strip()

    def _reset_parameters(self) -> None:
        # Xavier initialization is the standard choice for Transformer linear layers.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed source tokens and run the encoder stack."""
        src_emb = self.positional_encoding(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed target tokens, run the decoder stack, and return logits."""
        tgt_emb = self.positional_encoding(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        decoded = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.generator(decoded)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, german_sentence: str) -> str:
        """
        Translate one German sentence to English using greedy autoregressive decoding.
        """
        self._load_translation_assets()
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        src_tokens = self._tokenize_de(german_sentence)
        src_ids = self.src_vocab.encode(src_tokens)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, self.src_vocab.pad_idx).to(device)

        sos_idx = self.tgt_vocab.sos_idx
        eos_idx = self.tgt_vocab.eos_idx
        tgt = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

        generated_tokens = []
        with torch.no_grad():
            memory = self.encode(src, src_mask)
            for _ in range(self.max_decode_len - 1):
                tgt_mask = make_tgt_mask(tgt, self.tgt_vocab.pad_idx).to(device)
                logits = self.decode(memory, src_mask, tgt, tgt_mask)
                next_idx = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                if next_idx == eos_idx:
                    break
                token = self.tgt_vocab.lookup_token(next_idx)
                if token not in {"<sos>", "<pad>", "<unk>"}:
                    generated_tokens.append(token)
                next_token = torch.tensor([[next_idx]], dtype=torch.long, device=device)
                tgt = torch.cat([tgt, next_token], dim=1)

        self.train(was_training)
        return self._detokenize_en(generated_tokens)
