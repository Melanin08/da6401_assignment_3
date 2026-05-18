# DA6401 Assignment 3

## Implementing a Transformer for Machine Translation

This project implements the Transformer architecture from **Attention Is All You
Need** for German-to-English translation on the Multi30k dataset. The model is
built from basic PyTorch components and follows the encoder-decoder structure
described in the paper.

## Submission Links

GitHub repository:

```text
PASTE_REPOSITORY_LINK_HERE
```

Public W&B report:

```text
[PASTE_WANDB_REPORT_LINK_HERE](https://wandb.ai/ge26z814-iitm-india/da6401_assignment_3/reports/Implementing-the-Transformer-for-Machine-Translation--VmlldzoxNjc0NzExOA?accessToken=qk9h8nc9llh6h3g118olcar8gy9wpmn9wktd8tjfu4225ynoa3nhc4x1bn5zx339)
```

The implementation includes attention, masking, positional encoding, encoder and
decoder stacks, label smoothing, the Noam learning-rate scheduler, greedy
decoding, checkpointing, BLEU evaluation, and the W&B experiments required for
the report.

## Repository Structure

```text
.
├── model.py
├── dataset.py
├── lr_scheduler.py
├── train.py
├── q2_1_noam_vs_fixed.py
├── q2_2_scaling_ablation.py
├── q2_3_attention_heads.py
├── q2_4_positional_encoding_ablation.py
├── q2_5_label_smoothing.py
├── requirements.txt
└── README.md
```

## Main Files

### `model.py`

Contains the Transformer implementation:

- scaled dot-product attention
- multi-head attention
- source and target masks
- sinusoidal positional encoding
- feed-forward network
- encoder layer and decoder layer
- encoder stack and decoder stack
- full `Transformer` model

The code does not use `torch.nn.MultiheadAttention` or `nn.Transformer`.

### `dataset.py`

Loads the Multi30k dataset from Hugging Face and prepares it for training.

It handles:

- German and English tokenization using spaCy
- vocabulary creation
- special tokens: `<unk>`, `<pad>`, `<sos>`, `<eos>`
- conversion from text to token ids
- batch padding

The vocabulary is built only from the training split.

### `lr_scheduler.py`

Implements the Noam learning-rate schedule:

```text
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

### `train.py`

Runs the main training pipeline.

It includes:

- label smoothing
- training and validation loop
- greedy decoding
- BLEU evaluation
- checkpoint saving and loading
- W&B logging

The main checkpoint saved by this file is:

```text
checkpoint.pt
```

This is the checkpoint intended for the main test-set evaluation.

## Installation

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Log in to W&B:

```powershell
wandb login
```

## Running the Main Model

Train the main Transformer model:

```powershell
python train.py
```

This will train the model, save `checkpoint.pt`, evaluate BLEU, and log results
to the W&B project:

```text
da6401_assignment_3
```

## Quick Check

To check that the model runs:

```powershell
python -c "import torch; from model import Transformer, make_src_mask, make_tgt_mask; src=torch.tensor([[2,4,5,1]]); tgt=torch.tensor([[2,7,8,1]]); model=Transformer(20,30,d_model=32,N=2,num_heads=4,d_ff=64); out=model(src,tgt,make_src_mask(src),make_tgt_mask(tgt)); print(out.shape)"
```

Expected output:

```text
torch.Size([1, 4, 30])
```

## W&B Report Scripts

The assignment report has five experiments. Each experiment has a separate
script.

### 2.1 Noam Scheduler vs Fixed Learning Rate

```powershell
python q2_1_noam_vs_fixed.py
```

Compares:

- Noam scheduler
- fixed learning rate `1e-4`

Logs:

- training loss
- training accuracy
- validation accuracy
- validation loss
- learning rate

### 2.2 Scaling Factor Ablation

```powershell
python q2_2_scaling_ablation.py
```

Compares:

- attention with `1 / sqrt(d_k)`
- attention without the scaling factor

Logs Query and Key gradient norms during the first 1000 steps.

### 2.3 Attention Head Visualization

```powershell
python q2_3_attention_heads.py
```

This requires a trained `checkpoint.pt`.

It logs heatmaps for each attention head in the last encoder layer.

To visualize a custom German sentence:

```powershell
python q2_3_attention_heads.py --sentence "ein mann in einem roten hemd spielt gitarre ."
```

### 2.4 Sinusoidal vs Learned Positional Encoding

```powershell
python q2_4_positional_encoding_ablation.py
```

Compares:

- sinusoidal positional encoding
- learned positional embeddings

Logs validation BLEU for both settings.

### 2.5 Label Smoothing

```powershell
python q2_5_label_smoothing.py
```

Compares:

- label smoothing `epsilon = 0.1`
- label smoothing `epsilon = 0.0`

Logs prediction confidence, training loss, and validation loss.

## Checkpoints

The main checkpoint is:

```text
checkpoint.pt
```

The report scripts save separate checkpoints for their own comparisons. These
are useful for analysis, but `checkpoint.pt` is the main model checkpoint.

## Notes

- The main model uses `d_model=512`, `N=6`, `num_heads=8`, and `d_ff=2048`.
- The implementation uses Post-LayerNorm: `LayerNorm(x + Sublayer(x))`.
- Multi30k is downloaded through the `datasets` library.
- W&B runs are logged under the project `da6401_assignment_3`.
