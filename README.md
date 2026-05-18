# DA6401 Assignment 3
## Implementing a Transformer for Machine Translation

This project implements the Transformer architecture from the paper **Attention Is All You Need** using PyTorch for German-to-English machine translation on the Multi30k dataset.

The implementation was built completely from scratch using basic PyTorch modules such as `nn.Linear` and `nn.Module`. The project does not use `torch.nn.MultiheadAttention` or `torch.nn.Transformer`.

The implementation includes:

- Scaled Dot-Product Attention
- Multi-Head Attention
- Encoder and Decoder stacks
- Sinusoidal Positional Encoding
- Padding and Causal Masking
- Label Smoothing
- Noam Learning Rate Scheduler
- Greedy Decoding
- BLEU Score Evaluation
- Weights & Biases experiment tracking

---

# Dataset

The model is trained on the **Multi30k** dataset.

Dataset statistics:

- 29,000 training sentence pairs
- 1,014 validation sentence pairs
- 1,000 test sentence pairs

Source language: German  
Target language: English

The dataset is loaded using the Hugging Face `datasets` library.

---

# Repository Structure

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

---

# Main Files

## model.py

Contains the complete Transformer implementation.

Implemented components:

- Scaled Dot-Product Attention
- Multi-Head Attention
- Feed Forward Network
- Positional Encoding
- Encoder Layer
- Decoder Layer
- Encoder Stack
- Decoder Stack
- Transformer Model
- Source Mask
- Target Mask

The implementation follows the architecture described in the original Transformer paper.

---

## dataset.py

Handles dataset loading and preprocessing.

Features:

- German and English tokenization using spaCy
- Vocabulary construction
- Batch padding
- Text to token-id conversion

Special tokens used:

```text
<pad>
<unk>
<sos>
<eos>
```

---

## lr_scheduler.py

Implements the Noam learning rate scheduler.

Formula:

```text
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

---

## train.py

Runs the main Transformer training pipeline.

Features:

- Training loop
- Validation loop
- BLEU evaluation
- Label smoothing
- Greedy decoding
- Checkpoint saving
- W&B logging

Main checkpoint:

```text
checkpoint.pt
```

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Login to Weights & Biases:

```bash
wandb login
```

---

# Training the Main Model

Run:

```bash
python train.py
```

This will:

- Train the Transformer model
- Save the best checkpoint
- Evaluate BLEU score
- Log metrics to W&B

---

# Quick Model Test

Run:

```bash
python -c "import torch; from model import Transformer, make_src_mask, make_tgt_mask; src=torch.tensor([[2,4,5,1]]); tgt=torch.tensor([[2,7,8,1]]); model=Transformer(20,30,d_model=32,N=2,num_heads=4,d_ff=64); out=model(src,tgt,make_src_mask(src),make_tgt_mask(tgt)); print(out.shape)"
```

Expected output:

```text
torch.Size([1, 4, 30])
```

---

# Weights & Biases Experiments

The assignment includes five W&B experiments.

---

# 2.1 Noam Scheduler vs Fixed Learning Rate

Run:

```bash
python q2_1_noam_vs_fixed.py
```

Comparison:

- Noam Scheduler
- Fixed Learning Rate

Logged metrics:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Learning rate

Goal:

Study how the Noam scheduler stabilizes Transformer training during the early training phase.

---

# 2.2 Scaling Factor Ablation

Run:

```bash
python q2_2_scaling_ablation.py
```

Comparison:

- Attention with scaling factor `1/sqrt(d_k)`
- Attention without scaling factor

Logged metrics:

- Query gradient norms
- Key gradient norms
- Training loss
- Validation loss

Goal:

Analyze how the scaling factor prevents softmax saturation and vanishing gradients.

---

# 2.3 Attention Head Visualization

Run:

```bash
python q2_3_attention_heads.py
```

This experiment requires:

```text
checkpoint.pt
```

Optional custom sentence:

```bash
python q2_3_attention_heads.py --sentence "ein mann in einem roten hemd spielt gitarre ."
```

Logged outputs:

- Heatmap for each attention head
- Diagonal attention
- Next-token attention
- Previous-token attention
- Expected attention distance
- Attention entropy

Goal:

Analyze whether different heads learn specialized behaviors and investigate head redundancy.

---

# 2.4 Positional Encoding vs Learned Embeddings

Run:

```bash
python q2_4_positional_encoding_ablation.py
```

Comparison:

- Sinusoidal positional encoding
- Learned positional embeddings

Logged outputs:

- Validation BLEU score
- Training loss
- Validation loss
- Best validation BLEU comparison
- Positional encoding heatmaps

Goal:

Compare positional encoding methods and analyze why sinusoidal encoding can generalize to longer sequences.

---

# 2.5 Label Smoothing

Run:

```bash
python q2_5_label_smoothing.py
```

Comparison:

- Label smoothing `ϵ = 0.1`
- Standard Cross-Entropy `ϵ = 0.0`

Logged outputs:

- Prediction confidence
- Training loss
- Validation loss

Goal:

Analyze how label smoothing prevents over-confident predictions and improves generalization.

---

# Checkpoints

Main checkpoint:

```text
checkpoint.pt
```

Experiment checkpoints:

```text
checkpoint_2_4_sinusoidal.pt
checkpoint_2_4_learned.pt
checkpoint_2_5_eps_0.1.pt
checkpoint_2_5_eps_0.0.pt
```

---

# Model Configuration

Main Transformer settings:

```text
d_model = 512
N = 6
num_heads = 8
d_ff = 2048
dropout = 0.1
```

Layer normalization style:

```text
Post-LayerNorm
```

---

# GitHub Repository

https://github.com/Melanin08/da6401_assignment_3.git


---

# Public W&B Report

[https://wandb.ai/ge26z814-iitm-india/da6401_assignment_3/reports/Implementing-the-Transformer-for-Machine-Translation--VmlldzoxNjc0NzExOA](https://wandb.ai/ge26z814-iitm-india/da6401_assignment_3/reports/Implementing-the-Transformer-for-Machine-Translation--VmlldzoxNjc0NzExOA?accessToken=qk9h8nc9llh6h3g118olcar8gy9wpmn9wktd8tjfu4225ynoa3nhc4x1bn5zx339)


---

# References

Transformer paper:

https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

Multi30k dataset:

https://huggingface.co/datasets/bentrevett/multi30k

