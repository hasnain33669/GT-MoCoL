
# GT-MoCoL: Graph Transformer with Motif-level Contrastive Learning

Molecular property prediction framework using Graph Transformer with motif-aware attention and multi-level contrastive learning.

## Architecture

- **Motif Extraction**: Rings and functional groups (SMARTS-based)
- **Motif-Aware Augmentation**: Two complementary views (motif-preserving / motif-corrupting)
- **Graph Transformer**: Multi-head self-attention with motif bias
- **Multi-Level Contrastive Loss**: Graph-level + Motif-level + Alignment losses

## Installation

```bash
pip install -r requirements.txt
Dataset
The data underlying the results presented in this study are:
Dataset, downloaded from https://moleculenet.org/datasets-1

Usage
Step 1: Pre-training
bash
python main_pretrain.py
Step 2: Fine-tuning
bash
python main_finetune.py
Configuration
Edit config.yaml to adjust hyperparameters:

yaml
pretrain:
  epochs: 10
  batch_size: 16
  lr: 0.0001
  hidden_dim: 300
  num_layers: 4

finetune:
  epochs: 100
  batch_size: 16
  lr: 0.0001
File Structure
text
GT-MoCoL/
├── requirements.txt
├── config.yaml
├── data_utils.py          # Data loading & molecular featurization
├── motif_utils.py         # Motif extraction utilities
├── augmentations.py       # Motif-aware augmentations
├── model.py               # GT-MoCoL encoder & attention
├── losses.py              # Multi-level contrastive loss
├── dataset.py             # Pre-training & fine-tuning datasets
├── pretrain.py            # Self-supervised pre-training loop
├── finetune.py            # Supervised fine-tuning loop
├── evaluation.py          # Bootstrap evaluation
├── main_pretrain.py       # Entry point for pre-training
├── main_finetune.py       # Entry point for fine-tuning
└── BACE.csv               # Dataset file
Outputs
gtmocol_pretrained.pt - Pre-trained encoder weights

best_gtmocol_finetune.pt - Best fine-tuned model

Console output with training metrics and test results

Reference
Graph Transformer with Motif-level Contrastive Learning (GT-MoCoL) for molecular property prediction.
