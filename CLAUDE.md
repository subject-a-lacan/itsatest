# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep unrolled network for one-bit single-snapshot spectral compressed sensing. Based on the paper "One-bit Single-Measurement Spectral Compressed Sensing via Hankel Matrix Factorization" (OSCAR), this project unfolds the iterative optimization into trainable neural network layers (K unrolled layers, each = GradientModule + ConstraintModule).

## Key Architecture

- **`main.py`** — Entry point. Defines `SpectralDeepUnfolding` (K unrolled layers), `spectral_loss`, training loop `train_model`, and `test_model`.
- **`networks/Modules.py`** — `GradientModule` (one gradient descent step via squareplus-smoothed sign-consistency gradient + Hankel structure penalty) and `ConstraintModule` (learnable 3-layer MLP with residual connection to enforce constraints).
- **`utils/generate_signal_1D.py`** — `generate_signal_batch()` generates clean+noisy complex exponential signals for training/test.
- **`utils/misc.py`** — `spectral_init` (SVD-based Hankel factor initialization), `hankel_lifting`, `Dinv_G_adjoint` (Hankel adjoint with D^{-1} weighting), `MatPencilMethod_Batch` (frequency recovery via matrix pencil).
- **`utils/hankel_operators.py`** — `HankelOperator` class (forward/adjoint for vector↔Hankel matrix).

## Critical Design Choice: Symmetric U*U^T

The factorization uses symmetric Hankel decomposition `H(z) = U * U^T` (complex symmetric, not Hermitian). There is **no V matrix** — all V references were removed. The Takagi factorization requires `U @ U.T` without `.conj()`.

## Loss Function

`spectral_loss` in main.py has two terms:
```python
consistency_loss = mean( ||relu(-Re(y·x_hat), -Im(y·x_hat))||² ) / 2
structure_loss = mean( ||H_hat - H_target||²_fro )
total = consistency_loss + loss_lambda * structure_loss
```
`H_target` is the Hankel matrix of the **clean signal** — this is supervised and differs from the paper's self-supervised `||(I-GG*)(UU^T)||²`. `loss_lambda` (default 100) controls the balance.

## GradientModule Gradient Formula

Uses squareplus `(x - sqrt(x²+b))/2` as smooth approximation of `[w]_-` with b=0.01. The gradient computes `ϕ(w) ⊙ ϕ'(w) ⊙ y`, which corresponds to the p=2 case gradient (the paper's Eq. 39 with the corrected `ϕ·ϕ'` instead of `ϕ·ϕ`).

## Common Commands

```bash
# Single training run (CPU, quick test)
python main.py --device cpu --num_epochs 3 --num_samples 128 --batch_size 16 --num_layers 3 --signal_dim 32 --rank 3

# Production run (GPU)
python -u main.py --device cuda:0 --lr 3e-5 --batch_size 128 --num_layers 5 --num_samples 10000 --num_epochs 100 --loss_lambda 100.0

# Experiment sweep scripts
bash run_lambda_sweep.sh                    # sweep loss_lambda
MAX_JOBS=2 GPU_LIST="0 1" bash run_lambda_sweep.sh  # 2-GPU parallel

# Check experiment progress
for f in logs_lambda/*.log; do printf "%-50s %s\n" "$(basename $f .log)" "$(grep 'SR=' "$f" | tail -1)"; done
```

## CLI Arguments (main.py)

| Group | Argument | Default | Description |
|-------|----------|---------|-------------|
| Signal | `--signal_dim` | 63 | must be odd |
| Signal | `--rank` | 3 | number of sinusoids |
| Training | `--num_layers` | 5 | unrolled layers K |
| Training | `--num_samples` | 1000 | training sample count |
| Training | `--num_epochs` | 100 | training epochs |
| Training | `--lr` | 0.001 | learning rate |
| Training | `--batch_size` | 32 | batch size |
| Training | `--loss_lambda` | 100.0 | structure loss weight |
| Training | `--grad_clip` | 0.5 | gradient clipping |
| System | `--device` | cuda:0 | cpu or cuda:0 |

## Test Data SNR

Test data is generated with `snr='noiseless'` (hardcoded in main.py line ~347), while training data uses `snr=20` (hardcoded line ~328). Success criterion: DOA angle RMSE < 3.0°.

## Output Logs

- Console output logged to `shuju.txt` (tab-separated: Epoch, TrainLoss, Freq_MAE, SR, RMSE_suc, RMSE_all)
- Experiment logs in `logs_lambda/`, `logs_lr/`, `logs_samples/`

## Known Issues

- **Success rate near zero (0~0.016)** across all experiments. Training loss decreases but frequency estimation does not improve. Root cause: `loss_lambda=100` causes structure_loss (H_target matching) to dominate over sign consistency, and exact amplitude recovery from one-bit data is information-theoretically impossible. See `issues.md`.
- **b=0.01 in squareplus** is 10x larger than the paper's 0.001, making the smooth gradient approximation less accurate.
- **K=5 layers** is far fewer than the paper's 100 iterations for traditional OSCAR.
