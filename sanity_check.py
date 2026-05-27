"""Sanity check: clean signal -> MatPencilMethod -> verify SR ~ 1.0"""
import torch
import numpy as np
import sys
sys.path.insert(0, '.')
from utils.generate_signal_1D import generate_signal_batch
from utils.misc import MatPencilMethod_Batch

signal_dim = 63
rank = 3
batch_size = 128
L = signal_dim // 2 + 1  # matrix_row = 32

result = generate_signal_batch(
    batch_size, signal_dim, rank,
    separation=True, damp=False, snr='noiseless', random_seed=42
)

clean_signals = torch.from_numpy(result['clean'])
true_freqs = torch.from_numpy(result['freqs'])

print(f'Signal shape: {clean_signals.shape}')
print(f'Sample 0 true freqs: {np.sort(true_freqs[0].numpy())}')

clean_complex = clean_signals

a_hat, f_hat = MatPencilMethod_Batch(clean_complex, rank, L)

print(f'Sample 0 est  freqs: {f_hat[0].numpy()}')

threshold = 3.0 / 180.0
true_sorted = np.sort(true_freqs.numpy(), axis=1)

n_success = 0
all_errors = []

for i in range(batch_size):
    f_est_i = f_hat[i].numpy()
    f_true_i = true_sorted[i]
    errors = []
    for fe in f_est_i:
        diff = np.min(np.abs(fe - f_true_i))
        diff = min(diff, 1.0 - diff)
        errors.append(diff)
    max_error = np.max(errors)
    if max_error < threshold:
        n_success += 1
    all_errors.extend(errors)

sr = n_success / batch_size
rmse_deg = np.sqrt(np.mean(np.array(all_errors)**2)) * 180.0

print(f'\n=== RESULTS ===')
print(f'Success Rate (SR): {sr:.4f} ({n_success}/{batch_size})')
print(f'RMSE: {rmse_deg:.6f} deg')
print(f'Mean Freq Error: {np.mean(all_errors):.6f} (normalized)')

print(f'\nFirst 5 samples:')
for i in range(5):
    f_est_sorted = np.sort(f_hat[i].numpy())
    print(f'  True: {true_sorted[i]}, Est: {f_est_sorted}')
