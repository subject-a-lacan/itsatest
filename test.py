import numpy as np
from utils.misc import awgn

# 测试复数信号加噪
x_test = np.exp(1j * np.linspace(0, 2*np.pi, 100))
noisy = awgn(x_test, snr=10)

# 测试实数信号加噪
x_test_real = np.sin(np.linspace(0, 2*np.pi, 100))
noisy_real = awgn(x_test_real, snr=15)

# 验证功率比
def calculate_snr(original, noisy):
    signal_power = np.mean(np.abs(original)**2)
    noise_power = np.mean(np.abs(noisy - original)**2)
    return 10 * np.log10(signal_power / noise_power)

print(f"实测SNR: {calculate_snr(x_test, noisy):.2f} dB")
print(f"实测SNR: {calculate_snr(x_test_real, noisy_real):.2f} dB")