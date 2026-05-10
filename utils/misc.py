
import numpy as np
import torch
from typing import Tuple
from .hankel_operators import HankelOperator

import torch
import numpy as np

def MatPencilMethod(s, r, L):
    """
    基于矩阵铅笔法的单条信号频谱估计。

    参数:
        s : 1D numpy array 或张量，一维观测信号
        r : 需要估计的谱线条数 / 模型阶数
        L : Hankel 矩阵的列数（铅笔参数）

    返回:
        a : 估计的复振幅，shape [r]
        f : 估计的归一化频率，shape [r]，范围 [0, 1)
    """
    s = torch.as_tensor(s, dtype=torch.cfloat)
    N = s.numel()

    # 构造 Hankel 形式的 X1, X2
    X1 = torch.stack([s[L - i - 1:N - i - 1] for i in range(L)], dim=1)
    X2 = torch.stack([s[L - i:N - i] for i in range(L)], dim=1)

    # 求解广义特征问题
    A = torch.linalg.pinv(X1) @ X2
    D, U = torch.linalg.eig(A)  # D: 特征值, U: 右特征向量

    # 由特征值构造 Vandermonde 矩阵，估计振幅
    dg = D
    Vd = torch.stack([dg ** i for i in range(L)], dim=0)  # Vandermonde 矩阵

    a = torch.linalg.pinv(Vd) @ s[:L]

    # 按振幅大小排序并筛选主分量
    abs_a = torch.abs(a)
    index_a = torch.argsort(abs_a, descending=True)
    at = a[index_a]
    dg = dg[index_a]

    ft = torch.remainder(-torch.angle(dg) / (2 * torch.pi), 1.0)

    # 映射到 [0,1) 区间，并做一次筛选
    fe = ft.clone()
    fe = fe[:round(L * 0.5)]
    fe[fe < 0] += 1.0

    abs_fe = torch.abs(fe)
    index_f = torch.argsort(abs_fe)
    ac = at[index_f]
    fc = fe[index_f]

    # 依据能量和频率间隔选择 r 个主频（对应 MATLAB 中的启发式规则）
    index = []
    if torch.norm(at[:r]) ** 2 / torch.norm(at) ** 2 >= 0.50:
        for i in range(r):
            index.append((index_f == i).nonzero(as_tuple=True)[0].item())
    else:
        j = 0
        first_index = (index_f == 0).nonzero(as_tuple=True)[0].item()
        index.append(first_index)
        for i in range(1, round(L * 0.5)):
            index_t = (index_f == i).nonzero(as_tuple=True)
            if len(index_t[0]) == 0:
                continue
            idx = index_t[0].item()
            cond1 = torch.abs(ac[idx]) >= torch.abs(ac[idx - 1])
            cond2 = torch.abs(ac[idx]) >= torch.abs(ac[(idx + 1) % len(ac)])
            cond3 = all(torch.abs(fc[idx] - fc[i0]) >= 1 / (2 * L) for i0 in index)
            if cond1 and cond2 and cond3:
                index.append(idx)
                j += 1
            if j == r-1:
                break

    # 输出最终按频率排序的振幅和频率
    ad = ac[index]
    fd = fc[index]

    indexf2 = torch.argsort(torch.abs(fd))
    a_final = ad[indexf2]
    f_final,_ = torch.sort(1 - fd[indexf2])
    
    return a_final, f_final

# 下面保留了一份基于矩阵铅笔法的早期备选实现（已注释），如需对比不同实现可从 git 历史中恢复。

def MatPencilMethod_Batch(s_batch, r, L):
    """
    对一批 1D 信号批量应用矩阵铅笔法，估计每条信号的振幅和频率。

    参数:
        s_batch : 信号批次，形状 (batch_size, N)
        r       : 需要估计的谱线条数 / 模型阶数
        L       : Hankel 矩阵的列数（铅笔参数）

    返回:
        a_batch : 每条信号的复振幅，形状 (batch_size, r)
        f_batch : 每条信号的归一化频率，形状 (batch_size, r)
    """
    batch_size, N = s_batch.shape
    device = s_batch.device
    dtype = s_batch.dtype
    
    # 将输入统一转为复数类型
    if not s_batch.is_complex():
        s_batch_complex = torch.view_as_complex(
            torch.stack([s_batch, torch.zeros_like(s_batch)], dim=-1)
        )
    else:
        s_batch_complex = s_batch
    
    # 为输出预分配张量
    a_batch = torch.zeros((batch_size, r), dtype=dtype, device=device)
    f_batch = torch.zeros((batch_size, r), dtype=torch.float32, device=device)

    for i in range(batch_size):
        a_batch[i], f_batch[i] = MatPencilMethod(s_batch_complex[i], r, L)
    
    return a_batch, f_batch
def spectral_init(x: torch.Tensor, L: int, r: int):
    """
    谱初始化:
      1) 先对每个样本的 Hankel 矩阵按 Frobenius 范数归一化
      2) 再做 SVD，并用前 r 个奇异值/向量构造 U 作为初始化

    参数:
        x:  输入一维信号, 形状 [B, N]
        L:  Hankel 行数
        r:  目标秩 / 源数

    返回:
        U: Hankel 低秩分解的谱初始化因子, 形状 [B, L, r]
        [Modified] Removed V for U*U^T symmetry
    """
    B, N = x.shape
    K = N - L + 1

    # 构造 Hankel 矩阵 H: [B, L, K]
    H = torch.stack([x[:, i:i + K] for i in range(L)], dim=1)

    # 先对 Hankel 矩阵按 Frobenius 范数归一化，便于谱分解稳定
    H_norm = H.norm(p="fro", dim=(1, 2), keepdim=True) + 1e-8
    H_scaled = H / H_norm

    # 对归一化后的 Hankel 做 SVD
    U, S, Vh = torch.linalg.svd(H_scaled, full_matrices=False)  # [B, L, K] batched SVD

    # 取前 r 个奇异值/向量，并用 sqrt(S) 吸收到 U 中
    Sr = torch.sqrt(S[:, :r]).unsqueeze(1)  # [B, 1, r]
    U_r = U[:, :, :r] * Sr                   # [B, L, r]
    # [Modified] Removed V_r generation for U*U^T symmetry

    return U_r

def hankel_lifting(x, M, N):
    """
    将一维序列 x 映射为 Hankel 结构矩阵 H(x)。

    参数:
        x : [B, M+N-1] 的实/复���量
        M : Hankel 行数
        N : Hankel 列数

    返回:
        H : [B, M, N] 的 Hankel 矩阵
    """
    B = x.shape[0]
    out = torch.zeros(B, M, N, dtype=x.dtype, device=x.device)

    for i in range(M):
        for j in range(N):
            out[:, i, j] = x[:, i + j]
    # norm = out.norm(p='fro', dim=(1, 2), keepdim=True)
    return out

def Dinv_G_adjoint(H):
    """
    Hankel 伴随算子并施加对角加权 D^{-1/2}。

    参数:
        H : [B, M, N]，输入 Hankel 矩阵

    返回:
        x : [B, M+N-1]，加权后的反投影向量
    """
    B, M, N = H.shape
    n = M + N - 1
    H_adj = torch.zeros(B, n, device=H.device, dtype=H.dtype)
    for i in range(M):
        for j in range(N):
            H_adj[:, i + j] += H[:, i, j]
    weights = [min(i + 1, M, N, M + N - 1 - i) for i in range(M + N - 1)]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    D_sqrt_inv=1.0 / torch.sqrt(weights_tensor)
    D_sqrt_inv=D_sqrt_inv.to(H.device)
    return H_adj * D_sqrt_inv[None, :] * D_sqrt_inv[None, :]

def awgn(signal, snr, measured=True):
    """
    向信号中加入复/实高斯白噪声 (AWGN)。

    参数:
        signal   : ndarray，输入信号（实数或复数）
        snr      : float，信噪比 dB
        measured : bool，True 表示按信号实测功率加噪，False 则假定信号功率为 1

    返回:
        noisy_signal : 加噪后的信号（与输入同形状）
    """
    if measured:
        # 璁＄畻淇″彿鍔熺巼
        s_power = np.mean(np.abs(signal)**2)
    else:
        s_power = 1.0
    
    # 璁＄畻鍣０鍔熺巼
    target_snr = 10**(snr/10)
    n_power = s_power / target_snr
    
    # 鐢熸垚鍣０
    if np.iscomplexobj(signal):
        noise = np.sqrt(n_power/2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    else:
        noise = np.sqrt(n_power) * np.random.randn(*signal.shape)
    
    return signal + noise
