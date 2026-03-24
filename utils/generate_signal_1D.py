import numpy as np
from sklearn.utils import check_random_state


def generate_signal_1D(nd, r, separation, damp, snr, random_state=None):
    """
    生成1D谱稀疏信号（Python版本，参考 MATLAB generate_signal_1D.m 实现）

    参数:
    ----------
    nd : int
        信号长度
    r : int
        频谱稀疏度（频率分量数量）
    separation : {bool, str, int, float}
        频率间隔控制:
        - False/'false'/0 : 无间隔
        - True/'true'/1 : 保证最小间隔
        - 'fixed1'/'fixed2'/'fixed3' : 使用预设固定频率
        - 数值: 自定义间隔（目前按 2 个源设计）
    damp : {bool, str, int}
        衰减控制:
        - False/'false'/0 : 无衰减
        - True/'true'/1 : 指数衰减
    snr : {float, 'noiseless'}
        信噪比(dB)，'noiseless'表示无噪声
    random_state : int or None
        随机种子

    返回:
    -------
    xs : ndarray, shape (nd,)
        观测信号（含噪声）
    x_star : ndarray, shape (nd,)
        纯净信号
    f : ndarray, shape (r,)
        归一化频率
    amp : ndarray, shape (r,)
        频率分量幅度
    """
    rng = check_random_state(random_state)

    # 按给定 seed 驱动 numpy 全局状态，以兼容直接使用 np.random 的实现
    np.random.set_state(rng.get_state())

    # 1. 生成频率分量（参考 MATLAB 版本） =====================================
    # 幅度固定为 1
    amp = np.ones(r, dtype=np.complex128)

    # separation 行为
    if separation in [False, 'False', 'false', 0]:
        # 从 0..nd-1 中随机选 r 个，归一化
        freq_seed = np.random.choice(nd, r, replace=False) / nd

    elif separation in [True, 'True', 'true', 1]:
        d1 = 1.5 / nd
        E1 = 1 - r * d1
        if E1 < 0:
            raise ValueError("模型阶数过高，无法满足频率最小间隔条件")
        rand_vals = np.random.rand(r + 1)
        fs = E1 * rand_vals[:r] / np.sum(rand_vals)
        fs = d1 * np.ones(r) + fs
        freq_seed = np.cumsum(fs)

    elif separation == 'fixed1':
        freq_seed = np.array([(np.sin(np.deg2rad(20.3)) + 1) / 2,
                              (np.sin(np.deg2rad(22.5)) + 1) / 2])
        if r != 2:
            print("Warning: 'fixed1' separation is designed for r=2 sources.")

    elif separation == 'fixed2':
        freq_seed = np.array([(np.sin(np.deg2rad(-20.3)) + 1) / 2,
                              (np.sin(np.deg2rad(24.5)) + 1) / 2])
        if r != 2:
            print("Warning: 'fixed2' separation is designed for r=2 sources.")

    elif separation == 'fixed3':
        freq_seed = np.array([(np.sin(np.deg2rad(-56.3)) + 1) / 2,
                              (np.sin(np.deg2rad(-15.4)) + 1) / 2,
                              (np.sin(np.deg2rad(30.5)) + 1) / 2])
        if r != 3:
            print("Warning: 'fixed3' separation is designed for r=3 sources.")

    else:
        # 数值分离：目前按 2 源设计
        sep_val = float(separation)
        freq_seed = np.array([(np.sin(np.deg2rad(10)) + 1) / 2,
                              (np.sin(np.deg2rad(10 + sep_val * 1.2)) + 1) / 2])
        if r != 2:
            print("Warning: numerical separation is designed for r=2 sources.")

    f = np.sort(freq_seed)

    # 2. 构建无噪声信号 x_star（含/不含衰减） ===============================
    m_indices = np.arange(nd)
    if damp in {False, 'false', 'False', 0}:
        steering_matrix = np.exp(-2j * np.pi * m_indices[:, np.newaxis] * f)
        x_star = steering_matrix @ amp
    elif damp in {True, 'true', 'True', 1}:
        d1 = 16 + 16 * np.random.rand(r)
        d1 = -1.0 / d1
        damping_term = np.outer(m_indices, d1)
        freq_term = np.outer(m_indices, 1j * 2 * np.pi * f)
        x_star = np.exp(damping_term + freq_term) @ amp
    else:
        raise ValueError("damp 参数必须为 True/False 或 0/1")

    # 3. 添加噪声，得到观测信号 xs（Python 版 awgn） =========================
    if snr == 'noiseless':
        xs = x_star
    else:
        signal_power = np.mean(np.abs(x_star) ** 2)
        snr_linear = 10 ** (float(snr) / 10.0)
        noise_power = signal_power / snr_linear
        noise = (np.random.randn(nd) + 1j * np.random.randn(nd)) * np.sqrt(noise_power / 2)
        xs = x_star + noise

    return xs.squeeze(), x_star.squeeze(), f, amp

# def generate_signal_batch(batch_size=32, nd=256, r=3, 
#                         separation=True, damp=False, snr=20,
#                         random_seed=None):
#     """
#     函数式批量生成
#     Returns:
#         Dict[str, np.ndarray]: {
#             'observed': (batch_size, nd)观测信号,
#             'clean': (batch_size, nd)纯净信号,
#             'freqs': (batch_size, r)频率分量,
#             'amps': (batch_size, r)幅度分量
#         }
#     """
#     rng = np.random.RandomState(random_seed)
#     return {
#         k: np.stack([
#             generate_signal_1D(nd, r, separation, damp, snr, 
#                              random_state=rng.randint(0, 2**10))[i] 
#             for _ in range(batch_size)
#         ], axis=0)
#         for i, k in enumerate(['observed', 'clean', 'freqs', 'amps'])
#     }

def generate_signal_batch(batch_size=32, nd=256, r=3,
                          separation=True, damp=False, snr=20,
                          random_seed=None, debug=False):
    """
    重构后的信号批量生成函数，增加中间变量便于调试
    
    参数:
        batch_size: 批量大小
        nd: 信号长度
        r: 频率成分数量
        separation: 是否强制频率分离
        damp: 是否包含衰减因子
        snr: 信噪比设置。
             - 若为数值（如 20）：所有样本使用同一个 SNR（单位 dB）
             - 若为 ('noiseless'): 无噪声
             - 若为 (low, high) 元组：每个样本在 [low, high] 内均匀随机采样 SNR
        random_seed: 随机种子
        debug: 是否启用调试模式，打印详细信息
        
    返回:
        Dict[str, np.ndarray]: {
            'observed': (batch_size, nd)观测信号,
            'clean': (batch_size, nd)纯净信号,
            'freqs': (batch_size, r)频率分量,
            'amps': (batch_size, r)幅度分量
        }
    """
    # 初始化随机数生成器
    rng = np.random.RandomState(random_seed)
    
    # 预分配输出数组
    observed_signals = np.zeros((batch_size, nd), dtype=np.complex128)
    clean_signals = np.zeros((batch_size, nd), dtype=np.complex128)
    freqs_array = np.zeros((batch_size, r), dtype=np.float64)
    amps_array = np.zeros((batch_size, r), dtype=np.complex128)
    
    # 用于调试的统计信息
    diff_stats = []
    
    for batch_idx in range(batch_size):
        # 为每个样本生成唯一的随机种子
        sample_seed = rng.randint(0, 2**10)

        # 为当前样本确定 SNR：支持数值或 (low, high) 范围
        if isinstance(snr, (tuple, list)) and len(snr) == 2 and not isinstance(snr[0], str):
            snr_curr = float(rng.uniform(snr[0], snr[1]))
        else:
            snr_curr = snr

        # 生成单个信号
        signal_data = generate_signal_1D(
            nd, r, separation, damp, snr_curr, random_state=sample_seed
        )
        
        # 解包结果
        observed = signal_data[0]
        clean = signal_data[1]
        freqs = signal_data[2]
        amps = signal_data[3]
        
        # 存储结果
        observed_signals[batch_idx] = observed
        clean_signals[batch_idx] = clean
        freqs_array[batch_idx] = freqs
        amps_array[batch_idx] = amps
        
        # 调试信息：检查无噪时信号是否一致
        if debug and snr > 100:  # 高SNR表示无噪
            diff = np.abs(observed - clean)
            max_diff = np.max(diff)
            avg_diff = np.mean(diff)
            diff_stats.append((max_diff, avg_diff))
            
            if max_diff > 1e-10:
                print(f"⚠️ 样本 {batch_idx} 不一致: max_diff={max_diff:.2e}, avg_diff={avg_diff:.2e}")
                print(f"  随机种子: {sample_seed}")
                
                # 分析差异位置
                diff_idx = np.argmax(diff)
                print(f"  最大差异位置: {diff_idx}")
                print(f"  观测值: {observed[diff_idx]}")
                print(f"  纯净值: {clean[diff_idx]}")
                
                # 检查频率和振幅
                print(f"  频率: {freqs}")
                print(f"  振幅: {amps}")
    
    # 汇总调试信息
    if debug and snr > 100 and diff_stats:
        max_diffs, avg_diffs = zip(*diff_stats)
        print("\n===== 无噪信号一致性报告 =====")
        print(f"最大差异范围: {np.min(max_diffs):.2e} - {np.max(max_diffs):.2e}")
        print(f"平均差异范围: {np.min(avg_diffs):.2e} - {np.max(avg_diffs):.2e}")
        inconsistent_count = sum(1 for d in max_diffs if d > 1e-10)
        print(f"不一致样本数: {inconsistent_count}/{batch_size}")
    
    return {
        'observed': observed_signals,
        'clean': clean_signals,
        'freqs': freqs_array,
        'amps': amps_array
    }
