# issues

## 当前状态

- 改法 2（去掉 spectral_init 最后一行归一化）已应用，效果无明显变化
- n5000 lambda sweep 正在进行中（4 GPU 并行），目前仅跑到 epoch ~20
- SR 仍然接近 0，RMSE_all ~26-30°

## 待验证的改法

### 改法 A：H_target 归一化（优先级更高）

**位置**：`main.py` 的 `spectral_loss` 函数

**做法**：在计算 structure_loss 之前，把 `H_target` 按 Frobenius 范数归一化到大小=1：
```python
H_target_norm = H_target / (H_target.norm(p='fro', dim=(1,2), keepdim=True) + 1e-8)
structure_loss = mean(||H_hat - H_target_norm||²_fro)
```

**理由**：当前 H_hat（由 one-bit 数据驱动）的尺度始终 ≈ 1，而 H_target（干净信号）的尺度 ≈ 55。structure_loss 的梯度大小由 `2*(H_hat - H_target)` 决定，尺度差 50 倍导致这个梯度完全由 H_target 的绝对值主导，而不是由 H_hat 和 H_target 之间的**结构差异**主导。归一化后两者尺度相同，structure_loss 才能真正比较 Hankel 矩阵的结构相似性而非幅度匹配。

### 改法 B：b 从 0.01 改为 0.001

**位置**：`networks/Modules.py` 第 33、36 行

**做法**：`squareplus_function(w, b=0.001)` 和 `squareplus_derivative(w, b=0.001)`

**理由**：b 控制 squareplus 平滑近似的精度。b=0.01 是论文值（0.001）的 10 倍，导致函数在零点附近过于平缓，降低了梯度对符号不一致条目的敏感度。改回 0.001 可以提高 sign consistency 梯度的信噪比约 3 倍。

## 建议验证顺序

1. 先单独测改法 A（改动最小，风险最低）
2. 再单独测改法 B
3. 两者一起测
4. 每次用 `--num_samples 1000 --num_epochs 10` 快速验证 SR 是否有变化，确认有效后再跑完整实验
