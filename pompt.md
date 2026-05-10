# OSCAR Deep Unrolled Network — 诊断报告

## 背景

项目是基于 OSCAR 论文的 one-bit 单快照谱压缩感知深度展开网络。模型包含 K 个展开层（每层 = GradientModule + ConstraintModule），输入是 1-bit 测量值 y ∈ {±1}，输出重建的 Hankel 矩阵，最后通过 Matrix Pencil 方法恢复频率。

**核心现象**：所有实验中 Success Rate (SR, 即 DOA 角度误差 < 3°) 都在 0.000~0.016 之间，无论怎么调参都不动。Loss 从 300 降到 63 但 SR 始终在 0.016。

## 排查过程

### 嫌疑人 #1：学习率太小 ❌ 已排除

**初始怀疑**：默认 lr=3e-5 太小，模型几乎学不动。

**验证过程**：
1. 最早一批实验 (logs_lr/) lr 扫了 1e-5 ~ 7e-5 (K=5, lambda=100, n=5000)，SR 全部 ≤ 0.008，loss 从 301,000 降到 280,000~295,000，几乎没有下降。
2. 但这批实验 lambda=100（默认值），structure_loss 权重太大，不能单独归因于 lr。
3. 后来扫了 lr 从 1e-4 到 1e-2 (lambda=0.1, K=10, n=5000)：

| lr | 最终 SR | Loss 轨迹 | 最终状态 |
|------|---------|-----------|---------|
| 1e-4 | SR=0.008 | 300→103 (epoch 60) | 正常收敛 |
| 3e-4 | SR=0.000 | 300→5 (epoch 50) | 正常收敛，loss 极低但 SR=0 |
| 1e-3 | SR=0.016 | 300→63 (epoch 13) | 梯度爆炸 (inf)，提前终止 |
| 3e-3 | — | 300→120 (epoch 3) | 梯度爆炸，提前终止 |
| 1e-2 | — | 302→NaN (epoch 2) | 立即 NaN |

**关键证据**：lr=1e-3 时 loss 下降 5 倍快到 63，但 SR 还是 0.016，和 lr=3e-5 时一模一样。

4. 回头看原始版本代码 (C:\Users\19355\Desktop\OnebitUnrolledNet-master\)：lr=0.001（比现在的 3e-5 大 33 倍），eta_init=0.1（比现在大 100 倍），效果依然烂。

**结论**：学习率不是根因。提高学习率可以让 loss 快速下降，但降的是 structure_loss，和频率估计没关系。

---

### 嫌疑人 #2：梯度裁剪 (grad_clip=0.5) ❌ 已排除

**初始怀疑**：grad_clip=0.5 裁剪过于激进，阻碍了梯度正常流动。

**验证过程**：
1. Adam 优化器内部有自适应学习率（二阶矩估计归一化），对梯度幅值的大幅变化不敏感。
2. 原始版本代码完全没有梯度裁剪，效果依然烂。
3. lr sweep 中 lr=1e-3 的实验：即使有梯度裁剪，loss 依然快速从 300 降到 63，说明梯度裁剪没有阻碍 loss 的优化。

**结论**：梯度裁剪不是导致 SR 差的原因。

---

### 嫌疑人 #3：squareplus 的 b=0.01 太大（b 应为 0.001）△ 次要因素

**初始怀疑**：b=0.01 比论文原文的 0.001 大 10 倍，导致 squareplus 在零点附近太平坦，梯度无法区分"符号正确"和"符号错误"的条目。

**详细分析**：

squareplus 公式：`squareplus(w) = (w - √(w² + b)) / 2`

当 w 的典型值 ~0.014，√b = 0.1：
- w = +0.014 (符号正确)：squareplus ≈ (0.014 - 0.101) / 2 = -0.043
- w = -0.014 (符号错误)：squareplus ≈ (-0.014 - 0.101) / 2 = -0.057

区分度只有 1.33x。如果 b=0.001，√b = 0.0316：
- w = +0.014：squareplus ≈ (0.014 - 0.0345) / 2 = -0.010
- w = -0.014：squareplus ≈ (-0.014 - 0.0345) / 2 = -0.024

区分度 ~2.4x。如果 w 能到 0.1（b=0.001 时）：
- w = +0.1：squareplus ≈ (0.1 - 0.141) / 2 = -0.020
- w = -0.1：squareplus ≈ (-0.1 - 0.141) / 2 = -0.121

区分度 ~6x。

w 为什么小？因为 w = Re(y ∘ (D⁻¹ · G* · X))，其中 X = U@U^T。U 来自 spectral_init（SVD 初始化），其条目约 0.07，X 条目约 0.014。经过 Hankel 伴随算子和 D⁻¹ 加权后，w 保持 ~0.014 的量级。

**但是**：b=0.01 能解释梯度对比度从 6x 降到 1.33x，但不能解释 SR 为什么是 0。1.33x 的对比度虽然弱，但如果有足够多轮迭代，方向还是对的。原始版本 b 可能就是 0.001（待确认），但原始版本也失败了。

**结论**：b=0.01 降低了 consistency_loss 梯度的信噪比，是贡献因素但不是根因。

---

### 嫌疑人 #4：spectral_init 最后的归一化 ❌ 已排除

**初始怀疑**：`spectral_init()` 最后一行 `U_norm = U_r / U_r.norm(...)` 把 U 的 F-norm 归一化到 1（每个 batch 独立），导致 U 的条目太小。

**已实施修改**：去掉了最后的 F-norm 归一化，直接返回 U_r。

**结果**：几乎无变化。因为前面已经做了 `H_scaled = H / H.norm(p='fro', dim=(1,2), keepdim=True)`，SVD 出来的 U_r 自然 F-norm < 1，去掉归一化只是少了 ~1.4x。

**结论**：不是根因。

---

## 真正的根因：Loss 函数与目标任务不对齐

### 当前 Loss 的结构

```python
consistency_loss = mean( ||relu(-Re(y · x_hat), -Im(y · x_hat))||² ) / 2
structure_loss = mean( ||H_hat - H_target||²_fro )
total = consistency_loss + loss_lambda * structure_loss
```

- `consistency_loss`：惩罚那些不满足 1-bit 符号一致性的重建。**这个 loss 直接和"频率对不对"有关**，因为频率错误会导致符号预测错误。
- `structure_loss = ||H_hat - H_target||²_fro`：惩罚重建的 Hankel 矩阵和干净信号 Hankel 矩阵之间的 Frobenius 距离。**H_target 是原始幅值的 Hankel 矩阵**（F-norm ≈ 55），而 **H_hat 的 F-norm 始终 ≈ 1**（因为从 one-bit 数据初始化，后续 forward pass 保持 ~1 量级）。

### 为什么 structure_loss 主导了一切

在初始化时：
- consistency_loss ≈ 0.64（所有条目都是 ±1 的 one-bit 数据，随机初始化时约一半不符合符号约束）
- structure_loss = ||H_hat - H_target||²_fro ≈ (55)² ≈ 3000

即使 lambda=0.1，structure_loss 项 ≈ 300，consistency_loss ≈ 0.64。**structure_loss 的梯度比 consistency_loss 大 ~500x**。

模型在训练时，Adam 看到的主要信号是"让 H_hat 接近 H_target"。但 H_target 的 F-norm 是 55，H_hat 的 F-norm 是 1，优化器最简单的做法是：

1. 先让 H_hat 的 F-norm 从 1 膨胀到 ~55（快速降低 loss 的主要途径）
2. 再调整 H_hat 的结构去匹配 H_target

而第 1 步（幅度匹配）完全不需要频率信息正确。这就是为什么 loss 从 300 降到 63 但 SR 纹丝不动——模型只是在做幅度匹配。

### 为什么 original 版本也失败

原始版本虽然参数设置不同（lr=0.001, eta=0.1, 无梯度裁剪），loss 同样以 structure_loss 为主导，面临相同的 loss 不对齐问题。learning rate 大只是让它更快地收敛到一个"H_hat F-norm 接近 H_target"但对频率估计无用的解。

### 可观测的证据汇总

1. **lambda sweep** (K=10, n=5000/10000, lr=3e-5): lambda=0.1~5.0，SR 全部 ≤ 0.008。lambda 越小 structure_loss 应该越不重要，但 lambda=0.1 时 structure_loss 仍然比 consistency_loss 大 ~500x，还是主导。
2. **lr sweep**: lr=1e-3 的 loss 下降最快（300→63），但 SR 仍然只有 0.016。说明 loss 下降确实不代表频率估计变好。
3. **lr=3e-4 的极端案例**: loss 降到了 5，但 SR=0.000。模型几乎完美地重建了 H_target 的幅度，但频率估计完全错误。
4. **老实验 lambda=100**: loss 在 300,000 量级（structure_loss × 100），SR ≤ 0.008。loss 几乎全部是 structure_loss，consistency_loss 的信号完全被淹没。

---

## 修复方案（按优先级排列）

### 方案 A：在 loss 中归一化 H_target（最关键）

**位置**：`main.py` 的 `spectral_loss` 函数

**做法**：
```python
H_target_norm = H_target / (H_target.norm(p='fro', dim=(1,2), keepdim=True) + 1e-8)
structure_loss = mean( ||H_hat - H_target_norm||²_fro )
```

**预期效果**：H_target_norm 的 F-norm = 1，和 H_hat 的 F-norm 在同一量级。structure_loss ≈ 1~2，consistency_loss ≈ 0.64，两者量级对等。此时 structure_loss 才能真正比较 Hankel 矩阵的**结构相似性**而不是幅度匹配。consistency_loss 的梯度才能被模型"听到"。

### 方案 B：将 squareplus 的 b 从 0.01 改为 0.001

**位置**：`networks/Modules.py` 中 squareplus 的 b 参数

**做法**：`b=0.01` → `b=0.001`

**预期效果**：恢复 squareplus 对符号正负的辨别力，梯度对比度从 1.33x 提升到 ~2-6x（取决于 w 的量级）。

### 建议验证顺序

1. 先只做方案 A，跑一次快速实验 (--num_samples 1000 --num_epochs 10 --loss_lambda 0.1)，看 SR 是否有变化
2. 如果 SR 提升但仍不理想，再加方案 B 一起测
3. 确认有效后再跑完整参数扫描

---

## 实验汇总数据

### Lambda Sweep (K=10, lr=3e-5, n=10000, ep=100)
| lambda | SR max | RMSE_all | 备注 |
|--------|--------|----------|------|
| 0.1 | 0.000 | 28.9° | 跑满 100 epoch |
| 0.5 | 0.000 | 30.0° | 跑满 100 epoch |
| 1.0 | 0.008 | 31.8° | epoch 60 停止 |
| 5.0 | 0.008 | 31.2° | epoch 50 停止 |

### LR Sweep (K=10, lambda=0.1, n=5000, ep=100)
| lr | SR max | Loss 最终 | 状态 |
|------|--------|-----------|------|
| 1e-4 | 0.008 | ~103 (ep60) | 正常运行 |
| 3e-4 | 0.000 | ~5 (ep50) | 正常运行，loss 极低但 SR=0 |
| 1e-3 | 0.016 | ~63 (ep13) | grad_norm=inf，提前终止 |
| 3e-3 | — | ~120 (ep3) | grad_norm=inf，提前终止 |
| 1e-2 | — | NaN (ep2) | 立即 NaN |

### 早期 LR Sweep (K=5, lambda=100, n=5000, ep=30)
| lr | SR max | Loss 范围 | 备注 |
|------|--------|-----------|------|
| 1e-5 | 0.000 | 301,000→296,000 | 几乎学不动 |
| 3e-5 | 0.008 | 301,000→277,000 | 缓慢下降 |
| 5e-5 | 0.000 | — | 类似 |
| 7e-5 | 0.000 | — | 类似 |

**注意**：lambda=100 时 loss 是 30 万量级（structure_loss × 100），consistency_loss 完全被淹没。
