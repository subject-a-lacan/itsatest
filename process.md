# 进展记录

## 阶段 1：锁死 SR≈0

所有实验 SR 在 0~0.16，训练损失下降但频率估计完全不灵。

## 阶段 2：定位评测 bug

发现 `utils/misc.py:87`：`f_final = 1 - fd` 把正确恢复的频率全翻转了。
修复后 clean signal 进 MatPencilMethod → SR=1.0。根因确认。

## 阶段 3：lambda 扫 (lr=1e-3, n=5000, ep=30)

SR 从 0 跳到 0.10~0.30。lambda=5.0 最好 (0.297)，lambda=0.01 其次 (0.266)。

## 阶段 4：网格扫 lambda×K (lr=5e-4, n=100000, ep=30)

9 lambda × 6 K = 54 实验。
- K=8 可以，K≥9 基本炸
- lambda6_K8 SR=0.469 (最高)
- lambda 只要不炸，SR 全在 0.35~0.47，模型对 lambda 不敏感

## 阶段 5：深网络突破 (lr=2e-4/5e-4, n=40000, ep=100)

4 lambda × (K=8@2lr + K=9,10@2e-4) = 16 实验，进行中。
- SR **首次破 0.5**：lambda=0.03, K=10, lr=2e-4 → **0.508**
- lr=2e-4 让 K=9,10 全部干净存活，零 INF
- 新问题：后期过拟合（SR 冲高后跌回），需 lr scheduler
