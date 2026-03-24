# OnebitUnrolledNet

基于深度展开（Deep Unfolding）的单快拍单比特谱压缩感知实现。核心思想来自《神经网络流程.docx》：
- 利用符号一致性（one-bit sign consistency）约束
- 利用 Hankel 升维后的低秩结构
- 将迭代优化过程展开为“初始化层 + K 个展开层（梯度下降模块 + 约束满足模块）”

## 1. 文档与代码逻辑对照

### 1.1 已对应实现
- 展开结构：`main.py` 中 `SpectralDeepUnfolding` 使用 `K` 层，每层包含：
  - `GradientModule`（`networks/Modules.py`）
  - `ConstraintModule`（`networks/Modules.py`）
- 初始化：`utils/misc.py` 的 `spectral_init` 对 Hankel 矩阵做 SVD 并归一化。
- 符号一致性损失：`main.py` 的 `spectral_loss` 使用 `relu(- y .* x_hat)` 的二范数平方惩罚。
- Hankel 结构监督：训练中使用 `H_target`（由干净信号 Hankel 化得到）做 Frobenius 误差项。

### 1.2 与文档存在差异（当前版本）
- 文档中给出的是以约束/惩罚为核心的非监督优化形式；当前训练是“监督式”结构损失（依赖 `H_target`）。
- 文档提到可比较多种平滑函数（SoftPlus/SquarePlus/Swish）；当前梯度模块固定使用 SquarePlus。
- 文档提到逐层学习（layer-wise）；当前实现为端到端训练所有展开层参数。

结论：当前代码与文档主流程一致，但属于“文档方法的一个可训练化实现版本”，不是完全一一等式级复现。

## 2. 本次检查中修复的问题

为保证“实验可用、指标可解释”，修复了以下逻辑问题：
- 修复 `MatPencilMethod` 频率计算不稳定：
  - 从 `log(eigenvalue)` 改为基于相角 `angle` 的频率恢复，避免 `NaN`。
  - 文件：`utils/misc.py`
- 修复损失重复按 batch 缩放：
  - `spectral_loss` 原本已按 batch `mean`，不应再除以 `args.batch_size`。
  - 文件：`main.py`
- 修复训练 `DataLoader` 未使用命令行 `--batch_size` 参数。
  - 文件：`main.py`
- 修复测试频率误差计算：
  - 使用有限值掩码过滤并计算平均绝对误差，避免 `NaN` 传播。
  - 文件：`main.py`
- 去除测试阶段硬编码参数：
  - 将 `MatPencilMethod_Batch(X_hat, 3, 32)` 改为 `MatPencilMethod_Batch(X_hat, args.rank, args.matrix_row)`。
  - 并按 batch 对齐 `F_true`，避免多 batch 时指标错位。
  - 文件：`main.py`

## 3. 环境与安装

建议使用 Python 3.11 + Conda。

安装依赖：
```powershell
conda install -y pytorch cpuonly -c pytorch
python -m pip install numpy scipy scikit-learn
```

若出现 OpenMP 冲突（`libiomp5md.dll already initialized`），运行前设置：
```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

## 4. 运行训练与测试

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python main.py --device cpu --num_epochs 3 --num_samples 128 --batch_size 16 --num_layers 3 --signal_dim 32 --rank 3
```

参数说明：
- `signal_dim`：信号长度
- `rank`：谱稀疏度/分解秩
- `num_layers`：展开层数
- `num_samples`：训练样本数
- `num_epochs`：训练轮数
- `batch_size`：批大小

## 5. 本次实验结果（2026-03-11）

实验命令：
```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python main.py --device cpu --num_epochs 3 --num_samples 128 --batch_size 16 --num_layers 3 --signal_dim 32 --rank 3
```

关键输出：
- 训练损失（示例）：
  - Epoch 1 Batch 0: `81602.2516`
  - Epoch 2 Batch 0: `80713.7185`
  - Epoch 3 Batch 0: `79683.2751`
- 测试：
  - `Freq Error: 0.1846`
  - `Average Test Loss: 79587.2521`

说明：
- 当前小规模实验表明训练可运行且损失下降。
- 频率误差仍偏大，后续可通过更长训练、层数扫描、平滑函数/初始化策略对比来进一步优化。

