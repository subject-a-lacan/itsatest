问题不在超参，最硬的根因在 后处理/评估链路把频率翻反了。

我做了一个 sanity check：不用网络，直接把干净真值信号送进当前 MatPencilMethod_Batch。结果是：

text



clean MatPencil SR/RMSE/FMAE: 0.0234375, 28.32°, 0.1925



也就是 干净真值都只能对 3/128。这和你说的“最好只能 128 个样本里对 3 个”完全吻合。再把输出频率做一次 1 - F_hat 后：

text



complement corrected: SR=1.0, RMSE≈0.000006°



第一条样本也能看出来：

text



true: [0.2647, 0.5610, 0.7055]
hat : [0.2945, 0.4390, 0.7353]  # 正好是 1 - true



原因是数据生成用了负指数：

utils/generate_signal_1D.py (line 97)

python



np.exp(-2j * np.pi * m_indices[:, np.newaxis] * f)



而 MatPencilMethod 里频率相位在这里其实已经按这个符号恢复了：

utils/misc.py (line 46)

python



ft = torch.remainder(-torch.angle(dg) / (2 * torch.pi), 1.0)



但最后又翻了一次：

utils/misc.py (line 87)

python



f_final,_ = torch.sort(1 - fd[indexf2])



这里的 1 - 应该先去掉，至少对当前 damp=False 的主训练设置是这样：

python



f_final, _ = torch.sort(fd[indexf2])



DeepSeek 说的 loss 不对齐也有道理，但它不是“3/128”的第一根因。因为当前评估链路坏到这种程度：网络就算输出完美干净信号，也会被评估成几乎全错。我还测了 one-bit 谱初始化：按旧评估只有 2/128 左右，频率翻正后已经有 83/128 成功，这说明模型/初始化并没有你现在看到的那么烂。

后续顺序建议：

先修 utils/misc.py:87 的频率翻转。
跑 clean-signal sanity check，确认真值输入 SR 接近 1.0。
再重新看训练 SR。
然后再处理 loss：main.py:70 的 H_hat - H_target 确实有尺度问题，H_target norm 约 55，初始化 H_hat norm 约 0.84，归一化监督是合理的下一步。另一个小坑是 --loss_lambda 只影响外部 loss，不影响展开层里的 lambda_reg=0.01。