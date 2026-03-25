import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from utils.generate_signal_1D import generate_signal_1D,generate_signal_batch
from networks.Modules import GradientModule,ConstraintModule
from utils.misc import spectral_init
import numpy as np
from utils.misc import hankel_lifting,Dinv_G_adjoint,MatPencilMethod_Batch,MatPencilMethod
from torch.utils.data import TensorDataset, DataLoader
import argparse
#?
class SpectralDeepUnfolding(nn.Module):
    def __init__(self, M, N, rank, K=10, lambda_reg=0.01):
        super().__init__()
        self.K = K
        self.rank = rank
        self.grad_modules = nn.ModuleList([
            GradientModule(M, N, lambda_reg=lambda_reg) for _ in range(K)
        ])
        self.constraint_modules = nn.ModuleList([
            ConstraintModule(M, N, rank) for _ in range(K)
        ])

        # Learnable step size eta_k (one per unrolled layer)
        self.etas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.001)) for _ in range(K)
        ])
#改动初始步长为0.001
    def forward(self, Z, W, Y):
        """
        Z, W: Factor matrices (U, V), shapes [B, M, r], [B, N, r]
        Y: One-bit observation matrix, shape [B, M, N]
        """
        # for k in range(self.K):
        #     grad_Z, grad_W = self.grad_modules[k](Z, W, Y)
        #     eta = self.etas[k]

        #     Z_half = Z - eta * grad_Z
        #     W_half = W - eta * grad_W
        # #     Z, W = self.constraint_modules[k](Z_half, W_half)

        # # X_hat = torch.bmm(Z, torch.conj(W).transpose(1, 2))  # Reconstruct low-rank matrix estimate
        # Z = self.constraint_modules[k](Z_half)
        # X_hat = torch.bmm(Z, Z.transpose(1, 2)) 
        # #对称分解
        # return X_hat

        # 去掉 forward 里的 W 参数
    def forward(self, Z, Y):
        for k in range(self.K):
            # 1. 梯度模块现在只返回一个值
            grad_Z = self.grad_modules[k](Z, Y)
            eta = self.etas[k]

            Z_half = Z - eta * grad_Z
            # 2. 约束模块现在只处理和返回 Z
            Z = self.constraint_modules[k](Z_half)

        # 3. 最终重建改为 ZZ^T
        X_hat = torch.bmm(Z, Z.transpose(1, 2))  
        return X_hat

def spectral_loss(H_hat, Y, H_target, args, lambda_reg=1.0):
    """
    H_hat: Predicted Hankel matrix, shape [B, M, N]
    Y: One-bit observation matrix, shape [B, M, N]
    H_target: Ground-truth Hankel matrix, shape [B, M, N]
    Returns: total loss (consistency + structure)
    """
    B, M, N = H_hat.shape
    
    # Map Hankel-domain estimate back to 1D signal domain
    X_hat = Dinv_G_adjoint(H_hat)  # [B, M+N-1]
    
    # # Real/imag one-bit consistency terms (Hadamard product)
    # real_part = torch.real(Y) * torch.real(X_hat)
    # imag_part = torch.imag(Y) * torch.imag(X_hat)
    
    # # Stack real and imaginary constraints into one vector
    # combined = torch.stack([real_part, imag_part], dim=-1).view(B, -1)  # [B, 2*(M+N-1)]
    
    # # Penalize only sign-inconsistent entries (negative margin)
    # neg_part = torch.nn.functional.relu(-combined)
    # consistency_loss = (torch.norm(neg_part, p=2, dim=1) ** 2).mean() / 2
    # 修改后：严格对应论文 Eq (9) 和 Eq (18) 的单边范数惩罚
    # w = Re(Y_conj ⊙ X_hat)
    w = (torch.conj(Y) * X_hat).real  # [B, M+N-1] 
    
    # [-w]_+ 等价于 relu(-w)，只惩罚符号不一致的部分
    neg_part = torch.nn.functional.relu(-w)
    
    # 计算平方 l2 范数
    consistency_loss = (torch.norm(neg_part, p=2, dim=-1) ** 2).mean() / 2
    structure_loss = (torch.norm(H_hat - H_target, p='fro', dim=(1, 2)) ** 2).mean() 
    return consistency_loss + lambda_reg * structure_loss

# Training loop
def train_model(model, train_loader, test_loader, F_true_test, A_true_test, args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    F_true_test = F_true_test.to(device)
    A_true_test = A_true_test.to(device)
    
    for epoch in range(args.num_epochs):
        model.train()
        # for batch_idx, (Z, W, Y, H_target, F_true, A_true) in enumerate(train_loader):
        #     Z, W, Y, H_target = Z.to(device), W.to(device), Y.to(device), H_target.to(device)
        #     F_true, A_true = F_true.to(device), A_true.to(device)
        #     optimizer.zero_grad()
            
        #     Forward pass
        #     H_hat = model(Z, W, Y)
        # 用 _ 代替 W 接收，直接忽略它
        for batch_idx, (Z,  Y, H_target, F_true, A_true) in enumerate(train_loader):
            Z, Y, H_target = Z.to(device), Y.to(device), H_target.to(device)
            F_true, A_true = F_true.to(device), A_true.to(device)
            optimizer.zero_grad()
            
            # Forward pass 也只需要传 Z 和 Y 了
            H_hat = model(Z, Y)
            # Compute loss
            loss = spectral_loss(H_hat, Y, H_target, args, lambda_reg=100)
            
            # Backward pass and parameter update
            loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        # 每 10 轮在 test 数据上评估一次频率 / 角度误差和成功率
        if (epoch + 1) % 10 == 0:
            model.eval()
            total_freq_err = []
            per_sample_rmse_list = []
            num_success = 0
            num_total = 0
            success_deg = 3.0  # 判定成功的角度阈值（度）
            with torch.no_grad():
                for batch_idx, (Z_t,Y_t, H_target_t) in enumerate(test_loader):
                    Z_t = Z_t.to(device)
                    # W_t = W_t.to(device)
                    Y_t = Y_t.to(device)
                    H_target_t = H_target_t.to(device)

                    # Forward on test batch
                    # H_hat_t = model(Z_t, W_t, Y_t)
                    H_hat_t = model(Z_t, Y_t)
                    X_hat_t = Dinv_G_adjoint(H_hat_t)
                    A_hat_t, F_hat_t = MatPencilMethod_Batch(X_hat_t, args.rank, args.matrix_row)

                    F_hat_t = F_hat_t.to(device)

                    # 取对应的真值频率
                    bsz = F_hat_t.shape[0]
                    start = batch_idx * bsz
                    end = start + bsz
                    F_true_batch = F_true_test[start:end]

                    finite_mask = torch.isfinite(F_hat_t).all(dim=1)
                    if finite_mask.any():
                        # 有效样本的频率 (Bf, r)
                        f_hat_masked = F_hat_t[finite_mask]
                        f_true_masked = F_true_batch[finite_mask]

                        # 利用周期性，将频率 wrap 到 [0,1) 区间
                        f_hat_wrapped = torch.remainder(f_hat_masked, 1.0)
                        f_true_wrapped = torch.remainder(f_true_masked, 1.0)

                        # 为了消除“频率顺序”带来的旋转/排列误差，按频率升序对每个样本排序
                        f_hat_sorted, _ = torch.sort(f_hat_wrapped, dim=1)
                        f_true_sorted, _ = torch.sort(f_true_wrapped, dim=1)

                        # 频率误差（排序后逐元素对齐，归一化频率差的 L1 均值）
                        freq_error = torch.mean(torch.abs(f_hat_sorted - f_true_sorted))
                        total_freq_err.append(freq_error)

                        # DOA 角度：theta = asin(2*f - 1) * 180/pi
                        x_hat = 2.0 * f_hat_sorted - 1.0
                        x_true = 2.0 * f_true_sorted - 1.0

                        theta_hat = torch.arcsin(x_hat) * (180.0 / np.pi)
                        theta_true = torch.arcsin(x_true) * (180.0 / np.pi)

                        # 按样本计算 DOA 角度 RMSE（在 r 个源上做均方根）
                        # theta_hat/theta_true 形状: [Bf_valid, r]
                        sq_err = (theta_hat - theta_true) ** 2
                        rmse_per_sample = torch.sqrt(sq_err.mean(dim=1))  # [Bf_valid]

                        per_sample_rmse_list.append(rmse_per_sample)

                        # 成功率统计：RMSE < success_deg 判为成功
                        success_mask = rmse_per_sample < success_deg
                        num_success += success_mask.sum().item()
                        num_total += rmse_per_sample.numel()

            if total_freq_err and per_sample_rmse_list and num_total > 0:
                avg_freq_err = torch.stack(total_freq_err).mean().item()
                all_rmse = torch.cat(per_sample_rmse_list)
                rmse_all = all_rmse.mean().item()
                if num_success > 0:
                    rmse_suc = all_rmse[all_rmse < success_deg].mean().item()
                else:
                    rmse_suc = float("nan")
                suc_rate = num_success / num_total
            else:
                avg_freq_err = float("nan")
                rmse_all = float("nan")
                rmse_suc = float("nan")
                suc_rate = 0.0

            print(
                f"[Epoch {epoch+1}] Test Freq MAE={avg_freq_err:.4f}, "
                f"SR={suc_rate:.3f}, RMSE_suc={rmse_suc:.3f} deg, RMSE_all={rmse_all:.3f} deg"
            )
    
    return model

def test_model(model, test_loader, F_true, args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    total_loss = 0
    with torch.no_grad():  # Disable gradients
        # for batch_idx, (Z, W, Y, H_target) in enumerate(test_loader):
        #     Z, W, Y, H_target = Z.to(device), W.to(device), Y.to(device), H_target.to(device)
        for batch_idx, (Z, Y, H_target) in enumerate(test_loader):
            # 【修改点 2】：干掉 W.to(device)
            Z, Y, H_target = Z.to(device), Y.to(device), H_target.to(device)
            # Forward pass
            # H_hat = model(Z, W, Y)
            H_hat = model(Z, Y)
            X_hat = Dinv_G_adjoint(H_hat)
            A_hat, F_hat = MatPencilMethod_Batch(X_hat, args.rank, args.matrix_row)
            # X_true_e = Dinv_G_adjoint(H_target)
            # _, F_true_e = MatPencilMethod_Batch(X_true_e, 3, 32)
            # Compute loss
            loss = spectral_loss(H_hat, Y, H_target, args, lambda_reg=100)
            total_loss += loss.item()
            F_hat = F_hat.to(device)
            batch_start = batch_idx * F_hat.shape[0]
            batch_end = batch_start + F_hat.shape[0]
            F_true_batch = F_true[batch_start:batch_end].to(device)
            finite_mask = torch.isfinite(F_hat).all(dim=1)
            if finite_mask.any():
                freq_error = torch.mean(torch.abs(F_hat[finite_mask] - F_true_batch[finite_mask]))
            else:
                freq_error = torch.tensor(float("nan"), device=device)
            print(f"Batch {batch_idx}, Freq Error: {freq_error.item():.4f}")
            # Optional: print some outputs
            if batch_idx % 2 == 0:
                print(f"Test Batch {batch_idx}, Loss: {loss.item():.4f}")
                print(f"Output shape: {H_hat.shape}")
                
    avg_loss = total_loss / len(test_loader)
    print(f"\nAverage Test Loss: {avg_loss:.4f}")
    return avg_loss

# Main script entry
if __name__ == "__main__":
    # Build argument parser
    parser = argparse.ArgumentParser(description="One-bit Unrolled Network Preparation")
        # Signal parameters group
    signal_group = parser.add_argument_group('Signal Parameters')
    signal_group.add_argument('--signal_dim', type=int, default=63,
                            help='Must to be odd, Dimension of the signal (default: 63)')
    signal_group.add_argument('--rank', type=int, default=3,
                            help='Rank of the matrix (default: 3)')
    
    # Training parameters group
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--num_layers', type=int, default=5,
                           help='Number of unrolled layers (default: 5)')
    train_group.add_argument('--num_samples', type=int, default=1000,
                           help='Total number of samples (default: 1000)')
    train_group.add_argument('--num_epochs', type=int, default=100,
                           help='Number of training epochs (default: 100)')
    train_group.add_argument('--lr', type=float, default=0.001,
                           help='Learning rate (default: 0.001)')
    train_group.add_argument('--batch_size', type=int, default=32,
                           help='Batch size (default: 32)')
    
    # System parameters group
    system_group = parser.add_argument_group('System Parameters')
    system_group.add_argument('--seed', type=int, default=42,
                            help='Random seed (default: 42)')
    system_group.add_argument('--device', type=str, default='cuda:0',
                            choices=['cpu', 'cuda:0'],
                            help='Training device (default: cuda)')
    
    args = parser.parse_args()
    
    # Calculate dependent parameters
    args.matrix_row = int(np.floor((args.signal_dim+1)/2))
                                     
    signal_dim = args.signal_dim  # Signal dimension
    num_layers = args.num_layers    # Number of unrolled layers
    num_samples =args.num_samples  # Number of training samples
    matrix_row = args.matrix_row
    rank = args.rank
    num_epochs = args.num_epochs
    lr=args.lr
    # Build model
    model = SpectralDeepUnfolding(M=matrix_row, N=signal_dim+1-matrix_row, K=num_layers,rank=rank)
    
    # xs, x_true, freqs, amps = generate_signal_1D(nd=signal_dim, r=rank, separation=True, damp=False,snr=20, random_state=42)
    # _, F_true_e = MatPencilMethod(x_true, rank, matrix_row)
    # freqs = np.sort(freqs)  # Sort frequencies
    # # xs_bit = np.sign(xs.real) + 1j * np.sign(xs.imag)
    # Generate training data
    Dict_train = generate_signal_batch(batch_size=num_samples, nd=signal_dim, r=rank,
                                       separation=True, damp=False, snr=20, random_seed=None)
    X_noisy_train = torch.from_numpy(Dict_train['observed'])
    X_true_train = torch.from_numpy(Dict_train['clean'])
    F_true_train = torch.from_numpy(Dict_train['freqs'])
    A_true_train = torch.from_numpy(Dict_train['amps'])
    H_target_train = hankel_lifting(X_true_train, M=matrix_row, N=signal_dim + 1 - matrix_row)
    Y_input_train = torch.sign(X_noisy_train.real) + 1j * torch.sign(X_noisy_train.imag)
    # U_input_train, V_input_train = spectral_init(Y_input_train, L=matrix_row, r=rank)
    U_input_train, _ = spectral_init(Y_input_train, L=matrix_row, r=rank)
    
    # Build training DataLoader
    # dataset_train = TensorDataset(U_input_train, V_input_train,
    #                               Y_input_train, H_target_train,
    #                               F_true_train, A_true_train)
    dataset_train = TensorDataset(U_input_train, 
                                  Y_input_train, H_target_train,
                                  F_true_train, A_true_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    # Generate test data
    Dict_test= generate_signal_batch(batch_size=args.batch_size, nd=signal_dim, r=rank,
                                     separation='True', damp=False, snr='noiseless', random_seed=42)  #改动
    X_noisy_test = torch.from_numpy(Dict_test['observed'])
    X_true_test = torch.from_numpy(Dict_test['clean'])
    F_true_test = torch.from_numpy(Dict_test['freqs'])
    A_true_test = torch.from_numpy(Dict_test['amps'])
    H_target_test = hankel_lifting(X_true_test, M=matrix_row, N=signal_dim + 1 - matrix_row)
    Y_input_test = torch.sign(X_noisy_test.real) + 1j * torch.sign(X_noisy_test.imag)
    # U_input_test, V_input_test = spectral_init(Y_input_test, L=matrix_row, r=rank)
    U_input_test,_= spectral_init(Y_input_test, L=matrix_row, r=rank)

    # X_true_e = Dinv_G_adjoint(H_target_test)
    # _, F_true_e = MatPencilMethod_Batch(X_true_e, rank, matrix_row)

    # Build test DataLoader
    # dataset_test = TensorDataset(U_input_test, V_input_test, Y_input_test, H_target_test)
    dataset_test = TensorDataset(U_input_test,Y_input_test, H_target_test)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # Train model with periodic evaluation on test data
    train_model(model, train_loader, test_loader, F_true_test, A_true_test, args)

    # Final evaluation on test set
    test_model(model, test_loader, F_true_test, args)
