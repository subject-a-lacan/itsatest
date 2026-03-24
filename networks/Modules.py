import torch
import torch.nn.functional as F
from torch import nn
class GradientModule(nn.Module):
    def __init__(self, M, N, lambda_reg=1.0):
        super().__init__()
        self.M = M
        self.N = N
        self.lambda_reg = lambda_reg

        # 预计算 sqrt(w_a) 和其倒数
        self.weights = self.get_diag_weights(M, N)  # [M+N-1]
        self.sqrt_w = torch.sqrt(self.weights)
        self.sqrt_w_inv = 1.0 / self.sqrt_w

    def forward(self, U, V, y):
        """
        U: [B, M, r]
        V: [B, N, r]
        y: [B, M+N-1], one-bit observation vector (±1)
        """
        # B = U.shape[0]
        X = torch.bmm(U, torch.conj(V).transpose(1, 2))  # [B, M, N]

        # Step 1: G^* X = D^{-1} H^* X
        Gadj_X = self.G_adjoint(X, self.sqrt_w_inv.to(X.device))  # [B, M+N-1]

        # Step 2: w = Re( y.conj ⊙ D^{-1} G^* X )
        w = (y.conj() * self.sqrt_w_inv.to(X.device)[None, :] * Gadj_X).real  # [B, M+N-1]

        # Step 3: g(w) 
        gw = self.squareplus_function(w,b=0.01)  # [B, M+N-1]

        # Step 4: g'(w) 
        gw_der = self.squareplus_derivative(w,b=0.01) # [B, M+N-1]

        # Step 4: g(w) ⊙ g'(w) ⊙ y
        gw2_y = gw * gw_der * y  # [B, M+N-1]

        # Step 5: 𝓖(gw2_y) = G(D^{-1} * gw2_y)
        G_gw2_y = self.hankel_lifting(gw2_y * self.sqrt_w_inv.to(X.device) * self.sqrt_w_inv.to(X.device))  # [B, M, N]
        GGadj_X = self.hankel_lifting(Gadj_X* self.sqrt_w_inv.to(X.device))

        # grad_U
        U_grad = torch.bmm(G_gw2_y, V) + self.lambda_reg * (torch.bmm(X,V) - torch.bmm(GGadj_X,V))

        # grad_V
        V_grad = torch.bmm(torch.conj(G_gw2_y).transpose(1, 2), U) + self.lambda_reg * (torch.bmm(torch.conj(X).transpose(1, 2),U) - torch.bmm(torch.conj(GGadj_X).transpose(1, 2), U))

        return U_grad, V_grad

    def hankel_lifting(self, x):
        # x: [B, M+N-1] → H(x): [B, M, N]
        B = x.shape[0]
        M, N = self.M, self.N
        out = torch.zeros(B, M, N, device=x.device, dtype=x.dtype)
        for i in range(M):
            for j in range(N):
                out[:, i, j] = x[:, i + j]
        return out
    
    def hankel_adjoint(self, H):
        """
        H: [B, M, N], Hankel matrix
        Return: [B, M+N-1] vector, each entry is the sum over a skew-diagonal
        """
        B, M, N = H.shape
        n = M + N - 1
        out = torch.zeros(B, n, device=H.device, dtype=H.dtype)
        for i in range(M):
            for j in range(N):
                out[:, i + j] += H[:, i, j]
        return out

    def get_diag_weights(self, M, N):
        """
        Returns a tensor of length M+N-1 indicating number of elements in each skew-diagonal
        """
        weights = [min(i + 1, M, N, M + N - 1 - i) for i in range(M + N - 1)]
        return torch.tensor(weights, dtype=torch.float32)
    
    def G_adjoint(self, X, D_sqrt_inv):
        """
        X: [B, M, N], input matrix
        D_sqrt_inv: [M+N-1], 1 / sqrt(w_a)
        Returns: [B, M+N-1] vector
        """
        H_adj = self.hankel_adjoint(X)  # [B, M+N-1]
        return H_adj * D_sqrt_inv[None, :]

    def softplus_function(self, x):
        return -torch.log(1 + torch.exp(-x))

    def softplus_derivative(self, x):
        return torch.sigmoid(x)
    
    def squareplus_function(self, x, b=0.01):
        return (x - torch.sqrt(x**2 + b)) / 2

    def squareplus_derivative(self, x, b=0.01):
        return (1 - x / torch.sqrt(x**2 + b)) / 2
    

class ConstraintModule(nn.Module):
    def __init__(self, M, N, rank, hidden_dim=128):
        super().__init__()
        self.M, self.N, self.rank = M, N, rank
        self.resnet_Z = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*M*rank, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*M*rank)
        )
        self.resnet_W = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*N*rank, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*N*rank)
        )

    def forward(self, Z, W):
        Zreal = Z.real.to(torch.float32)
        Zimag = Z.imag.to(torch.float32)
        Zcombine = torch.cat([Zreal, Zimag], dim=1) # [B, M * 2r]

        res_Z = self.resnet_Z(Zcombine) 

        Zreal_res,Zimag_res = res_Z.chunk(2, dim=-1) # 分离实部和虚部
        Zdelta = torch.complex(Zreal_res, Zimag_res)
        Z_corr = Z + Zdelta.view(-1, self.M, self.rank)

        Wreal = W.real.to(torch.float32)
        Wimag = W.imag.to(torch.float32)
        Wcombine = torch.cat([Wreal, Wimag], dim=-1)
        res_W = self.resnet_W(Wcombine)
        Wreal_res,Wimag_res = res_W.chunk(2, dim=-1)
        Wdelta = torch.complex(Wreal_res, Wimag_res)
        W_corr = W + Wdelta.view(-1, self.N, self.rank)

        return Z_corr, W_corr


