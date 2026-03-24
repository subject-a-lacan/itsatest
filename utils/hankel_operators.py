import torch
from typing import Callable

class HankelOperator:
    """Hankel矩阵升维算子与伴随算子"""
    def __init__(self, n: int, n1: int):
        self.n = n
        self.n1 = n1
        self.n2 = n - n1 + 1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """向量→Hankel矩阵"""
        batch_size = x.shape[0]
        H = torch.zeros(batch_size, self.n1, self.n2, dtype=x.dtype, device=x.device)
        for i in range(self.n1):
            for j in range(self.n2):
                H[:, i, j] = x[:, i + j]
        return H
    
    def adjoint(self, H: torch.Tensor) -> torch.Tensor:
        """Hankel矩阵→向量"""
        batch_size = H.shape[0]
        x = torch.zeros(batch_size, self.n, dtype=H.dtype, device=H.device)
        weights = torch.zeros(batch_size, self.n, dtype=H.dtype, device=H.device)
        
        for i in range(self.n1):
            for j in range(self.n2):
                x[:, i + j] += H[:, i, j]
                weights[:, i + j] += 1
                
        return x / weights.clamp(min=1)