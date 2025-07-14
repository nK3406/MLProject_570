import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """
    Trafik hız/akış tahmini için sadeleştirilmiş tek parça LSTM modeli.
    Girdi şekli  : (batch, seq_len, node, feature)
    Çıktı şekli  : (batch, horizon, node, 1)
    """
    def __init__(
        self,
        node_num: int,
        input_dim: int,
        horizon: int = 12,
        init_dim: int = 32,
        hid_dim: int = 64,
        end_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Hiperparametreler
        self.node_num   = node_num
        self.input_dim  = input_dim
        self.horizon    = horizon

        # Katmanlar
        self.start_conv = nn.Conv2d(
            in_channels=input_dim,
            out_channels=init_dim,
            kernel_size=(1, 1)
        )
        self.lstm = nn.LSTM(
            input_size=init_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.end_linear1 = nn.Linear(hid_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, F)  -> (batch, seq_len, node, feature)
        """
        # (B, T, N, F) -> (B, F, N, T)
        x = x.transpose(1, 3)
        B, F, N, T = x.shape

        # (B, F, N, T) -> (B*N, F, 1, T)
        x = x.transpose(1, 2).reshape(B * N, F, 1, T)

        # Başlangıç 1x1 konv
        x = self.start_conv(x).squeeze(-2).transpose(1, 2)   # (B*N, T, init_dim)

        # LSTM
        out, _ = self.lstm(x)                                 # (B*N, T, hid_dim)
        x = out[:, -1, :]                                     # (B*N, hid_dim)

        # Çıkış katmanları
        x = F.relu(self.end_linear1(x))                       # (B*N, end_dim)
        x = self.end_linear2(x)                               # (B*N, horizon)

        # (B*N, horizon) -> (B, horizon, N, 1)
        x = x.view(B, N, self.horizon, 1).transpose(1, 2)
        return x

    def param_num(self) -> int:
        """Toplam parametre sayısı"""
        return sum(p.numel() for p in self.parameters())

# Küçük kontrol
if __name__ == "__main__":
    model = LSTMModel(node_num=10, input_dim=2)
    dummy = torch.randn(4, 12, 10, 2)  # (batch=4)
    out   = model(dummy)
    print(out.shape)  # -> torch.Size([4, 12, 10, 1])
    print("Parametre:", model.param_num())