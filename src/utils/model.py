import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# LSTM tabanlı model
class LSTMModel(nn.Module):
    def __init__(self, sensors: int, x: int, y: int, hidden_dim: int = 64, num_layers: int = 2):
        super(LSTMModel, self).__init__()
        self.sensors = sensors
        self.x = x
        self.y = y
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=sensors, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, sensors * y)

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        # x_input: (batch, x, sensors)
        h0 = torch.zeros(self.num_layers, x_input.size(0), self.hidden_dim).to(x_input.device)
        c0 = torch.zeros(self.num_layers, x_input.size(0), self.hidden_dim).to(x_input.device)
        out, _ = self.lstm(x_input, (h0, c0))
        out = out[:, -1, :]                      # son adım
        out = self.fc(out)                      # (batch, sensors * y)
        out = out.view(-1, self.y, self.sensors)  # (batch, y, sensors)
        return out

# SCGNN tabanlı model: sensörler arası ilişkileri grafik konvolüsyonu ile işler
try:
    from torch_geometric.nn import GCNConv
except ImportError:  # torch-geometric kurulu değilse SCGNNModel kullanılamaz
    GCNConv = None

class SCGNNModel(nn.Module):
    def __init__(self, sensors: int, x: int, y: int, hidden_dim: int = 64, edge_index=None):
        super(SCGNNModel, self).__init__()
        if GCNConv is None:
            raise ImportError("torch-geometric kurulu değil, SCGNNModel kullanılamaz.")
        if edge_index is None:
            raise ValueError("edge_index (graf adjacency) parametresi gerekli.")
        self.sensors = sensors
        self.x = x
        self.y = y
        self.gc1 = GCNConv(x, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, y)
        self.edge_index = edge_index

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        # x_input: (batch, x, sensors)
        batch_size = x_input.size(0)
        # (batch, sensors, x) -> (batch*sensors, x)
        h = x_input.permute(0, 2, 1).reshape(-1, self.x)
        h = self.gc1(h, self.edge_index)
        h = torch.relu(h)
        h = self.gc2(h, self.edge_index)        # (batch*sensors, y)
        h = h.view(batch_size, self.sensors, self.y)
        h = h.permute(0, 2, 1)                  # (batch, y, sensors)
        return h

# Model sarmalayıcı sınıfı: eğitim, doğrulama ve metrik hesaplama
class TrafficPredictor:
    def __init__(
        self,
        model_type: str,
        sensors: int,
        x: int,
        y: int,
        edge_index=None,
        device: str = None
    ):
        if model_type not in ["lstm", "scgnn"]:
            raise ValueError("model_type 'lstm' veya 'scgnn' olmalı.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sensors = sensors
        self.x = x
        self.y = y
        if model_type == "lstm":
            self.model = LSTMModel(sensors, x, y).to(self.device)
        else:
            self.model = SCGNNModel(sensors, x, y, edge_index=edge_index).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mse": [],
            "val_mae": [],
            "val_r2": []
        }
        self.final_metrics = {}

    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        y_t = y_true.detach().cpu().numpy().reshape(-1)
        y_p = y_pred.detach().cpu().numpy().reshape(-1)
        mse = mean_squared_error(y_t, y_p)
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }

    def train(self, train_loader, val_loader, epochs: int = 10, save_path: str | None = None, save_optimizer: bool = False):
        best_val_loss = float('inf') if save_path else None
        best_epoch = -1
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                Xb = X_batch.to(self.device).float()
                yb = y_batch.to(self.device).float()
                self.optimizer.zero_grad()
                yp = self.model(Xb)
                loss = self.criterion(yp, yb)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            self.history['train_loss'].append(train_loss)

            # Validation aşaması
            self.model.eval()
            val_losses = []
            all_true = []
            all_pred = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    Xb = X_batch.to(self.device).float()
                    yb = y_batch.to(self.device).float()
                    yp = self.model(Xb)
                    loss = self.criterion(yp, yb)
                    val_losses.append(loss.item())
                    all_true.append(yb)
                    all_pred.append(yp)
            val_loss = np.mean(val_losses)
            # checkpoint
            if save_path and val_loss < best_val_loss:
                checkpoint = self.model.state_dict()
                if save_optimizer:
                    checkpoint = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }
                torch.save(checkpoint, save_path)
                best_val_loss = val_loss
                best_epoch = epoch
                print(f"  >> Model saved to {save_path} (val_loss improved)")
            self.history['val_loss'].append(val_loss)

            metrics = self.compute_metrics(torch.cat(all_true), torch.cat(all_pred))
            self.history['val_mse'].append(metrics['mse'])
            self.history['val_mae'].append(metrics['mae'])
            self.history['val_r2'].append(metrics['r2'])

            print(
                f"Epoch {epoch}/{epochs}  "
                f"Train Loss: {train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}  "
                f"Val MSE: {metrics['mse']:.4f}  "
                f"Val R2: {metrics['r2']:.4f}"
            )

        # Son metrikler
        self.final_metrics = self.compute_metrics(torch.cat(all_true), torch.cat(all_pred))
        if best_epoch != -1:
            print(f"Training finished. Best epoch: {best_epoch} with val_loss={best_val_loss:.4f}")

    def save_model(self, path: str):
        """Save current model parameters"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str, load_optimizer: bool = False):
        """Load model (and optionally optimizer) parameters from file"""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            if load_optimizer and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

    def show(self):
        print("== Son Performans Metrikleri ==")
        for name, value in self.final_metrics.items():
            print(f"{name}: {value}")
