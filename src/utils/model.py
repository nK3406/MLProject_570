import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
except ImportError:
    raise ImportError("SCGNNModel için torch-geometric'in kurulması gerekiyor.")

class SCGNNModel(nn.Module):
    def __init__(self, sensors: int, x: int, y: int, hidden_dim: int = 64, edge_index=None):
        super(SCGNNModel, self).__init__()
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
            "train_acc": [],
            "val_acc": []
        }
        self.final_metrics = {}

    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        y_t = y_true.detach().cpu().numpy().reshape(-1)
        y_p = y_pred.detach().cpu().numpy().reshape(-1)
        # Regresyon metriği
        mse = mean_squared_error(y_t, y_p)
        # Sınıflandırma metrikleri (tamsayıya yuvarlayarak)
        y_t_int = np.round(y_t).astype(int)
        y_p_int = np.round(y_p).astype(int)
        acc = accuracy_score(y_t_int, y_p_int)
        prec = precision_score(y_t_int, y_p_int, average='macro', zero_division=0)
        rec = recall_score(y_t_int, y_p_int, average='macro', zero_division=0)
        f1 = f1_score(y_t_int, y_p_int, average='macro', zero_division=0)
        cm = confusion_matrix(y_t_int, y_p_int)
        return {
            'mse': mse,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm
        }

    def train(self, train_loader, val_loader, epochs: int = 10):
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
            self.history['val_loss'].append(val_loss)

            metrics = self.compute_metrics(torch.cat(all_true), torch.cat(all_pred))
            self.history['train_acc'].append(metrics['accuracy'])
            self.history['val_acc'].append(metrics['accuracy'])

            print(
                f"Epoch {epoch}/{epochs}  "
                f"Train Loss: {train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}  "
                f"Val Acc: {metrics['accuracy']:.4f}"
            )

        # Son metrikler
        self.final_metrics = self.compute_metrics(torch.cat(all_true), torch.cat(all_pred))

    def show(self):
        print("== Son Performans Metrikleri ==")
        for name, value in self.final_metrics.items():
            print(f"{name}: {value}")
