import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import PROJECT_ROOT
import re

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
        device: str = None,
        pca_model=None
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
            "train_mse": [],
            "train_mae": [],
            "train_r2": [],
            "val_loss": [],
            "val_mse": [],
            "val_mae": [],
            "val_r2": []
        }
        self.final_metrics = {}
        self.model_type = model_type
        # PCA model for inverse transformation (optional)
        self.pca_model = pca_model


    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        # Convert to numpy
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        # If PCA model provided, inverse-transform to original feature space
        if getattr(self, 'pca_model', None) is not None:
            # y_true_np shape: (batch, y_steps, n_components)
            batch, y_steps, comps = y_true_np.shape
            # flatten time and batch dims
            y_true_flat = y_true_np.reshape(-1, comps)
            y_pred_flat = y_pred_np.reshape(-1, comps)
            # inverse transform
            y_true_orig = self.pca_model.inverse_transform(y_true_flat)
            y_pred_orig = self.pca_model.inverse_transform(y_pred_flat)
            # flatten for metrics
            y_t = y_true_orig.flatten()
            y_p = y_pred_orig.flatten()
        else:
            y_t = y_true_np.flatten()
            y_p = y_pred_np.flatten()
        mse = mean_squared_error(y_t, y_p)
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        return {'mse': mse, 'mae': mae, 'r2': r2}
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
            # Collect predictions and truths for train metrics
            train_true = []
            train_pred = []
            for X_batch, y_batch in train_loader:
                Xb = X_batch.to(self.device).float()
                yb = y_batch.to(self.device).float()
                self.optimizer.zero_grad()
                yp = self.model(Xb)
                loss = self.criterion(yp, yb)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                # append for train metrics
                train_true.append(yb.detach().cpu())
                train_pred.append(yp.detach().cpu())
            train_loss = np.mean(train_losses)
            self.history['train_loss'].append(train_loss)
            # compute train metrics on original feature space
            train_metrics = self.compute_metrics(torch.cat(train_true), torch.cat(train_pred))
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['train_r2'].append(train_metrics['r2'])

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
        # store final predictions for plotting
        self.val_true_all = torch.cat(all_true).detach().cpu().numpy().flatten()
        self.val_pred_all = torch.cat(all_pred).detach().cpu().numpy().flatten()
        # increment run count after training


    def inference(self, X, to_numpy: bool = True):
        """
        Modeli evaluation modunda çalıştırarak tahmin üretir.

        Parametreler
        ------------
        X : torch.Tensor | torch.utils.data.DataLoader | np.ndarray | list
            Model girdisi. Tensor ise şekli (batch, x, sensors) olmalıdır.
            DataLoader verilirse her öğe (X_batch, y_batch) veya sadece X_batch
            olabilir; yalnızca X_batch kullanılır.
        to_numpy : bool
            True ise çıktıyı numpy array olarak döndürür.

        Dönüş
        -----
        torch.Tensor | np.ndarray
            (batch, y, sensors) boyutlu tahminler.
        """
        self.model.eval()
        preds = []
        with torch.no_grad():
            if isinstance(X, torch.utils.data.DataLoader):
                for batch in X:
                    Xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                    Xb = torch.as_tensor(Xb, dtype=torch.float32, device=self.device)
                    yp = self.model(Xb)
                    preds.append(yp.cpu())
                preds = torch.cat(preds, dim=0)
            else:
                Xb = torch.as_tensor(X, dtype=torch.float32, device=self.device)
                preds = self.model(Xb.to(self.device)).cpu()
        return preds.numpy() if to_numpy else preds

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
        """Plot & save training curves and prediction heatmap with auto-incrementing filenames."""
        graphs_dir = PROJECT_ROOT / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        # determine next train id by scanning existing subdirectories graphs/trainX
        subdirs = [p.name for p in graphs_dir.iterdir() if p.is_dir() and re.match(r'train\d+$', p.name)]
        ids = [int(re.match(r'train(\d+)', name).group(1)) for name in subdirs]
        train_id = max(ids) + 1 if ids else 0

        # create run-specific directory
        run_dir = graphs_dir / f"train{train_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        suffix = self.model_type + ("_pca" if self.pca_model is not None else "")
        fname = f"{suffix}.png"

        epochs = list(range(1, len(self.history['train_loss']) + 1))
        plt.figure(figsize=(12, 5))
        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='train')
        if self.history['val_loss']:
            plt.plot(epochs, self.history['val_loss'], label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()

        # MSE subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_mse'], label='train')
        if self.history['val_mse']:
            plt.plot(epochs, self.history['val_mse'], label='val')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE per Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig(run_dir / fname)
        plt.show()
        plt.close()

        # Heatmap of true vs predicted
        if hasattr(self, 'val_true_all') and hasattr(self, 'val_pred_all') and len(self.val_true_all) > 0:
            cm_fname = "cm_" + fname
            cmatrix, _, _ = np.histogram2d(self.val_true_all, self.val_pred_all, bins=50)
            plt.figure(figsize=(6, 6))
            sns.heatmap(cmatrix, cmap='viridis')
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.title('True vs Predicted')
            plt.tight_layout()
            plt.savefig(run_dir / cm_fname)
            plt.close()

        # Print metrics
        print("== Final Metrics ==")
        for k, v in self.final_metrics.items():
            print(f"{k}: {v}")
