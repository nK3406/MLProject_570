import pandas as pd
import numpy as np
import random

# ---- Deterministic settings ----
random.seed(0)
np.random.seed(0)
try:
    import torch
    torch.manual_seed(0)
except ImportError:
    pass
# --------------------------------
from sklearn.decomposition import PCA

torch_import = False
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    torch_import = True
except ImportError:
    raise ImportError("PyTorch kurulu değil: torch ve torch.utils.data gereklidir.")

# --------------------
# Hiperparametreler
# --------------------
BATCH_SIZE = 64         # minibatch boyutu
TRAIN_RATIO = 0.8       # eğitim-verification ayırma oranı
X = 12                  # geçmiş zaman dilimi sayısı (örneğin 12*5 dakikalık adım)
Y = 3                   # tahmin edilecek zaman dilimi sayısı
PARQUET_PATH = "data/traffic.parquet"  # veri yolu

class TrafficDataset(Dataset):
    def __init__(self, data_array: np.ndarray, x: int, y: int, pca_components: int | None = None, pca_model: PCA | None = None):
        """
        data_array: numpy.ndarray, shape (timesteps, num_sensors)
        x: geçmiş adım sayısı (input sequence length)
        y: tahmin adım sayısı (output sequence length)
        pca_components: PCA bileşen sayısı. None ise PCA uygulanmaz.
        pca_model: Önceden eğitilmiş sklearn PCA nesnesi. Verilirse yeniden fit edilmez.
        """
        self.x = x
        self.y = y

        # Optional PCA transformation
        if pca_components is not None:
            if pca_model is None:
                self.pca = PCA(n_components=pca_components, svd_solver='full', random_state=0)
                self.data = self.pca.fit_transform(data_array)
            else:
                self.pca = pca_model
                self.data = self.pca.transform(data_array)
        else:
            self.pca = None
            self.data = data_array

        # Number of sensors/features after optional PCA
        self.sensors = self.data.shape[1]
        # oluşturulabilecek örnek sayısı
        self.num_samples = self.data.shape[0] - x - y + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # girdi ve çıktı dizilerini al
        seq_x = self.data[idx : idx + self.x]               # (x, sensör)
        seq_y = self.data[idx + self.x : idx + self.x + self.y]  # (y, sensör)
        # tensöre çevir
        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
        )


    def inverse_pca_transform(self, reduced_data):
        """
        Restores the original feature dimensionality of data reduced by PCA.
        If data is a torch.Tensor, returns a torch.Tensor; otherwise returns a numpy array.
        """
        if self.pca is None:
            return reduced_data
        is_tensor = isinstance(reduced_data, torch.Tensor)
        if is_tensor:
            data_np = reduced_data.detach().cpu().numpy()
        else:
            data_np = reduced_data
        # Flatten last dimension and reshape to 2D array for inverse transform
        orig_shape = data_np.shape
        flat = data_np.reshape(-1, orig_shape[-1])
        # Apply inverse PCA
        inv_flat = self.pca.inverse_transform(flat)
        # Restore original feature count
        feature_count = self.pca.components_.shape[1]
        restored = inv_flat.reshape(*orig_shape[:-1], feature_count)
        # Convert back to tensor if needed
        if is_tensor:
            return torch.tensor(restored, dtype=torch.float32)
        else:
            return restored


def load_data(parquet_path: str = PARQUET_PATH) -> np.ndarray:
    """
    Parquet formatındaki veri setini okuyup zaman sıralı numpy dizisine çevirir.
    """
    df = pd.read_parquet(parquet_path)
    # datetime olarak sırala ve index olarak ata
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    return df.values  # (timestep, sensör_sayısı)


def get_dataloaders(
    parquet_path: str = PARQUET_PATH,
    batch_size: int = BATCH_SIZE,
    train_ratio: float = TRAIN_RATIO,
    x: int = X,
    y: int = Y,
    pca_components: int | None = None,
):
    """
    Veri setini okuyup eğitim ve doğrulama DataLoader'larını döndürür.
    """
    data_array = load_data(parquet_path)
    dataset = TrafficDataset(data_array, x, y, pca_components=pca_components)
    total_samples = len(dataset)
    train_len = int(total_samples * train_ratio)
    val_len = total_samples - train_len
    # Deterministic split and loaders
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# örnek kullanım\ if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    print(f"Eğitim loader örnek sayısı: {len(train_loader)}")
    print(f"Doğrulama loader örnek sayısı: {len(val_loader)}")
