import pandas as pd
import numpy as np
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
    def __init__(self, data_array: np.ndarray, x: int, y: int):
        """
        data_array: numpy.ndarray, şekil (timestep, sensor_sayısı)
        x: geçmiş adım sayısı, y: tahmin adım sayısı
        """
        self.x = x
        self.y = y
        self.data = data_array
        # oluşturulabilecek örnek sayısı
        self.num_samples = data_array.shape[0] - x - y + 1

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
):
    """
    Veri setini okuyup eğitim ve doğrulama DataLoader'larını döndürür.
    """
    data_array = load_data(parquet_path)
    dataset = TrafficDataset(data_array, x, y)
    total_samples = len(dataset)
    train_len = int(total_samples * train_ratio)
    val_len = total_samples - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

# örnek kullanım\ if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    print(f"Eğitim loader örnek sayısı: {len(train_loader)}")
    print(f"Doğrulama loader örnek sayısı: {len(val_loader)}")
