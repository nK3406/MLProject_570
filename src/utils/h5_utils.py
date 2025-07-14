"""
LargeST HDF5 dosyalarını inceleme ve Parquet’e dönüştürme yardımcı işlevleri.
"""
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def list_h5_structure(h5_path: Path, max_depth: int = 4) -> None:
    """
    HDF5 dosyasındaki grup/dataset yapısını hiyerarşik olarak yazdırır.
    Dataset düğümlerinde .items() çağrısı yapılmaz.
    """

    def _traverse(name, obj, depth):
        indent = "  " * depth
        if isinstance(obj, h5py.Dataset):
            # Dataset: sadece ad, shape ve dtype göster
            print(f"{indent}📄 {name}  {obj.shape}  {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            # Group: adı yaz, sonra içindekileri gez
            print(f"{indent}📁 {name}/")
            if depth < max_depth:
                for key, val in obj.items():
                    child_name = f"{name}/{key}" if name else key
                    _traverse(child_name, val, depth + 1)
        else:
            # Diğer hdf5 objeleri (nadir)
            print(f"{indent}❔ {name}")

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        _traverse("", f, 0)

def h5_to_parquet(
    h5_path: Path,
    parquet_path: Path,
    dataset_key: str,
    timestamp_key: str = "t/axis1",
    columns: list = None,
    chunk_rows: int = None,
) -> None:
    """
    Belirtilen bir HDF5 dataset'ini pandas DataFrame'e çevirip, zaman damgasını (timestamp) ilk sütuna ekleyerek Parquet olarak kaydeder.

    Args:
        h5_path (Path): HDF5 dosya yolu
        parquet_path (Path): Parquet dosya yolu (veya klasörü)
        dataset_key (str): HDF5 içindeki dataset anahtarı (örn: 'mygroup/mydataset')
        timestamp_key (str): Zaman damgası dataset anahtarı (örn: 't/axis1'). DataFrame'in ilk sütunu olarak eklenir.
        columns (list, optional): DataFrame sütun isimleri. None ise varsayılan isimler kullanılır.
        chunk_rows (int, optional): Satır bazında parça parça okuma (büyük dosyalar için). None ise tamamı okunur.
    """
    h5_path = Path(h5_path)
    parquet_path = Path(parquet_path)
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_key]
        timestamps = f[timestamp_key][:]
        n_rows = ds.shape[0]
        n_cols = ds.shape[1] if len(ds.shape) > 1 else 1
        if chunk_rows is None:
            data = ds[:]
            df = pd.DataFrame(data, columns=columns)
            df.index = pd.Index(timestamps, name="timestamp")
            df.to_parquet(parquet_path, index=True)
        else:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            for start in tqdm(range(0, n_rows, chunk_rows), desc=f"{h5_path.name} chunks"):
                stop = min(start + chunk_rows, n_rows)
                data = ds[start:stop]
                df = pd.DataFrame(data, columns=columns)
                df.index = pd.Index(timestamps[start:stop], name="timestamp")
                out_file = parquet_path.parent / f"{h5_path.stem}_{start}_{stop-1}.parquet"
                df.to_parquet(out_file, index=True)


def h5_block_to_parquet_wide(
    h5_path: Path,
    out_dir: Path,
    block_key: str = "t/block0_values",
    axis0_key: str = "t/axis0",
    axis1_key: str = "t/axis1",
    chunk_rows: int = 24 * 60,        # 1 gün = 1440 satır
    add_timestamp: bool = True,
) -> None:
    """
    LargeST HDF5 dosyasındaki 'block0_values' (zaman × sensör) matrisini
    geniş formatta (sensor_id'ler sütun) Parquet dosyalarına yazar.

    - Her 'chunk_rows' kadar satır bir Parquet dosyasına gider.
    - Bellek verimliliği için blok blok okur.
    """
    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        values_ds  = f[block_key]
        sensor_ids = f[axis0_key][:].astype(str)      # (8600,)
        timestamps = f[axis1_key][:]                  # (N,)

        n_rows = values_ds.shape[0]

        for start in tqdm(range(0, n_rows, chunk_rows), desc=h5_path.name):
            stop = min(start + chunk_rows, n_rows)

            # --- Chunk’ı oku ---
            val_chunk = values_ds[start:stop, :]        # ndarray  (chunk_rows × 8600)

            # --- DataFrame oluştur ---
            df = pd.DataFrame(val_chunk, columns=sensor_ids)

            if add_timestamp:
                df.insert(0, "timestamp", timestamps[start:stop])

            # --- Parquet kaydet ---
            out_file = out_dir / f"{h5_path.stem}_{start}_{stop-1}.parquet"
            df.to_parquet(out_file, index=False)