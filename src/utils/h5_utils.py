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

def h5_dataset_to_parquet(
    h5_path: Path,
    dataset_key: str,
    out_parquet_dir: Path,
    chunk_rows: int = 250_000,
    partition_cols: list[str] | None = None,
) -> None:
    """
    Seçili dataset'i satırlar hâlinde chunk-chunk okuyup Parquet dosyalarına yazar.
    """
    out_parquet_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        dset = f[dataset_key]
        n_rows = dset.shape[0]

        for start in tqdm(range(0, n_rows, chunk_rows), desc=f"{h5_path.name}:{dataset_key}"):
            stop = min(start + chunk_rows, n_rows)
            chunk_arr = dset[start:stop]
            df = pd.DataFrame(chunk_arr)

            file_name = f"{h5_path.stem}_{dataset_key}_{start}_{stop-1}.parquet"
            df.to_parquet(out_parquet_dir / file_name, index=False, partition_cols=partition_cols)
