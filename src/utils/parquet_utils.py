"""Parquet dosyalarını önizlemek için yardımcı işlevler."""
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds

def preview_parquet(parquet_path: Path, n_rows: int = 5) -> pd.DataFrame:
    """Belirtilen Parquet dosyasından ilk ``n_rows`` satırı döndürür.

    Parameters
    ----------
    parquet_path : Path
        Gözlencek Parquet dosyasının yolu.
    n_rows : int, optional
        Okunacak satır sayısı, varsayılan 5.

    Returns
    -------
    pandas.DataFrame
        Dosyanın ilk ``n_rows`` satırı.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    dataset = ds.dataset(parquet_path)
    table = dataset.head(n_rows)
    return table.to_pandas()
