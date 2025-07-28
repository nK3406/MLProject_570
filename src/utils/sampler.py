"""Utility to create a small sample Parquet file for quick experimentation.

Reads the first *n_rows* rows from the partitioned Parquet dataset under
`config.DATA_TRAFIK` and writes them to `config.DATA_INTERIM/sampleParquet.parquet`.

Usage (CLI):
    python -m src.utils.sampler --n_rows 100 --output sampleParquet.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Projeyi paket olarak (python -m src.utils.sampler) **veya** dosyayı
# doğrudan (python src/utils/sampler.py) çalıştırmaya olanak tanımak için
# "config" modülünü hem görece hem de mutlak yoldan deniyoruz.
try:
    from .. import config  # Paket içinden çağrıldığında
except (ImportError, ValueError):  # No parent package
    try:
        from src import config  # Proje kökünden çağrıldığında
    except ModuleNotFoundError:
        import importlib, sys, pathlib

        # src klasörünü sys.path'e ekleyip yeniden dene
        ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
        sys.path.append(str(ROOT_DIR))
        config = importlib.import_module("src.config")


def create_sample(n_rows: int = 100, output_filename: str = "sampleParquet.parquet") -> Path:
    """Create a sample Parquet file with the first *n_rows* rows.

    Parameters
    ----------
    n_rows : int, default 100
        Number of rows to include in the sample.
    output_filename : str, default "sampleParquet.parquet"
        Name of the output Parquet file that will be placed inside
        ``config.DATA_INTERIM``.

    Returns
    -------
    pathlib.Path
        Path to the newly created Parquet file.
    """

    data_trafik = config.DATA_TRAFIK
    data_interim = config.DATA_INTERIM

    if not data_trafik.exists():
        raise FileNotFoundError(f"DATA_TRAFIK directory not found: {data_trafik}")

    # Ensure interim directory exists
    data_interim.mkdir(parents=True, exist_ok=True)

    # Build a PyArrow dataset from partitioned Parquet files
    dataset = ds.dataset(data_trafik, format="parquet")

    # Retrieve first *n_rows* rows efficiently
    try:
        # PyArrow ≥ 8.0.0 supports Dataset.head()
        table = dataset.head(n_rows)
    except AttributeError:  # Fallback for older versions
        scanner = dataset.scanner(limit=n_rows)
        table = scanner.to_table()
    output_path = data_interim / output_filename
    pq.write_table(table, output_path)

    return output_path


def _parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Create a sample Parquet file from DATA_TRAFIK dataset.")
    parser.add_argument("--n_rows", type=int, default=100, help="Number of rows to sample (default: 100)")
    parser.add_argument(
        "--output",
        type=str,
        default="sampleParquet.parquet",
        help="Output filename inside DATA_INTERIM directory (default: sampleParquet.parquet)",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    out_path = create_sample(args.n_rows, args.output)
    print(f"✅ Sample written to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()