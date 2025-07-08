"""
SWITRS SQLite → CSV aktarımı için araçlar (chunk destekli).
"""
import sqlite3
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def export_sqlite_table_to_csv(
    sqlite_path: Path,
    table_name: str,
    out_csv_path: Path,
    chunksize: int = 200_000,
    columns: list[str] | None = None,
) -> None:
    """
    Belirtilen tabloyu chunk'lar hâlinde CSV'ye aktarır.
    """
    con = sqlite3.connect(sqlite_path)
    sql = f"SELECT {', '.join(columns) if columns else '*'} FROM {table_name}"
    first = True

    for chunk in tqdm(pd.read_sql_query(sql, con, chunksize=chunksize)):
        chunk.to_csv(out_csv_path, mode="w" if first else "a",
                     header=first, index=False)
        first = False
    con.close()
