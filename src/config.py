"""Proje genel yolları & temel ayarlar."""
from pathlib import Path

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW       = PROJECT_ROOT / "data" / "kaza"
DATA_INTERIM   = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "trafik"

# Dış veri klasörleri (bu yolları kendi ortamınıza göre güncelleyin)
LARGEST_DIR   = Path("/home/bistek/Downloads/archive_largest")
SWITRS_SQLITE = Path("/home/bistek/Downloads/archive/switrs.sqlite")
