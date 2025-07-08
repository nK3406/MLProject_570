# Veri Önişleme Adımları

Bu klasör yapısı, LargeST (`.h5`) → Parquet ve SWITRS (`.sqlite`) → CSV
dönüşümlerini modüler şekilde yürütmek için hazırlanmıştır.

## Hızlı Başlangıç

1. `pip install -r requirements.txt`
2. `jupyter notebook notebooks/00_inspect_largest_h5.ipynb`  
   - HDF5 dosya yapısını inceleyin.
3. `jupyter notebook notebooks/01_prep_largest_to_parquet.ipynb`  
   - Seçili dataset'i Parquet'e dönüştürün.
4. `jupyter notebook notebooks/02_prep_switrs_to_csv.ipynb`  
   - `collisions` tablosunu CSV'ye aktarın.

Çıktılar `data/processed/` altında oluşacaktır.
