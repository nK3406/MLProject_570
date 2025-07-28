# README – Veri Ön İşleme (Preprocessing) Pipeline

Bu belge, `MLProject_570` deposunda ham veriden **LSTM modeline** beslenmeye hazır
nümerik tensörlere kadar takip edilen tüm veri ön işleme (preprocessing)
adımlarını ayrıntılı olarak açıklar. Amaç, süreci yeniden çalıştırmak veya
kendi verinize uygulamak isteyen araştırmacılara **uçtan uca yol haritası**
sunmaktır.

> 📁 **Kısa Özet**  
> 1. Ham kaynak dosyalarını içe aktarın (LargeST *HDF5* & SWITRS *SQLite*).  
> 2. Notebook’larla biçim dönüşümlerini yapın (HDF5→Parquet, SQLite→CSV).  
> 3. `src/utils/lstm_prep.py` ile Parquet + meta veriyi LSTM dizilerine çevirin.  
> 4. Çıktıları `data/processed/` ve `preprocessed_data/` altında bulun.

---

## 1 | Kurulum

```bash
# Sanal ortam (önerilir)
python -m venv .venv && source .venv/bin/activate

# Gerekli paketler
pip install -r requirements.txt
```

Python ≥ 3.9, Dask ve Scikit-learn uyumlu bir ortam gereklidir.

---

## 2 | Hammadde Kaynakları

| Kaynak | Biçim | Yol / URL | İçerik |
| ------ | ----- | --------- | ------ |
| **LargeST** | `.h5` | `data/raw/largest.h5` | Kaliforniya 5-dk trafik yoğunluk sensör kayıtları |
| **SWITRS** | `.sqlite` | `data/raw/SWITRS.sqlite` | Kaza veritabanı (2008-2019) |
| **Meta** | `.csv` | `data/raw/ca_meta.csv` | Sensör konum & özellikler |

---

## 3 | Notebook Tabanlı Dönüşümler

### 3.1 `00_inspect_largest_h5.ipynb`
* **Amaç:** HDF5 hiyerarşisini keşfetmek, uygun sensör ve
yıl aralıklarını belirlemek.
* **Çıktı:** Seçilen düğümlerin listesi (`largest_selected.json`).

### 3.2 `01_prep_largest_to_parquet.ipynb`
* **Girdi:** `largest.h5` & seçim listesi.
* **İşlem Adımları**
  1. HDF5 tabakalarını okunabilir parçalara böler.
  2. Her parçayı Dask ile DataFrame’e aktarır.
  3. Blok boyutu = 100 MB olacak şekilde **Parquet** dosyalarına yazar.
* **Çıktı:** `data/processed/largest_wide_2017/` (partitioned Parquet).

### 3.3 `02_prep_switrs_to_csv.ipynb`
* **Girdi:** `SWITRS.sqlite`.
* **İşlem Adımları**
  1. `collisions` tablosunu SQLite’dan çek.
  2. Gereksiz sütunları düşür; eksik kodlamaları düzelt.
  3. Son DataFrame’i **CSV** olarak kaydet.
* **Çıktı:** `data/processed/collisions_2017.csv`.

> **Not:** Notebook’lar görsel inceleme ve ara adım doğrulaması için
> interaktiftir; tam otomasyon tercih ediyorsanız aynı kodu bir Python
> betiğine taşıyabilirsiniz.

---

## 4 | LSTM’e Hazırlık – `src/utils/lstm_prep.py`
Aşağıdaki fonksiyonlar boru hattının kodlanmış halidir. Komut satırından
veya başka bir modülden çağrılabilir.

| Adım | Fonksiyon | Açıklama |
| ---- | --------- | -------- |
| 1 | `load_data` | Parquet (büyük) + meta CSV’yi Dask & Pandas ile yükler. |
| 2 | `impute_missing` | %50’den fazla eksik veriye sahip sensörleri eler, kalan
  değerleri **lineer enterpolasyon** ile doldurur. |
| 3 | `preprocess_timestamps` | Zaman damgalarını `datetime64[ns]`’e çevirir,
  5-dakikalık aralıklarla **yeniden örnekler** ve enterpole eder. |
| 4 | `merge_metadata` | Sensör-ID ile meta veriyi birleştirir; sayısallar
  Min-Max ölçeklenir, kategorikler **One-Hot** kodlanır. |
| 5 | `create_sequences` | Her sensör için [seq_length] uzunlukta kayan pencereler
  oluşturur; meta özellikleri her time-step’e ekler. |
| 6 | `split_data` | Dizileri **train / val / test** oranlarına göre böler. |
| 7 | `save_sequences` | Numpy `.npy` olarak diske yazar (örn. `train_sequences.npy`). |

Öntanımlı parametreler:
```python
train_ratio = 0.70
val_ratio   = 0.20  # Test = 0.10
seq_length  = 12    # 12 × 5-dk = 1 saatlik pencere
overlap     = 12    # %100 örtüşme
```

Çalıştırma örneği:
```bash
python -m src.utils.lstm_prep \
  --parquet_path data/processed/largest_wide_2017 \
  --meta_path    data/raw/ca_meta.csv \
  --seq_length 12 --overlap 12
```

---

## 5 | Çıktı Dizinleri

```
├── data/
│   └── processed/
│       ├── largest_wide_2017/   # Partitioned Parquet
│       └── collisions_2017.csv
└── preprocessed_data/
    ├── train_sequences.npy
    ├── val_sequences.npy
    ├── test_sequences.npy
    ├── train_targets.npy
    ├── ...
    └── processed_meta.csv
```

Bu çıktılar doğrudan *model eğitimi* aşamasında kullanılmaya hazırdır.

---

## 6 | İpuçları & Sorun Giderme

* **Bellek Hatası:** `dd.read_parquet(..., blocksize="50MB")` parametresiyle
  boyutları küçültün.
* **Tarih Uyumsuzluğu:** HDF5 sensör zaman aralıkları yıl bazında değişebilir;
  notebook’ta filtreleme yaptığınızdan emin olun.
* **Meta-Veri Eksik:** `ca_meta.csv` içinde sensör ID eşleşmiyorsa ilgili
  sensörü **impute_missing** adımında zaten elenmiş olabilir.

---

🎉 Artık veriniz LSTM tabanlı trafik yoğunluk tahmini için hazır!
