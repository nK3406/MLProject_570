import pandas as pd

def join_traffic_and_collisions(traffic_csv_path, collisions_parquet_path, output_path):
    """
    Trafik verilerini içeren CSV dosyası ile kaza detaylarını içeren Parquet dosyasını
    ortak ID'ler üzerinden birleştirir ve sonucu yeni bir Parquet dosyası olarak kaydeder.
    """
    try:
        # 1. Gerekli dosyaları Pandas DataFrame olarak oku
        print(f"'{traffic_csv_path}' dosyası okunuyor...")
        df_traffic = pd.read_csv(traffic_csv_path)
        csv_join_column = 'case_id'
        
        if csv_join_column not in df_traffic.columns:
            print(f"HATA: '{traffic_csv_path}' dosyasında '{csv_join_column}' adında bir sütun bulunamadı!")
            return

        print(f"'{collisions_parquet_path}' dosyası okunuyor...")
        df_collisions = pd.read_parquet(collisions_parquet_path)
        parquet_join_column = 'case_id'
        
        print("Dosyalar başarıyla okundu.")

        # --- HATA ÇÖZÜMÜ: Veri tiplerini eşitle ---
        # Birleştirme yapmadan önce her iki anahtar sütunu da metin (string) formatına dönüştür.
        print("ID sütunlarının veri tipleri birleştirme için eşitleniyor...")
        df_traffic[csv_join_column] = df_traffic[csv_join_column].astype(str)
        df_collisions[parquet_join_column] = df_collisions[parquet_join_column].astype(str)
        # -----------------------------------------

        # 2. İki DataFrame'i birleştir (Inner Join)
        print("Veri setleri birleştiriliyor...")
        df_merged = pd.merge(
            left=df_traffic,
            right=df_collisions,
            left_on=csv_join_column,
            right_on=parquet_join_column,
            how='inner'
        )

        # df_merged = df_merged.drop(columns=[parquet_join_column])
        
        print(f"Birleştirme tamamlandı. Sonuçta {len(df_merged)} eşleşen satır bulundu.")

        # 3. Birleştirilmiş DataFrame'i Parquet formatında kaydet
        print(f"Sonuç '{output_path}' dosyasına kaydediliyor...")
        df_merged.to_parquet(output_path, index=False)
        
        print("\nİşlem başarıyla tamamlandı!")
        print(f"Birleştirilmiş veriler '{output_path}' adıyla kaydedildi.")

    except FileNotFoundError as e:
        print(f"HATA: Dosya bulunamadı -> {e.filename}")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")

if __name__ == '__main__':
    TRAFFIC_FILE = 'veriler.csv'
    COLLISIONS_FILE = 'collisions_selected.parquet'
    OUTPUT_FILE = 'kazaTrafik.parquet'
    
    join_traffic_and_collisions(
        traffic_csv_path=TRAFFIC_FILE,
        collisions_parquet_path=COLLISIONS_FILE,
        output_path=OUTPUT_FILE
    )