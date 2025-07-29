import pandas as pd
import numpy as np
import json
from tqdm import tqdm

def preprocess_final_explicit(input_path: str, output_path: str):
    """
One-Hot Encode.
    """
    print("--- NİHAİ VE EN SAĞLAM İŞLEM BAŞLADI ---")
    print(f"1. Adım: '{input_path}' dosyası okunuyor...")
    df = pd.read_parquet(input_path)

    print("\n2. Adım: ID ve Zaman sütunları işleniyor...")
    if 'case_id' in df.columns:
        df.rename(columns={'case_id': 'case_id'}, inplace=True)
    
    df['Time'] = pd.to_datetime(df['Time'])
    print("   -> 'case_id' ve 'Time' sütunları doğru formatlarına getirildi.")

    print("\n3. Adım: Gereksiz sütunlar atılıyor...")
    columns_to_drop = [
        'latitude', 'longitude', 'weather_2', 'direction', 'side_of_highway',
        'killed_victims', 'injured_victims', 'road_condition_2',
        'motorcycle_collision', 'primary_road', 'secondary_road',
        'primary_ramp', 'secondary_ramp', 'collision_date', 'collision_time',
        'lightning' , 'lighting' 
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    print("\n4. Adım: SADECE önceden tanımlanmış kategorik sütunlar işleniyor...")
    
    EXPLICIT_CATEGORICAL_COLS = [
        'chp_beat_type',
        'weather_1',
        'location_type',
        'ramp_intersection',
        'collision_severity',
        'type_of_collision',
        'road_surface',
        'road_condition_1',
        'lighting'
    ]
    
    cols_to_encode = [col for col in EXPLICIT_CATEGORICAL_COLS if col in df.columns]
    print(f"   -> One-Hot Encoding uygulanacak olanlar (AÇIKÇA SEÇİLDİ): {cols_to_encode}")

    if len(cols_to_encode) > 0:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
        print("   -> One-Hot Encoding tamamlandı.")
    else:
        print("   -> Encode edilecek kategorik sütun bulunamadı.")
        
    print("\n5. Adım: Veri, kaza bazında gruplanıp dosyaya yazılıyor...")
    df = df.sort_values(['case_id', 'Time'])
    static_feature_columns = [col for col in df.columns if col not in ['case_id', 'Time', 'sensor_id', 'value']]
    
    num_groups = df['case_id'].nunique()
    
    with open(output_path, 'w') as f:
        for name, group in tqdm(df.groupby('case_id'), total=num_groups, desc="Kazalar işleniyor"):
            if group.empty: continue

            traffic_sequence = group['value'].to_numpy().tolist()
            static_features = group.iloc[0][static_feature_columns].to_numpy(dtype=np.float32).tolist()
            
            record = {'case_id': int(name), 'traffic_sequence': traffic_sequence, 'static_features': static_features}
            f.write(json.dumps(record) + '\n')
            
    print("\n--- İŞLEM BAŞARIYLA TAMAMLANDI ---")
    print(f"Veri '{output_path}' dosyasına kaydedildi.")


if __name__ == '__main__':
    INPUT_FILE = 'kazaTrafik.parquet'
    OUTPUT_FILE = 'kaza_veriseti_islenmis_son.jsonl'
    preprocess_final_explicit(INPUT_FILE, OUTPUT_FILE)