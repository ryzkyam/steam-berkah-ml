import os
import datetime
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from supabase import create_client, Client
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # URL Next.js 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- KONFIGURASI SUPABASE ---
SUPABASE_URL = "https://vmhuvevvnoaawaqlrlzd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtaHV2ZXZ2bm9hYXdhcWxybHpkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg1OTk1MTUsImV4cCI6MjA4NDE3NTUxNX0.vzSfGDIy2hQTZdR1MB2xRl9mCTSiyAuzQQJNANvREKo"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


@app.get("/api/prediksi-omset")
def get_prediksi_omset(branch_id: int):
    try:
        # 1. Ambil seluruh data transaksi historis dari Supabase tanpa filter batasan tahun yang ketat
        response = (
            supabase.table("Transaksis")
            .select("Tanggal, Total")
            .eq("branch_id", branch_id)
            .order("Tanggal", desc=False)
            .range(0, 10000)
            .execute()
        )
        data = response.data

        # Fallback aman jika data dari DB kosong / kurang untuk training
        if not data or len(data) < 3:
            today = datetime.date.today()
            return [{
                "ds": (today + datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
                "yhat": 1500000,
                "yhat_lower": 1000000,
                "yhat_upper": 2000000
            } for i in range(7)]

        # 2. Convert ke Pandas DataFrame & Pembersihan Format Tanggal Murni
        df = pd.DataFrame(data)
        df = df.rename(columns={"Tanggal": "ds", "Total": "y"})
        
        # PANGKAS JAM/TIMEZONE: Paksa jadi format tanggal murni (YYYY-MM-DD)
        df['ds'] = pd.to_datetime(df['ds']).dt.date
        
        # Pastikan nominal omset berupa angka murni
        df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0)
        
        # WAJIB: Gumpalkan seluruh omset motor berdasarkan hari yang sama
        df = df.groupby('ds')['y'].sum().reset_index()
        
        # Kembalikan tipe menjadi datetime agar disukai oleh Prophet object
        df['ds'] = pd.to_datetime(df['ds'])

        # Fallback jika setelah di-grouping jumlah hari unik masih kurang dari 3
        if len(df) < 3:
            today = datetime.date.today()
            return [{
                "ds": (today + datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
                "yhat": 1200000,
                "yhat_lower": 800000,
                "yhat_upper": 1600000
            } for i in range(7)]

        # 3. Setup Model Prophet dengan Batasan Logistik Rasional
        model = Prophet(growth='logistic', weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        
        # Batas kapasitas omset harian rasional (Floor 0, Cap 5 Juta)
        df['floor'] = 0
        df['cap'] = 5000000
        
        model.fit(df)

        # 4. Buat Horizon Prediksi 7 Hari ke Depan (Secara Live)
        today = datetime.date.today() 
        future_dates = [today + datetime.timedelta(days=i) for i in range(7)]
        
        future = pd.DataFrame({'ds': future_dates})
        future['floor'] = 0
        future['cap'] = 5000000

        # 5. Eksekusi Model Forecasting
        forecast = model.predict(future)

        # 6. Ambil Kolom Esensial & Proteksi Angka Minus/Anomali
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result['yhat'] = result['yhat'].fillna(0).clip(lower=0)
        result['yhat_lower'] = result['yhat_lower'].fillna(0).clip(lower=0)
        result['yhat_upper'] = result['yhat_upper'].fillna(0).clip(lower=0)
        
        # Format tanggal menjadi string ramah JSON
        result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')
        
        return result.to_dict(orient="records")

    except Exception as e:
        print(f"Error pada endpoint prediksi branch {branch_id}: {str(e)}")
        today = datetime.date.today()
        return [{
            "ds": (today + datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
            "yhat": 1400000,
            "yhat_lower": 900000,
            "yhat_upper": 1900000
        } for i in range(7)]


@app.get("/api/segmentasi-pelanggan")
def get_segmentation(branch_id: int):
    try:
        response = supabase.table("Transaksis")\
            .select("Tanggal, Total, customer_id")\
            .eq("branch_id", branch_id)\
            .execute()
        
        data = response.data
        
        # Validasi data: Harus ada data
        if not data or len(data) < 5:
            return {"cluster_terbesar": "DATA KURANG", "persentase": 0, "detail": "Data tidak cukup."}

        df = pd.DataFrame(data)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='mixed', utc=True)
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)

        # Pembuatan Fitur
        if 'customer_id' in df.columns and df['customer_id'].notna().sum() > 0:
            max_date = df['Tanggal'].max()
            X = df.groupby('customer_id').agg({
                'Tanggal': lambda x: (max_date - x.max()).days,
                'customer_id': 'count',
                'Total': 'sum'
            })
            X.columns = ['Recency', 'Frequency', 'Monetary']
        else:
            df['Hari'] = df['Tanggal'].dt.date
            X = df.groupby('Hari').agg({'Total': ['sum', 'count']})
            X.columns = ['Total_Omset', 'Jumlah_Transaksi']
        
        # BERSIHKAN DATA (Penting agar tidak ERROR)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        
        # JIKA DATA TERLALU SEDIKIT UNTUK CLUSTERING
        if len(X) < 2:
            return {"cluster_terbesar": "REGULER", "persentase": 100, "detail": "Data terlalu sedikit."}
        
        # Clustering K-Means
        # Gunakan min() agar n_clusters tidak lebih besar dari jumlah data
        n_clusters = min(3, len(X))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        X['cluster_label'] = kmeans.fit_predict(X_scaled)

        counts = X['cluster_label'].value_counts()
        persentase = round((counts.max() / len(X)) * 100, 1)
        
        return {
            "cluster_terbesar": "SEGMEN AKTIF",
            "persentase": persentase,
            "detail": "Analisis sukses."
        }

    except Exception as e:
        # LOG DETAIL ERROR KE TERMINAL
        print(f"DEBUG ERROR: {str(e)}") 
        return {"cluster_terbesar": "ERROR", "persentase": 0, "detail": str(e)}
    try:
        
        response = supabase.table("Transaksis").select("Tanggal, Total, customer_id").eq("branch_id", branch_id).execute()
        data = response.data
        
        if not data or len(data) < 5:
            return {"cluster_terbesar": "DATA MINIM", "persentase": 0, "detail": "Data tidak cukup."}

        df = pd.DataFrame(data)
        
        # --- PERBAIKAN DI SINI ---
        # Gunakan 'mixed' untuk menangani berbagai variasi format ISO 8601 dari Supabase
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='mixed', utc=True)
        # -------------------------
        
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)

        if 'customer_id' in df.columns and df['customer_id'].notna().sum() > 0:
            max_date = df['Tanggal'].max()
            X = df.groupby('customer_id').agg({
                'Tanggal': lambda x: (max_date - x.max()).days, 
                'customer_id': 'count', 
                'Total': 'sum'
            })
            X.columns = ['Recency', 'Frequency', 'Monetary']
        else:
            df['Hari'] = df['Tanggal'].dt.date
            X = df.groupby('Hari').agg({'Total': ['sum', 'count']})
            X.columns = ['Total_Omset', 'Jumlah_Transaksi']
        
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Sisa logika KMeans tetap sama...
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        X['cluster_label'] = kmeans.fit_predict(X_scaled)

        counts = X['cluster_label'].value_counts()
        persentase = round((counts.max() / len(X)) * 100, 1)
        
        return {"cluster_terbesar": "SEGMEN AKTIF", "persentase": persentase, "detail": "Analisis sukses."}
    except Exception as e:
        print(f"Error di Backend: {str(e)}") # Cek terminal untuk detailnya
        return {"status": "error", "message": str(e)}
    try:
        # 1. Tarik data transaksi dari Supabase berdasarkan branch_id
        response = (
            supabase.table("Transaksis")
            .select("Tanggal, Total, customer_id")
            .eq("branch_id", branch_id)
            .execute()
        )
        data = response.data

        if not data:
            return {
                "cluster_terbesar": "NO DATA",
                "persentase": 0,
                "detail": "Data transaksi untuk cabang ini masih kosong, bro."
            }

        df = pd.DataFrame(data)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df['Total'] = pd.to_numeric(df['Total'])

        # Cek apakah kolom customer_id ada isinya dan valid
        if 'customer_id' in df.columns and df['customer_id'].notna().sum() > 0:
            # --- STRATEGI A: SEGMENTASI BERBASIS PELANGGAN (RFM) ---
            max_date = df['Tanggal'].max()
            
            rfm = df.groupby('customer_id').agg({
                'Tanggal': lambda x: (max_date - x.max()).days, # Recency
                'customer_id': 'count',                         # Frequency
                'Total': 'sum'                                  # Monetary
            }).rename(columns={'Tanggal': 'Recency', 'customer_id': 'Frequency', 'Total': 'Monetary'}).reset_index()
            
            X = rfm[['Recency', 'Frequency', 'Monetary']]
            mode_type = "Pelanggan"
        else:
            # --- STRATEGI B: SEGMENTASI BERBASIS TRANSAKSI HARIAN (FALLBACK) ---
            df['Hari'] = df['Tanggal'].dt.date
            daily_features = df.groupby('Hari').agg({
                'Total': ['sum', 'count', 'mean']
            })
            daily_features.columns = ['Total_Omset', 'Jumlah_Transaksi', 'Rata_Rata_Transaksi']
            X = daily_features.reset_index(drop=True)
            mode_type = "Transaksi Harian"

        # Proteksi jumlah data untuk K-Means
        n_clusters = 3
        if len(X) < n_clusters:
            return {
                "cluster_terbesar": "REGULAR CUSTOMER",
                "persentase": 100,
                "detail": f"Data {mode_type} terlalu sedikit ({len(X)}) untuk clustering. Otomatis masuk kategori reguler."
            }

        # 2. Eksekusi Algoritma K-Means Clustering dengan Normalisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        X['cluster_label'] = kmeans.fit_predict(X_scaled)

        # 3. Hitung Cluster Terbesar
        cluster_counts = X['cluster_label'].value_counts()
        cluster_terbesar_id = int(cluster_counts.idxmax())
        total_data = len(X)
        jumlah_cluster_terbesar = int(cluster_counts.max())
        persentase = round((jumlah_cluster_terbesar / total_data) * 100, 1)

        # Mapping Karakteristik Cluster ke Istilah Bisnis
        #  
        centroids = kmeans.cluster_centers_
        
        mapping_nama = {}
        if mode_type == "Pelanggan":
            idx_loyal = centroids[:, 2].argmax() # Monetary tertinggi
            idx_churn = centroids[:, 0].argmax() # Recency terlama
            
            mapping_nama[idx_loyal] = "LOYAL CUSTOMER"
            mapping_nama[idx_churn] = "AT-RISK / CHURN"
            for i in range(3):
                if i not in mapping_nama:
                    mapping_nama[i] = "NEW / OCCASIONAL"
        else:
            idx_high = centroids[:, 0].argmax() # Omset harian tertinggi
            idx_low = centroids[:, 0].argmin()  # Omset harian terendah
            
            mapping_nama[idx_high] = "HIGH-TRAFFIC DAYS"
            mapping_nama[idx_low] = "LOW-TRAFFIC DAYS"
            for i in range(3):
                if i not in mapping_nama:
                    mapping_nama[i] = "NORMAL DAYS"

        nama_cluster_terbesar = mapping_nama.get(cluster_terbesar_id, "REGULAR CUSTOMER")

        return {
            "cluster_terbesar": nama_cluster_terbesar,
            "persentase": persentase,
            "detail": f"Analisis K-Means berbasis {mode_type}. Menghasilkan segmentasi dominan pada kelompok {nama_cluster_terbesar}."
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    

    except Exception as e:
        print(f"DEBUG ERROR: {str(e)}") # <--- INI PENTING
        return {"status": "error", "message": str(e)}

@app.get("/api/analisis-produk")
def get_analisis_produk(branch_id: int):
    try:
        # 1. Tarik kolom Layanan dari Transaksis DAN Kategori dari tabel Motors (Join Relasi Supabase)
        response = (
            supabase.table("Transaksis")
            .select("Layanan, Motors(Kategori)")
            .eq("branch_id", branch_id)
            .execute()
        )
        data = response.data

        # Fallback jika data transaksi di cabang tersebut masih kosong
        if not data:
            return {
                "motor_terbanyak": "BELUM ADA DATA",
                "layanan_laris_lain": "BELUM ADA DATA"
            }

        # 2. Parsing data mentah ke dalam Pandas DataFrame
        parsed_records = []
        for item in data:
            # Ambil nilai string layanan langsung
            layanan_nama = item.get("Layanan", "Cuci Standar")
            
            # Ambil nilai kategori motor dari object relasi tabel Motors
            motor_obj = item.get("Motors", {})
            kategori_motor = motor_obj.get("Kategori", "SMALL") if motor_obj else "SMALL"
            
            parsed_records.append({
                "Layanan": layanan_nama,
                "Kategori": kategori_motor
            })

        df = pd.DataFrame(parsed_records)

        # 3. Hitung Kategori Motor Terbanyak di cabang ini
        motor_terbanyak = "SMALL"
        if not df['Kategori'].dropna().empty:
            motor_terbanyak = df['Kategori'].value_counts().idxmax()

        # 4. Hitung Layanan Terlaris SELAIN "Cuci Standar" / "Cuci Standard"
        layanan_laris_lain = "TIDAK ADA"
        # Kita antisipasi pencatatan tulisan Standar atau Standard
        df_filtered = df[~df['Layanan'].str.contains("standar|standard", case=False, na=True)]
        
        if not df_filtered.empty and not df_filtered['Layanan'].dropna().empty:
            layanan_laris_lain = df_filtered['Layanan'].value_counts().idxmax()

        return {
            "motor_terbanyak": str(motor_terbanyak).upper(),
            "layanan_laris_lain": str(layanan_laris_lain).upper()
        }

    except Exception as e:
        print(f"Error Analisis Produk branch {branch_id}: {str(e)}")
        return {
            "motor_terbanyak": "ERROR DATA",
            "layanan_laris_lain": f"Detail: {str(e)}"
        }
    
if __name__ == "__main__":
    # Ini cara supaya Railway bisa menjalankan aplikasi kamu dengan port yang tepat
    port = int(os.environ.get("PORT", 8000))
    # Host 0.0.0.0 wajib supaya aplikasi bisa diakses dari luar (internet)
    uvicorn.run(app, host="0.0.0.0", port=port)