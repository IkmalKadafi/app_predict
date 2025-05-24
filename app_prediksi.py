import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta
import tensorflow as tf

st.set_page_config(
    page_title="Prediksi Musim Jawa Timur",
    page_icon="⛅"
)

# Load model
model_311 = tf.keras.models.load_model('Data/model/model_311.h5')
model_303 = tf.keras.models.load_model('Data/model/model_303.h5')
model_349 = tf.keras.models.load_model('Data/model/model_349.h5')

# Load data
data_311 = pd.read_excel("Data/data/data_311.xlsx")
data_303 = pd.read_excel("Data/data/data_303.xlsx")
data_349 = pd.read_excel("Data/data/data_349.xlsx")

# Load scalers
scaler_x_311 = joblib.load('Data/scaler/scaler_x_311.pkl')
scaler_x_303 = joblib.load('Data/scaler/scaler_x_303.pkl')
scaler_x_349 = joblib.load('Data/scaler/scaler_x_349.pkl')

scaler_y_311 = joblib.load('Data/scaler/scaler_y_311.pkl')
scaler_y_303 = joblib.load('Data/scaler/scaler_y_303.pkl')
scaler_y_349 = joblib.load('Data/scaler/scaler_y_349.pkl')

def pilih_topografi():
    dropdown = st.selectbox(
        "Pilih Topografi yang diinginkan:",
        ("Dataran Tinggi", "Dataran Rendah", "Pesisir")
    )
    if dropdown == 'Dataran Tinggi':
        model = model_311
        data = data_311
        scaler_x = scaler_x_311
        scaler_y = scaler_y_311
    elif dropdown == 'Dataran Rendah':
        model = model_303
        data = data_303
        scaler_x = scaler_x_303
        scaler_y = scaler_y_303
    elif dropdown == 'Pesisir':
        model = model_349
        data = data_349
        scaler_x = scaler_x_349
        scaler_y = scaler_y_349
    st.write(f'Jenis topografi **{dropdown}** telah dipilih.')
    return model, data, scaler_x, scaler_y, dropdown

def pilih_musim():
    musim = st.selectbox(
        "Pilih Musim yang Ingin Diprediksi:",
        ("Musim Kemarau", "Musim Hujan")
    )
    st.write(f"Musim yang dipilih: **{musim}**")
    return musim

def buat_sequence(data_x, look_back=36):
    X_seq = []
    for i in range(len(data_x) - look_back):
        X_seq.append(data_x[i:i+look_back])
    return np.array(X_seq)

def prediksi_recursive(model, scaler_y, X_test_seq, n_future=36):
    recursive_input = X_test_seq[-1].copy()
    predictions = []
    for _ in range(n_future):
        input_reshaped = recursive_input.reshape(1, recursive_input.shape[0], recursive_input.shape[1])
        next_pred = model.predict(input_reshaped, verbose=0)
        predictions.append(next_pred[0, 0])
        # Update recursive_input: geser 1 langkah, tambahkan prediksi sebagai fitur RR, pertahankan fitur lain (misal fitur kedua)
        recursive_input = np.append(recursive_input[1:], [[next_pred[0, 0], recursive_input[-1, 1]]], axis=0)
    pred_array = np.array(predictions).reshape(-1, 1)
    pred_array_rescaled = scaler_y.inverse_transform(pred_array)
    return pred_array_rescaled

def detect_seasons(pred_array_rescaled, start_date, days_per_dasarian=10):
    n = len(pred_array_rescaled)
    dates = [start_date + timedelta(days=days_per_dasarian * i) for i in range(n)]

    df = pd.DataFrame({
        'Tanggal': dates,
        'RR': pred_array_rescaled.flatten()
    })
    df['Bulan'] = df['Tanggal'].dt.month

    # Awal musim hujan
    awal_hujan = None
    for i in range(n - 2):
        window = pred_array_rescaled[i:i+3].flatten()
        if (window[0] >= 50) and (np.all(window >= 50) or np.sum(window) >= 150):
            awal_hujan = i
            break

    # Puncak musim hujan
    puncak_hujan_idx = None
    max_total = -np.inf
    for i in range(n - 2):
        window = pred_array_rescaled[i:i+3].flatten()
        total = np.sum(window)
        if total > max_total:
            max_total = total
            puncak_hujan_idx = i

    bulan_window_hujan = df['Bulan'][puncak_hujan_idx:puncak_hujan_idx+3]
    bulan_dominan_hujan = bulan_window_hujan.mode().iloc[0]

    # Awal musim kemarau
    awal_kemarau = None
    if awal_hujan is not None:
        for i in range(awal_hujan + 3, n - 2):
            window = pred_array_rescaled[i:i+3].flatten()
            if (window[0] < 50) and (np.all(window < 50) or np.sum(window) < 150):
                awal_kemarau = i
                break

    # Puncak musim kemarau
    puncak_kemarau_idx = None
    min_total = np.inf
    for i in range(n - 2):
        window = pred_array_rescaled[i:i+3].flatten()
        total = np.sum(window)
        if total < min_total:
            min_total = total
            puncak_kemarau_idx = i

    window_zero = pred_array_rescaled[puncak_kemarau_idx:puncak_kemarau_idx+3].flatten()
    if np.all(window_zero == 0):
        puncak_kemarau_idx += 1  # dasarian tengah

    bulan_window_kemarau = df['Bulan'][puncak_kemarau_idx:puncak_kemarau_idx+3]
    bulan_dominan_kemarau = bulan_window_kemarau.mode().iloc[0]

    # Durasi musim hujan
    durasi_hujan = None
    if awal_hujan is not None and awal_kemarau is not None:
        durasi_hujan = awal_kemarau - awal_hujan

    # Durasi musim kemarau
    durasi_kemarau = None
    if awal_kemarau is not None:
        akhir_kemarau_idx = awal_hujan if (awal_hujan is not None and awal_hujan > awal_kemarau) else n
        durasi_kemarau = akhir_kemarau_idx - awal_kemarau

    def idx_to_str(idx):
        if idx is None:
            return "Tidak terdeteksi"
        dasarian_ke = idx + 1
        tanggal = start_date + timedelta(days=days_per_dasarian * idx)
        return f"{tanggal.strftime('%Y-%m-%d')} (dasarian ke-{dasarian_ke})"

    result = {
        "musim_hujan": {
            "awal": idx_to_str(awal_hujan),
            "puncak": f"{idx_to_str(puncak_hujan_idx)}, Bulan dominan: {bulan_dominan_hujan}",
            "durasi (dasarian)": durasi_hujan if durasi_hujan is not None else "Tidak terdeteksi"
        },
        "musim_kemarau": {
            "awal": idx_to_str(awal_kemarau),
            "puncak": f"{idx_to_str(puncak_kemarau_idx)}, Bulan dominan: {bulan_dominan_kemarau}",
            "durasi (dasarian)": durasi_kemarau if durasi_kemarau is not None else "Tidak terdeteksi"
        }
    }
    return result

def main():
    st.title("Prediksi Musim di Jawa Timur")

    # Pilih topografi
    model, data, scaler_x, scaler_y, zona = pilih_topografi()

    # Pilih musim
    musim = pilih_musim()

    look_back = 36

    # Siapkan data input fitur untuk prediksi
    # Misal data fitur di kolom 1 ke seterusnya (asumsi sesuai data), 
    # scaling sudah ada, ini contoh buat sequence input
    X_all = data.iloc[:, 1:].values
    X_scaled = scaler_x.transform(X_all)

    X_test_seq = buat_sequence(X_scaled, look_back=look_back)

    # Prediksi recursive
    pred_array_rescaled = prediksi_recursive(model, scaler_y, X_test_seq, n_future=36)

    # Tampilkan hasil prediksi per dasarian
    st.subheader("Hasil Prediksi Curah Hujan (mm) 36 Dasarian ke Depan:")
    for i, val in enumerate(pred_array_rescaled):
        st.write(f"Prediksi ke-{i+1}: {val[0]:.2f} mm")

    # Tanggal mulai prediksi diasumsikan
    start_date = pd.to_datetime("2024-10-01")

    # Deteksi musim dari hasil prediksi
    result = detect_seasons(pred_array_rescaled, start_date)

    st.subheader("Hasil Deteksi Musim:")
    if musim == "Musim Hujan":
        st.write("=== MUSIM HUJAN ===")
        st.write(f"Awal     : {result['musim_hujan']['awal']}")
        st.write(f"Puncak   : {result['musim_hujan']['puncak']}")
        st.write(f"Durasi   : {result['musim_hujan']['durasi (dasarian)']} dasarian\n")
    else:
        st.write("=== MUSIM KEMARAU ===")
        st.write(f"Awal     : {result['musim_kemarau']['awal']}")
        st.write(f"Puncak   : {result['musim_kemarau']['puncak']}")
        st.write(f"Durasi   : {result['musim_kemarau']['durasi (dasarian)']} dasarian\n")

    # Data normal musim untuk perbandingan
    normal_musim = {
        'musim_hujan': {
            '303': {'awal': 'November III', 'akhir': 'April III', 'durasi': 16},
            '311': {'awal': 'November I', 'akhir': 'April III', 'durasi': 18},
            '349': {'awal': 'November II', 'akhir': 'Mei I', 'durasi': 18},
        },
        'musim_kemarau': {
            '303': {'awal': 'April III', 'akhir': 'November II', 'durasi': 21},
            '311': {'awal': 'April III', 'akhir': 'Oktober III', 'durasi': 19},
            '349': {'awal': 'Mei I', 'akhir': 'November I', 'durasi': 19},
        }
    }

    st.subheader(f"Data Normal Musim Zona {zona}")
    if musim == "Musim Hujan":
        st.write("Musim Hujan:")
        st.write(f"Awal Normal  : {normal_musim['musim_hujan'][zona]['awal']}")
        st.write(f"Akhir Normal : {normal_musim['musim_hujan'][zona]['akhir']}")
        st.write(f"Durasi       : {normal_musim['musim_hujan'][zona]['durasi']} dasarian\n")
    else:
        st.write("Musim Kemarau:")
        st.write(f"Awal Normal  : {normal_musim['musim_kemarau'][zona]['awal']}")
        st.write(f"Akhir Normal : {normal_musim['musim_kemarau'][zona]['akhir']}")
        st.write(f"Durasi       : {normal_musim['musim_kemarau'][zona]['durasi']} dasarian\n")

if __name__ == "__main__":
    main()
