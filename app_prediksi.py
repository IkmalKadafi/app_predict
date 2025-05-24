import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="Prediksi Musim Jawa Timur",
    page_icon="⛅"
)

# Load model dan data serta scaler (sesuaikan path-nya)
model_311 = tf.keras.models.load_model('Data/model/model_311.h5', custom_objects={'mse': tf.keras.losses.mse})
model_303 = tf.keras.models.load_model('Data/model/model_303.h5', custom_objects={'mse': tf.keras.losses.mse})
model_349 = tf.keras.models.load_model('Data/model/model_349.h5', custom_objects={'mse': tf.keras.losses.mse})

data_311 = pd.read_excel("Data/data/data_311.xlsx")
data_303 = pd.read_excel("Data/data/data_303.xlsx")
data_349 = pd.read_excel("Data/data/data_349.xlsx")

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
    else:
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
        recursive_input = np.append(recursive_input[1:], [[next_pred[0, 0], recursive_input[-1, 1]]], axis=0)
    pred_array = np.array(predictions).reshape(-1, 1)
    pred_array_rescaled = scaler_y.inverse_transform(pred_array)
    return pred_array_rescaled

def main():
    st.title("Prediksi Musim di Jawa Timur")

    model, data, scaler_x, scaler_y, zona = pilih_topografi()
    musim = pilih_musim()
    look_back = 36

    # Ambil fitur dari data, abaikan kolom tanggal (kolom pertama)
    X_all = data.iloc[:, 1:].values

    st.write(f"Jumlah fitur data input sebelum transform: {X_all.shape[1]}")
    st.write(f"Jumlah fitur yang diharapkan scaler_x: {scaler_x.scale_.shape[0]}")

    expected_features = scaler_x.scale_.shape[0]
    if X_all.shape[1] > expected_features:
        X_all = X_all[:, :expected_features]
        st.write(f"Jumlah fitur input disesuaikan menjadi: {X_all.shape[1]}")

    # Scaling fitur input
    X_scaled = scaler_x.transform(X_all)

    # Buat sequence data input sesuai look_back
    X_seq = buat_sequence(X_scaled, look_back)

    st.write(f"Bentuk data input sequence untuk prediksi: {X_seq.shape}")

    # Prediksi dengan pendekatan rekursif
    pred_rescaled = prediksi_recursive(model, scaler_y, X_seq)

    # Tampilkan hasil prediksi (bisa dimodifikasi sesuai kebutuhan)
    st.write("Hasil prediksi curah hujan (mm):")
    st.dataframe(pd.DataFrame(pred_rescaled, columns=["Curah Hujan Prediksi"]))

if __name__ == "__main__":
    main()
