import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings
import joblib
warnings.filterwarnings("ignore")

# Cấu hình để tận dụng GPU nếu có
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --- 1. Đọc và xử lý dữ liệu ---
print("Loading and preprocessing data...")
df = pd.read_csv("global_unemployment_data.csv")

# Chuyển từ wide -> long format
df_long = df.melt(
    id_vars=["country_name", "sex", "age_group", "age_categories"],
    value_vars=[str(year) for year in range(2014, 2025)],
    var_name="year",
    value_name="unemployment_rate"
)
df_long["year"] = df_long["year"].astype(int)
print(df_long)
# Loại bỏ dữ liệu thiếu và xử lý outliers
df_model = df_long.dropna(subset=["unemployment_rate"]).copy()
Q1 = df_model["unemployment_rate"].quantile(0.25)
Q3 = df_model["unemployment_rate"].quantile(0.75)
IQR = Q3 - Q1
df_model = df_model[
    (df_model["unemployment_rate"] >= Q1 - 1.5 * IQR) & 
    (df_model["unemployment_rate"] <= Q3 + 1.5 * IQR)
]

# Mã hóa các biến phân loại
label_encoders = {}
for col in ["country_name", "sex", "age_group"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Chuẩn hóa dữ liệu
scaler_rate = MinMaxScaler()
scaler_year = MinMaxScaler()
df_model["unemployment_rate_scaled"] = scaler_rate.fit_transform(df_model[["unemployment_rate"]])
df_model["year_scaled"] = scaler_year.fit_transform(df_model[["year"]])

# Lưu scaler và encoder để dùng ở backend
joblib.dump(scaler_rate, 'scaler_rate.save')
joblib.dump(scaler_year, 'scaler_year.save')
joblib.dump(label_encoders, 'label_encoders.save')

# --- 2. Tạo chuỗi thời gian với augmentation ---
def create_sequences(data, time_steps=5):
    X, y = [], []
    groups = data.groupby(["country_name", "sex", "age_group"])
    for _, group in groups:
        group = group.sort_values("year")
        if len(group) <= time_steps:
            continue
        features = group[["country_name", "sex", "age_group", "year_scaled", "unemployment_rate_scaled"]].values
        target = group["unemployment_rate_scaled"].values
        for i in range(len(group) - time_steps):
            X.append(features[i:i + time_steps])
            y.append(target[i + time_steps])
    return np.array(X), np.array(y)

# Tạo sequences với time steps dài hơn và augmentation
time_steps = 5  # Tăng time steps để capture xu hướng dài hạn hơn
X, y = create_sequences(df_model, time_steps)

# Chia dữ liệu với validation set
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

# --- 3. Xây dựng mô hình LSTM cải tiến ---
# Cấu trúc mô hình BiLSTM này dựa trên phác thảo trong file 'sơ đồ BiLSTM.py'
def build_advanced_lstm_model(input_shape):
    model = Sequential([
        # First Bidirectional LSTM layer
        Bidirectional(LSTM(128, activation='tanh', return_sequences=True), 
                     input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second Bidirectional LSTM layer
        Bidirectional(LSTM(64, activation='tanh', return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third Bidirectional LSTM layer
        Bidirectional(LSTM(32, activation='tanh')),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')  # Sigmoid vì đã normalize dữ liệu về khoảng [0,1]
    ])
    
    return model

# Khởi tạo mô hình
model = build_advanced_lstm_model((time_steps, X.shape[2]))

# Compile với optimizer được tinh chỉnh
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])

# Callbacks để tối ưu quá trình huấn luyện
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,  # Giảm patience xuống để dừng sớm hơn nếu không cải thiện
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Huấn luyện mô hình với batch size và epochs được tinh chỉnh
print("\nTraining model...")
print("Note: Using EarlyStopping, training will stop automatically if validation loss stops improving")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  
    batch_size=32,  
    callbacks=callbacks,
    verbose=1
)

# --- 4. Đánh giá mô hình ---
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_test_inv = scaler_rate.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler_rate.inverse_transform(y_pred)

print("\n----- LSTM Model Evaluation -----")
print("MAE:", mean_absolute_error(y_test_inv, y_pred_inv))
print("MSE:", mean_squared_error(y_test_inv, y_pred_inv))
print("RMSE:", np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
print("R² Score:", r2_score(y_test_inv, y_pred_inv))

# Vẽ biểu đồ loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Vẽ biểu đồ MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

# --- 5. Dự đoán 2025–2029 ---
print("\nGenerating predictions for 2025-2029...")
future_years = [2025, 2026, 2027, 2028, 2029]
unique_combinations = df_model[["country_name", "sex", "age_group"]].drop_duplicates()

def predict_future_improved(model, last_sequence, time_steps, future_steps, scaler_year, scaler_rate, start_year=2024, ensemble_size=10, noise_level=0.01):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(future_steps):
        # Tạo ensemble predictions với nhiều noise levels
        ensemble_preds = []
        for _ in range(ensemble_size):
            noisy_sequence = current_sequence + np.random.normal(0, noise_level, current_sequence.shape)
            pred = model.predict(noisy_sequence[np.newaxis, :, :], verbose=0)[0, 0]
            ensemble_preds.append(pred)
        
        # Lấy trung bình của ensemble
        pred_mean_scaled = np.mean(ensemble_preds)
        pred_mean = scaler_rate.inverse_transform([[pred_mean_scaled]])[0, 0]
        predictions.append(pred_mean)
        
        # Cập nhật sequence
        new_row = current_sequence[-1].copy()
        next_year = start_year + _ + 1
        next_year_scaled = scaler_year.transform([[next_year]])[0, 0]
        new_row[-2] = next_year_scaled
        new_row[-1] = pred_mean_scaled
        current_sequence = np.vstack((current_sequence[1:], new_row))
    
    return predictions

# Tạo dự đoán
future_predictions = []
groups = df_model.groupby(["country_name", "sex", "age_group"])

for _, group in groups:
    group = group.sort_values("year").tail(time_steps)
    if len(group) < time_steps:
        continue
    last_sequence = group[["country_name", "sex", "age_group", "year_scaled", "unemployment_rate_scaled"]].values
    preds = predict_future_improved(
        model, last_sequence, time_steps, len(future_years),
        scaler_year, scaler_rate, start_year=2024
    )
    future_predictions.extend(preds)

# Tạo DataFrame kết quả
future_data = []
for i, year in enumerate(future_years * len(unique_combinations)):
    row_idx = i // len(future_years)
    row = unique_combinations.iloc[row_idx]
    year_scaled = scaler_year.transform([[future_years[i % len(future_years)]]])[0][0]
    
    # Chuyển đổi encoded values về giá trị gốc
    country = label_encoders['country_name'].inverse_transform([row['country_name']])[0]
    sex = label_encoders['sex'].inverse_transform([row['sex']])[0]
    age_group = label_encoders['age_group'].inverse_transform([row['age_group']])[0]
    
    future_data.append({
        "country_name": country,
        "sex": sex,
        "age_group": age_group,
        "year": future_years[i % len(future_years)],
        "year_scaled": year_scaled,
        "unemployment_rate_pred_scaled": future_predictions[i]
    })

future_df = pd.DataFrame(future_data)
future_df["unemployment_rate_pred"] = scaler_rate.inverse_transform(
    np.array(future_df["unemployment_rate_pred_scaled"]).reshape(-1, 1)
)

# --- 6. Vẽ biểu đồ kết quả ---
print("\nPlotting results...")
historical_avg = df_model.groupby("year")["unemployment_rate"].mean().reset_index()
future_avg = future_df.groupby("year")["unemployment_rate_pred"].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(historical_avg["year"], historical_avg["unemployment_rate"], 
         "o-", label="Historical Data", linewidth=2)
plt.plot(future_avg["year"], future_avg["unemployment_rate_pred"], 
         "s--", label="LSTM Predictions", linewidth=2)

plt.fill_between(future_avg["year"], 
                 future_avg["unemployment_rate_pred"] * 0.9,
                 future_avg["unemployment_rate_pred"] * 1.1,
                 alpha=0.2, label="Prediction Interval")

plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.title("Global Average Unemployment Rate Prediction (LSTM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Lưu kết quả dự đoán
print("\nSaving predictions...")
future_df.to_csv("lstm_predictions.csv", index=False)
print("Predictions saved to lstm_predictions.csv")

# Lưu mô hình, scaler, encoder
model.save('best_lstm_model.h5')
joblib.dump(scaler_rate, 'scaler_rate.save')
joblib.dump(scaler_year, 'scaler_year.save')
joblib.dump(label_encoders, 'label_encoders.save')
