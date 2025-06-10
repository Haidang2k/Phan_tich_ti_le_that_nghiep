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
