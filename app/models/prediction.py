import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
from app.extensions import db

class UnemploymentPredictor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def load_and_preprocess_data(self, file_path="global_unemployment_data.csv"):
        """Load và tiền xử lý dữ liệu"""
        try:
            # Đọc dữ liệu
            print("Loading data from:", file_path)
            df = pd.read_csv(file_path)
            print("Data shape:", df.shape)
            print("Available countries:", df['country_name'].unique())
            
            # Chuyển từ wide -> long format
            df_long = df.melt(
                id_vars=["country_name", "sex", "age_group", "age_categories"],
                value_vars=[str(year) for year in range(2014, 2025)],
                var_name="year",
                value_name="unemployment_rate"
            )
            df_long["year"] = df_long["year"].astype(int)
            
            # Xử lý missing values
            df_long = df_long.dropna(subset=["unemployment_rate"])
            
            # Mã hóa các biến phân loại
            categorical_cols = ["country_name", "sex", "age_group"]
            for col in categorical_cols:
                le = LabelEncoder()
                # Fit the encoder with all possible values
                le.fit(df_long[col].unique())
                # Transform the data
                df_long[col] = le.transform(df_long[col])
                self.label_encoders[col] = le
                print(f"Encoded {col} values:", list(le.classes_))
            
            # Chuẩn bị features và target
            X = df_long[["country_name", "sex", "age_group", "year"]]
            y = df_long["unemployment_rate"]
            
            # Chia tập train/test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Chuẩn hóa dữ liệu
            self.scaler.fit(self.X_train)
            self.X_train_scaled = self.scaler.transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            print("Data preprocessing completed successfully")
            return True
            
        except Exception as e:
            print(f"Error in data loading and preprocessing: {str(e)}")
            return False
    
    def get_available_countries(self):
        """Lấy danh sách các quốc gia có sẵn"""
        if 'country_name' in self.label_encoders:
            return list(self.label_encoders['country_name'].classes_)
        return []
    
    def get_available_sexes(self):
        """Lấy danh sách các giới tính có sẵn"""
        if 'sex' in self.label_encoders:
            return list(self.label_encoders['sex'].classes_)
        return []
    
    def get_available_age_groups(self):
        """Lấy danh sách các nhóm tuổi có sẵn"""
        if 'age_group' in self.label_encoders:
            return list(self.label_encoders['age_group'].classes_)
        return []

    def save_model(self, model_dir="models"):
        """Lưu mô hình và các encoder"""
        try:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(model_dir, exist_ok=True)
            
            # Lưu scaler
            scaler_path = os.path.join(model_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            # Lưu label encoders
            for name, encoder in self.label_encoders.items():
                encoder_path = os.path.join(model_dir, f"{name}_encoder.joblib")
                joblib.dump(encoder, encoder_path)
            
            print("Model components saved successfully")
            return True
        except Exception as e:
            print(f"Error saving model components: {str(e)}")
            return False

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    country = db.Column(db.String(100), nullable=False)
    sex = db.Column(db.String(50), nullable=False)
    age_group = db.Column(db.String(50), nullable=False)
    
    # Historical values (2014-2024)
    value_2014 = db.Column(db.Float)
    value_2015 = db.Column(db.Float)
    value_2016 = db.Column(db.Float)
    value_2017 = db.Column(db.Float)
    value_2018 = db.Column(db.Float)
    value_2019 = db.Column(db.Float)
    value_2020 = db.Column(db.Float)
    value_2021 = db.Column(db.Float)
    value_2022 = db.Column(db.Float)
    value_2023 = db.Column(db.Float)
    value_2024 = db.Column(db.Float)
    
    # Predicted values (2025-2029)
    prediction_2025 = db.Column(db.Float)
    prediction_2026 = db.Column(db.Float)
    prediction_2027 = db.Column(db.Float)
    prediction_2028 = db.Column(db.Float)
    prediction_2029 = db.Column(db.Float)
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.country} - {self.timestamp}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'country': self.country,
            'sex': self.sex,
            'age_group': self.age_group,
            'historical_values': {
                '2014': self.value_2014,
                '2015': self.value_2015,
                '2016': self.value_2016,
                '2017': self.value_2017,
                '2018': self.value_2018,
                '2019': self.value_2019,
                '2020': self.value_2020,
                '2021': self.value_2021,
                '2022': self.value_2022,
                '2023': self.value_2023,
                '2024': self.value_2024
            },
            'predictions': {
                '2025': self.prediction_2025,
                '2026': self.prediction_2026,
                '2027': self.prediction_2027,
                '2028': self.prediction_2028,
                '2029': self.prediction_2029
            }
        }

# Khởi tạo predictor
predictor = UnemploymentPredictor()

# Load và tiền xử lý dữ liệu
predictor.load_and_preprocess_data()

# Lưu mô hình
predictor.save_model() 