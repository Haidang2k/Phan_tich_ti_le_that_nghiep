import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os
import seaborn as sns

# Kiểm tra và import xgboost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost is not installed. Please install it using: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Cấu hình matplotlib để hỗ trợ tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class UnemploymentPredictor:
    def __init__(self):
        # Định nghĩa các tham số cần tối ưu cho từng mô hình
        self.model_params = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                }
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            }
        }
        
        # Thêm XGBoost nếu có sẵn
        if XGBOOST_AVAILABLE:
            self.model_params['XGBoost'] = {
                'model': XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 7],
                    'learning_rate': [0.1, 0.2],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9]
                }
            }
        
        self.models = {}
        self.best_params = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def load_and_preprocess_data(self, file_path="global_unemployment_data.csv"):
        """Load và tiền xử lý dữ liệu"""
        try:
            # Đọc dữ liệu
            df = pd.read_csv(file_path)
            
            # Chuyển từ wide -> long format
            df_long = df.melt(
                id_vars=["country_name", "sex", "age_group", "age_categories"],
                value_vars=[str(year) for year in range(2014, 2025)],
                var_name="year",
                value_name="unemployment_rate"
            )
            df_long["year"] = df_long["year"].astype(int)
            
            # Tạo boxplot trước khi xử lý
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_long, x='year', y='unemployment_rate')
            plt.title('Unemployment Rate Distribution by Year (Before Processing)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('boxplot_before_processing.png')
            plt.show()
            plt.close()
            
            # Xử lý missing values
            df_long = df_long.dropna(subset=["unemployment_rate"])
            
            # Mã hóa các biến phân loại
            categorical_cols = ["country_name", "sex", "age_group"]
            for col in categorical_cols:
                le = LabelEncoder()
                df_long[col] = le.fit_transform(df_long[col])
                self.label_encoders[col] = le
            
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
            
            # Tạo boxplot sau khi xử lý
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_long, x='year', y='unemployment_rate')
            plt.title('Unemployment Rate Distribution by Year (After Processing)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('boxplot_after_processing.png')
            plt.show()
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error in data loading and preprocessing: {str(e)}")
            return False
    
    def train_models(self):
        """Huấn luyện và tối ưu các mô hình"""
        if not hasattr(self, 'X_train_scaled'):
            print("Please load and preprocess data first")
            return False
            
        try:
            self.model_predictions = {}
            self.model_metrics = {}
            
            for name, model_info in self.model_params.items():
                print(f"\n----- Optimizing {name} -----")
                
                # Tạo GridSearchCV
                grid_search = GridSearchCV(
                    estimator=model_info['model'],
                    param_grid=model_info['params'],
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Tìm tham số tối ưu
                grid_search.fit(self.X_train_scaled, self.y_train)
                
                # Lưu mô hình tốt nhất và các tham số
                self.models[name] = grid_search.best_estimator_
                self.best_params[name] = grid_search.best_params_
                
                # Dự đoán trên tập test
                y_pred = self.models[name].predict(self.X_test_scaled)
                
                # Tính toán các metrics
                metrics = {
                    'MAE': mean_absolute_error(self.y_test, y_pred),
                    'MSE': mean_squared_error(self.y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                    'R2': r2_score(self.y_test, y_pred)
                }
                
                self.model_predictions[name] = y_pred
                self.model_metrics[name] = metrics
                
                print(f"\nBest parameters for {name}:")
                for param, value in self.best_params[name].items():
                    print(f"{param}: {value}")
                
                print(f"\nMetrics for {name}:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
                
                # Phân tích feature importance nếu mô hình là Random Forest hoặc XGBoost
                if name in ['Random Forest', 'XGBoost']:
                    self.analyze_feature_importance(self.models[name], name)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            return False
    
    def analyze_feature_importance(self, model, model_name):
        """Phân tích và hiển thị tầm quan trọng của các đặc trưng cho một mô hình cụ thể"""
        try:
            # Lấy feature importance từ mô hình
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'get_booster'): # Dành cho XGBoost
                 feature_importance = model.get_booster().get_score(importance_type='weight')
                 # Chuyển dictionary của XGBoost sang format phù hợp
                 feature_names = list(feature_importance.keys())
                 importance_values = list(feature_importance.values())
                 importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance_values
                 })
                 # Sắp xếp theo tầm quan trọng giảm dần
                 importance_df = importance_df.sort_values('Importance', ascending=False)
                 # Đối với XGBoost, 'weight' importance chỉ đếm số lần feature được sử dụng. Có thể cần các loại khác.
                 print(f"\nNote: XGBoost Feature Importance ('weight' type) shows how many times a feature is used in the trees.")
            else:
                print(f"Feature importance not available for {model_name}")
                return
            
            if 'importance_df' not in locals(): # Tạo DataFrame nếu chưa được tạo từ XGBoost
                # Tạo DataFrame để dễ dàng hiển thị
                feature_names = self.X_train.columns
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                })
                
                # Sắp xếp theo tầm quan trọng giảm dần
                importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # In kết quả
            print(f"\nFeature Importance ({model_name}):")
            print(importance_df)
            
            # Vẽ biểu đồ
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title(f'Feature Importance in Unemployment Rate Prediction ({model_name})')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error in {model_name} feature importance analysis: {str(e)}")
    
    def predict_future(self, country, sex, age_group, years):
        """Dự đoán tỷ lệ thất nghiệp cho tương lai"""
        if not self.is_trained:
            print("Models are not trained yet")
            return None
            
        try:
            # Mã hóa các giá trị đầu vào
            country_encoded = self.label_encoders['country_name'].transform([country])[0]
            sex_encoded = self.label_encoders['sex'].transform([sex])[0]
            age_group_encoded = self.label_encoders['age_group'].transform([age_group])[0]
            
            # Tạo dữ liệu dự đoán
            future_data = []
            for year in years:
                future_data.append([country_encoded, sex_encoded, age_group_encoded, year])
            
            future_df = pd.DataFrame(future_data, columns=["country_name", "sex", "age_group", "year"])
            future_scaled = self.scaler.transform(future_df)
            
            # Dự đoán cho từng mô hình
            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(future_scaled)
                predictions[name] = pred.tolist()
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
    
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
    
    def save_models(self, directory="models"):
        """Lưu các mô hình và encoders"""
        if not self.is_trained:
            print("Models are not trained yet")
            return False
            
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Lưu các mô hình
            for name, model in self.models.items():
                joblib.dump(model, os.path.join(directory, f"{name.lower().replace(' ', '_')}.joblib"))
            
            # Lưu các encoders và scaler
            joblib.dump(self.label_encoders, os.path.join(directory, "label_encoders.joblib"))
            joblib.dump(self.scaler, os.path.join(directory, "scaler.joblib"))
            
            return True
            
        except Exception as e:
            print(f"Error in saving models: {str(e)}")
            return False
    
    def load_models(self, directory="models"):
        """Tải các mô hình và encoders"""
        try:
            # Tải các mô hình
            for name in self.models.keys():
                model_path = os.path.join(directory, f"{name.lower().replace(' ', '_')}.joblib")
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
            
            # Tải các encoders và scaler
            encoders_path = os.path.join(directory, "label_encoders.joblib")
            scaler_path = os.path.join(directory, "scaler.joblib")
            
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error in loading models: {str(e)}")
            return False

if __name__ == "__main__":
    # Tạo instance của UnemploymentPredictor
    predictor = UnemploymentPredictor()
    
    # Load và tiền xử lý dữ liệu
    if predictor.load_and_preprocess_data():
        print("Data loaded and preprocessed successfully!")
        
        # Huấn luyện mô hình
        if predictor.train_models():
            print("\nModels trained successfully!")
            
            # Lấy danh sách các quốc gia, giới tính và nhóm tuổi có sẵn
            print("\nAvailable countries:", predictor.get_available_countries())
            print("Available sexes:", predictor.get_available_sexes())
            print("Available age groups:", predictor.get_available_age_groups())
            
            # Lưu mô hình
            if predictor.save_models():
                print("\nModels saved successfully!")
        else:
            print("Failed to train models!")
    else:
        print("Failed to load and preprocess data!")