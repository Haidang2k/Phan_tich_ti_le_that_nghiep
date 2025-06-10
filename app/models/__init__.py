import os
import sys

# Thêm thư mục gốc vào path để có thể import global.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .prediction import UnemploymentPredictor

# Khởi tạo predictor
predictor = UnemploymentPredictor()

# Load và tiền xử lý dữ liệu
predictor.load_and_preprocess_data()

# Lưu mô hình
predictor.save_model() 