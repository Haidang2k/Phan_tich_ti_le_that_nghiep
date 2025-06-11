from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, send_file, send_from_directory, current_app
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
import io
import base64
from tensorflow.keras.models import load_model
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
from matplotlib.patches import Patch
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import json
from datetime import datetime
from io import BytesIO
import sys
from functools import lru_cache
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Set console output encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

main_bp = Blueprint('main', __name__)

# Khởi tạo các biến toàn cục cho model và scaler
model = None
label_encoders = {}
scaler_rate = MinMaxScaler()
scaler_year = MinMaxScaler()
df_cache = None  # Cache cho DataFrame

@lru_cache(maxsize=1)
def load_data():
    """Load and cache data"""
    return pd.read_csv("global_unemployment_data.csv")

def init_models():
    """Khởi tạo mô hình LSTM và các encoder"""
    global model, label_encoders, scaler_rate, scaler_year, df_cache
    try:
        print("Initializing model...")
        
        # Đọc và cache dữ liệu
        df_cache = load_data()
        
        # Chuyển đổi dữ liệu sang định dạng long
        df_long = df_cache.melt(
            id_vars=["country_name", "sex", "age_group", "age_categories"],
            value_vars=[str(year) for year in range(2014, 2025)],
            var_name="year",
            value_name="unemployment_rate"
        )
        df_long["year"] = df_long["year"].astype(int)
        
        # Loại bỏ dữ liệu thiếu và xử lý outliers
        df_model = df_long.dropna(subset=["unemployment_rate"]).copy()
        Q1 = df_model["unemployment_rate"].quantile(0.25)
        Q3 = df_model["unemployment_rate"].quantile(0.75)
        IQR = Q3 - Q1
        df_model = df_model[
            (df_model["unemployment_rate"] >= Q1 - 1.5 * IQR) & 
            (df_model["unemployment_rate"] <= Q3 + 1.5 * IQR)
        ]
        
        # Khởi tạo và fit các encoder
        label_encoders = {}
        for col in ["country_name", "sex", "age_group"]:
            label_encoders[col] = LabelEncoder()
            df_model[col] = label_encoders[col].fit_transform(df_model[col])
        
        # Fit các scaler
        scaler_rate = MinMaxScaler()
        scaler_year = MinMaxScaler()
        df_model["unemployment_rate_scaled"] = scaler_rate.fit_transform(df_model[["unemployment_rate"]])
        df_model["year_scaled"] = scaler_year.fit_transform(df_model[["year"]])
        
        # Tải mô hình LSTM đã train
        model = load_model('best_lstm_model.h5')
        
        print("Model initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return False

# Khởi tạo mô hình khi module được load
try:
    init_models()
except Exception as e:
    print(f"Error during model initialization: {str(e)}")

def create_sequences(data, time_steps=5):
    """Tạo chuỗi dữ liệu cho LSTM"""
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

def create_plot():
    """Helper function to create plot URLs"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@main_bp.route('/')
@main_bp.route('/prediction')
def prediction():
    """Prediction page"""
    try:
        # Ensure model is initialized
        if model is None:
            if not init_models():
                flash("Failed to initialize model", "error")
                return render_template('prediction.html')
        
        # Get available options for prediction
        countries = [country for country in label_encoders['country_name'].classes_]
        sexes = [sex for sex in label_encoders['sex'].classes_]
        age_groups = [age_group for age_group in label_encoders['age_group'].classes_]
        
        return render_template('prediction.html',
                             countries=countries,
                             sexes=sexes,
                             age_groups=age_groups)
                             
    except Exception as e:
        flash(f"Error loading prediction page: {str(e)}", "error")
        return render_template('prediction.html')

@main_bp.route('/predict', methods=['POST']) #API dự đoán
def predict():
    global df_cache
    try:
        data = request.get_json()
        country = data.get('country')
        sex = data.get('sex')
        age_group = data.get('age_group')
        
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model has not been initialized'
            })
        
        # Sử dụng dữ liệu đã cache
        if df_cache is None:
            df_cache = load_data()
            
        df_filtered = df_cache[
            (df_cache['country_name'] == country) & 
            (df_cache['sex'] == sex) & 
            (df_cache['age_group'] == age_group)
        ]
        
        if df_filtered.empty:
            return jsonify({
                'success': False,
                'error': 'No data found for selected filters'
            })
        
        # Lấy dữ liệu lịch sử
        historical_years = [str(year) for year in range(2014, 2025)]
        historical_rates = df_filtered[historical_years].values.flatten()
        
        # Chuẩn bị dữ liệu cho dự đoán
        country_encoded = label_encoders['country_name'].transform([country])[0]
        sex_encoded = label_encoders['sex'].transform([sex])[0]
        age_group_encoded = label_encoders['age_group'].transform([age_group])[0]
        
        # Chuẩn hóa dữ liệu
        rates_scaled = scaler_rate.transform(historical_rates.reshape(-1, 1))
        years = np.array(range(2014, 2025)).reshape(-1, 1)
        years_scaled = scaler_year.transform(years)
        
        # Tạo chuỗi input cho LSTM (5 năm gần nhất)
        sequence_length = 5
        input_sequence = []
        for i in range(sequence_length):
            idx = -(sequence_length - i)
            input_sequence.append([
                country_encoded,
                sex_encoded,
                age_group_encoded,
                years_scaled[idx][0],
                rates_scaled[idx][0]
            ])
        input_sequence = np.array(input_sequence)
        
        def predict_future_optimized(last_sequence, time_steps, future_steps, num_samples=5):
            predictions = []
            confidence_intervals = []
            current_sequence = last_sequence.copy()
            
            for year_idx in range(future_steps):
                # Tạo ensemble predictions với ít mẫu hơn
                ensemble_preds = []
                for _ in range(num_samples):
                    pred = model.predict(current_sequence.reshape(1, time_steps, 5), verbose=0)
                    pred_original = scaler_rate.inverse_transform([[pred[0][0]]])[0][0]
                    ensemble_preds.append(pred_original)
                
                # Tính trung bình và độ lệch chuẩn của ensemble
                pred_mean = np.mean(ensemble_preds)
                pred_std = np.std(ensemble_preds)
                
                # Áp dụng các ràng buộc
                pred_mean = max(0, min(pred_mean, 100))
                
                # Thêm vào kết quả
                predictions.append(pred_mean)
                confidence_intervals.append(pred_std)
                
                # Cập nhật sequence cho lần dự đoán tiếp theo
                new_row = [
                    country_encoded,
                    sex_encoded,
                    age_group_encoded,
                    scaler_year.transform([[2025 + year_idx]])[0][0],
                    scaler_rate.transform([[pred_mean]])[0][0]
                ]
                current_sequence = np.vstack((current_sequence[1:], new_row))
            
            return predictions, confidence_intervals
        
        # Dự đoán cho 5 năm tiếp theo (2025-2029)
        predictions, confidence_intervals = predict_future_optimized(
            input_sequence, sequence_length, 5, num_samples=5
        )
        
        # Tạo biểu đồ
        plt.figure(figsize=(12, 6))
        years = list(range(2014, 2030))
        
        # Vẽ dữ liệu lịch sử
        plt.plot(years[:11], historical_rates, 'o-', label='Dữ liệu lịch sử', color='blue', linewidth=2)
        
        # Vẽ dự đoán và khoảng tin cậy
        pred_years = years[10:]
        pred_values = [historical_rates[-1]] + predictions
        confidence = np.array([0] + confidence_intervals)
        
        plt.plot(pred_years, pred_values, 'o--', label='Dự đoán', color='red', linewidth=2)
        plt.fill_between(pred_years,
                        np.array(pred_values) - 2*confidence,
                        np.array(pred_values) + 2*confidence,
                        color='red', alpha=0.1, label='Khoảng tin cậy (95%)')
        
        plt.title(f'Tỷ lệ thất nghiệp - {country}')
        plt.xlabel('Năm')
        plt.ylabel('Tỷ lệ (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Format trục x
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: int(x)))
        
        # Tạo URL cho biểu đồ
        chart_url = create_plot()
        
        # Tạo dữ liệu cho bảng
        table_data = {
            'Năm': years,
            'Tỷ lệ thất nghiệp (%)': list(historical_rates) + predictions,
            'Độ không chắc chắn (±%)': [0]*11 + confidence_intervals
        }
        
        return jsonify({
            'success': True,
            'chart_url': chart_url,
            'table_data': table_data,
            'predictions': predictions,
            'confidence_intervals': confidence_intervals
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@main_bp.route('/explore')
def explore():
    """Data exploration page"""
    try:
        # Load data
        df = pd.read_csv("global_unemployment_data.csv")
        
        # Get unique values for filters - sử dụng đúng tên cột
        countries = sorted(df['country_name'].unique().tolist())
        sexes = sorted(df['sex'].unique().tolist())
        age_groups = sorted(df['age_group'].unique().tolist())
        years = list(range(2014, 2025))
        
        return render_template('explore.html',
                             countries=countries,
                             sexes=sexes,
                             age_groups=age_groups,
                             years=years)
    except Exception as e:
        print(f"Error loading exploration page: {str(e)}")
        flash(f"Error loading exploration page: {str(e)}", "error")
        return render_template('explore.html')

@main_bp.route('/api/explore', methods=['POST']) #API khám phá dữ liệu
def explore_data():
    """Handle data exploration requests"""
    try:
        data = request.get_json()
        countries = data.get('countries', [])
        sex = data.get('sex')
        age_group = data.get('age_group')
        start_year = int(data.get('start_year', 2014))
        end_year = int(data.get('end_year', 2024))
        chart_type = data.get('chart_type', 'line')
        
        # Load and filter data
        df = pd.read_csv("global_unemployment_data.csv")
        
        # Convert year columns to string
        year_cols = [str(year) for year in range(start_year, end_year + 1)]
        
        # Filter data
        mask = df['country_name'].isin(countries)
        if sex:
            mask &= df['sex'] == sex
        if age_group:
            mask &= df['age_group'] == age_group
            
        filtered_df = df[mask]
        
        if filtered_df.empty:
            return jsonify({
                'success': False,
                'error': 'No data found for the selected filters'
            })
        
        # Create plot based on chart type
        plt.figure(figsize=(12, 6))
        
        if chart_type == 'line':
            for country in countries:
                country_data = filtered_df[filtered_df['country_name'] == country]
                if not country_data.empty:
                    # Convert year columns to numeric and handle any non-numeric values
                    values = pd.to_numeric(country_data[year_cols].iloc[0], errors='coerce')
                    plt.plot(year_cols, values, marker='o', label=country)
            
            plt.title('Tỷ lệ thất nghiệp theo thời gian')
            plt.xlabel('Năm')
            plt.ylabel('Tỷ lệ (%)')
            plt.xticks(rotation=45)
            
        elif chart_type == 'bar':
            # For bar chart, we'll show the latest year by default
            latest_year = str(end_year)
            country_values = []
            
            for country in countries:
                country_data = filtered_df[filtered_df['country_name'] == country]
                if not country_data.empty:
                    # Convert to numeric and handle any non-numeric values
                    value = pd.to_numeric(country_data[latest_year].iloc[0], errors='coerce')
                    country_values.append(float(value))
            
            plt.bar(countries, country_values)
            plt.title(f'Tỷ lệ thất nghiệp năm {latest_year}')
            plt.xlabel('Quốc gia')
            plt.ylabel('Tỷ lệ (%)')
            plt.xticks(rotation=45)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Create chart URL
        chart_url = create_plot()
        
        # Prepare data for table
        # Convert numeric columns to float and round to 2 decimal places
        numeric_cols = year_cols
        for col in numeric_cols:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').round(2)
        
        table_data = filtered_df[['country_name', 'sex', 'age_group'] + year_cols].to_dict('records')
        
        return jsonify({
            'success': True,
            'chart_url': chart_url,
            'table_data': table_data
        })
        
    except Exception as e:
        print(f"Error in data exploration: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@main_bp.route('/api/export-csv', methods=['POST']) #API xuất dữ liệu ra CSV
def export_csv():
    """Export filtered data to CSV"""
    try:
        data = request.get_json()
        countries = data.get('countries', [])
        sex = data.get('sex')
        age_group = data.get('age_group')
        start_year = int(data.get('start_year', 2014))
        end_year = int(data.get('end_year', 2024))
        
        # Load and filter data
        df = pd.read_csv("global_unemployment_data.csv")
        year_cols = [str(year) for year in range(start_year, end_year + 1)]
        
        # Filter data
        if countries:  # Nếu có chọn quốc gia
            mask = df['country_name'].isin(countries)
        else:  # Nếu không chọn quốc gia nào, lấy tất cả
            mask = pd.Series([True] * len(df))
            
        if sex and sex != "Tất cả":
            mask &= df['sex'] == sex
        if age_group and age_group != "Tất cả":
            mask &= df['age_group'] == age_group
            
        filtered_df = df[mask]
        
        if filtered_df.empty:
            return jsonify({
                'success': False,
                'error': 'No data found for the selected filters'
            })
        
        # Chọn các cột cần xuất
        export_cols = ['country_name', 'sex', 'age_group'] + year_cols
        
        # Chuyển đổi các cột số thành float và làm tròn
        for col in year_cols:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').round(2)
        
        # Tạo buffer để lưu CSV
        csv_buffer = io.StringIO()
        filtered_df[export_cols].to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_buffer.seek(0)
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'unemployment_data_{timestamp}.csv'
        
        # Chuyển đổi StringIO thành BytesIO để gửi file
        output = io.BytesIO()
        output.write(csv_buffer.getvalue().encode('utf-8-sig'))
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename,
            max_age=0
        )
        
    except Exception as e:
        print(f"Error exporting CSV: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@main_bp.route('/quick-report')
def quick_report():
    """Quick country report page"""
    try:
        # Load data
        df = pd.read_csv("global_unemployment_data.csv")
        
        # Get unique countries
        countries = sorted(df['country_name'].unique().tolist())
        
        return render_template('quick_report.html', countries=countries)
    except Exception as e:
        print(f"Error loading quick report page: {str(e)}")
        flash(f"Error loading quick report page: {str(e)}", "error")
        return render_template('quick_report.html')

@main_bp.route('/api/quick-report', methods=['POST']) #API tạo báo cáo nhanh
def generate_quick_report():
    """Generate quick report for a country"""
    try:
        data = request.get_json()
        country = data.get('country')
        
        if not country:
            return jsonify({
                'success': False,
                'error': 'Country is required'
            })
        
        # Load data
        df = pd.read_csv("global_unemployment_data.csv")
        year_cols = [str(year) for year in range(2014, 2025)]
        
        # Filter data for the selected country
        country_data = df[df['country_name'] == country]
        
        if country_data.empty:
            return jsonify({
                'success': False,
                'error': 'No data found for the selected country'
            })
        
        # Create trend chart
        plt.figure(figsize=(12, 6))
        
        # Plot lines for each demographic group
        for _, row in country_data.iterrows():
            label = f"{row['sex']} - {row['age_group']}"
            values = [float(row[year]) for year in year_cols]
            plt.plot(year_cols, values, marker='o', label=label)
        
        plt.title(f'Tỷ lệ thất nghiệp tại {country} (2014-2024)')
        plt.xlabel('Năm')
        plt.ylabel('Tỷ lệ (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Convert plot to base64 string
        trend_chart_url = create_plot()
        
        # Analyze most affected groups
        recent_years = [str(year) for year in range(2022, 2025)]
        recent_data = country_data[['sex', 'age_group'] + recent_years].copy()
        recent_data['average'] = recent_data[recent_years].astype(float).mean(axis=1)
        recent_data['max'] = recent_data[recent_years].astype(float).max(axis=1)
        
        most_affected = {
            'by_average': recent_data.loc[recent_data['average'].idxmax()].to_dict(),
            'by_max': recent_data.loc[recent_data['max'].idxmax()].to_dict()
        }
        
        # Analyze trends
        trends = {}
        for _, row in country_data.iterrows():
            values = [float(row[year]) for year in year_cols]
            
            # Calculate trend
            start_value = values[0]
            end_value = values[-1]
            diff = end_value - start_value
            
            # Calculate volatility
            std_dev = np.std(values)
            
            trend_info = {
                'direction': 'increase' if diff > 0 else 'decrease' if diff < 0 else 'stable',
                'magnitude': abs(diff),
                'volatility': std_dev,
                'values': values
            }
            
            trends[f"{row['sex']} - {row['age_group']}"] = trend_info
        
        return jsonify({
            'success': True,
            'trend_chart': trend_chart_url,
            'most_affected': most_affected,
            'trends': trends,
            'raw_data': country_data[['sex', 'age_group'] + year_cols].to_dict('records')
        })
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@main_bp.route('/api/export-report', methods=['POST']) #API xuất báo cáo ra PDF
def export_report():
    """Export report as PDF with Vietnamese font support"""
    try:
        data = request.get_json()
        country = data.get('country')
        chart_data = data.get('chart_data')
        analysis_data = data.get('analysis_data')
        
        if not all([country, chart_data, analysis_data]):
            return jsonify({
                'success': False,
                'error': 'Thiếu dữ liệu cần thiết để tạo báo cáo'
            })
        
        # Log dữ liệu đầu vào để debug
        print("Country:", country)
        print("Analysis data:", analysis_data)
        
        # Tạo buffer để lưu PDF
        buffer = BytesIO()
        
        # Tạo PDF document với kích thước A4
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Đăng ký font Unicode DejaVuSerif
        font_path = 'app/static/fonts/DejaVuSerif.ttf'
        if not os.path.exists(font_path):
            print(f"Font file not found: {font_path}")
        else:
            pdfmetrics.registerFont(TTFont('DejaVuSerif', font_path))
        # Lấy styles mặc định và tạo custom styles với font Unicode
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName='DejaVuSerif',
            fontSize=24,
            spaceAfter=30,
            alignment=1
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName='DejaVuSerif',
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12
        )
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontName='DejaVuSerif',
            fontSize=12,
            spaceBefore=6,
            spaceAfter=6
        )
        # Bắt đầu tạo nội dung PDF
        story = []
        # Thêm tiêu đề
        title_text = f'Báo cáo thất nghiệp - {country}'
        story.append(Paragraph(title_text, title_style))
        story.append(Spacer(1, 20))
        # Thêm biểu đồ vào PDF
        if chart_data:
            try:
                if 'base64,' in chart_data:
                    chart_data = chart_data.split('base64,')[1]
                img_data = BytesIO(base64.b64decode(chart_data))
                img = Image(img_data)
                img.drawHeight = 4*inch
                img.drawWidth = 7*inch
                story.append(img)
                story.append(Spacer(1, 20))
            except Exception as e:
                print(f"Error processing chart image: {str(e)}")
        # Thêm phần nhóm chịu ảnh hưởng nhiều nhất
        story.append(Paragraph('Nhóm chịu ảnh hưởng nhiều nhất', heading_style))
        if 'most_affected' in analysis_data and 'by_average' in analysis_data['most_affected']:
            avg_group = analysis_data['most_affected']['by_average']
            avg_text = f"""<b>Theo trung bình 3 năm gần đây:</b><br/>
                         {avg_group['sex']} - {avg_group['age_group']}<br/>
                         Trung bình: {float(avg_group['average']):.2f}%"""
            story.append(Paragraph(avg_text, body_style))
            story.append(Spacer(1, 10))
        if 'most_affected' in analysis_data and 'by_max' in analysis_data['most_affected']:
            max_group = analysis_data['most_affected']['by_max']
            max_text = f"""<b>Theo giá trị cao nhất:</b><br/>
                         {max_group['sex']} - {max_group['age_group']}<br/>
                         Cao nhất: {float(max_group['max']):.2f}%"""
            story.append(Paragraph(max_text, body_style))
        story.append(Spacer(1, 20))
        # Thêm phần phân tích xu hướng
        story.append(Paragraph('Phân tích xu hướng', heading_style))
        if 'trends' in analysis_data:
            for group, trend in analysis_data['trends'].items():
                direction_text = 'Tăng' if trend['direction'] == 'increase' else 'Giảm' if trend['direction'] == 'decrease' else 'Ổn định'
                magnitude = float(trend['magnitude']) if isinstance(trend['magnitude'], (int, float)) else 0.0
                volatility = float(trend['volatility']) if isinstance(trend['volatility'], (int, float)) else 0.0
                trend_text = f"""<b>{group}</b><br/>
                               {direction_text} {magnitude:.2f}%<br/>
                               Độ biến thiên: {volatility:.2f}"""
                story.append(Paragraph(trend_text, body_style))
                story.append(Spacer(1, 10))
        # Thêm timestamp
        story.append(Spacer(1, 30))
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontName='DejaVuSerif',
            fontSize=8,
            textColor=colors.gray,
            alignment=1
        )
        story.append(Paragraph(f"Báo cáo được tạo lúc: {timestamp}", footer_style))
        # Build PDF
        try:
            doc.build(story)
        except Exception as e:
            print(f"Error building PDF: {str(e)}")
            return jsonify({'success': False, 'error': f'Lỗi khi build PDF: {str(e)}'})
        buffer.seek(0)
        # Trả về file PDF
        response = send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'bao_cao_that_nghiep_{country}.pdf'
        )
        # Thêm header để tránh cache
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Lỗi khi tạo PDF: {str(e)}'
        })

@main_bp.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(current_app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon') 