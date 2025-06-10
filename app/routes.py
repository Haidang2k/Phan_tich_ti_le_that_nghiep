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
import joblib
import matplotlib.ticker as mticker

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

# Load scaler, encoder, model khi khởi tạo app
scaler_rate = joblib.load('scaler_rate.save')
scaler_year = joblib.load('scaler_year.save')
label_encoders = joblib.load('label_encoders.save')
model = load_model('best_lstm_model.h5')

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

@main_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        country = data.get('country')
        sex = data.get('sex')
        age_group = data.get('age_group')
        # Encode input
        country_encoded = label_encoders['country_name'].transform([country])[0]
        sex_encoded = label_encoders['sex'].transform([sex])[0]
        age_group_encoded = label_encoders['age_group'].transform([age_group])[0]
        # Lấy dữ liệu lịch sử
        df = pd.read_csv("global_unemployment_data.csv")
        df_filtered = df[(df['country_name'] == country) & (df['sex'] == sex) & (df['age_group'] == age_group)]
        historical_years = [str(year) for year in range(2014, 2025)]
        historical_rates = df_filtered[historical_years].values.flatten()
        # Chuẩn hóa dữ liệu lịch sử
        rates_scaled = scaler_rate.transform(historical_rates.reshape(-1, 1)).flatten()
        years_scaled = scaler_year.transform(np.array(range(2014, 2025)).reshape(-1, 1)).flatten()
        # Tạo chuỗi đầu vào
        time_steps = 5
        last_seq = []
        for i in range(-time_steps, 0):
            last_seq.append([
                country_encoded,
                sex_encoded,
                age_group_encoded,
                years_scaled[i],
                rates_scaled[i]
            ])
        last_seq = np.array(last_seq)
        # Hàm dự đoán multi-step tối ưu
        def predict_future_lstm(model, last_sequence, time_steps, future_steps, scaler_year, scaler_rate, start_year=2024, ensemble_size=10, noise_level=0.01):
            predictions = []
            current_sequence = last_sequence.copy()
            for i in range(future_steps):
                ensemble_preds = []
                for _ in range(ensemble_size):
                    noisy_seq = current_sequence + np.random.normal(0, noise_level, current_sequence.shape)
                    pred = model.predict(noisy_seq[np.newaxis, :, :], verbose=0)[0, 0]
                    ensemble_preds.append(pred)
                pred_mean_scaled = np.mean(ensemble_preds)
                pred_mean = scaler_rate.inverse_transform([[pred_mean_scaled]])[0, 0]
                predictions.append(pred_mean)
                # Cập nhật chuỗi cho năm tiếp theo
                new_row = current_sequence[-1].copy()
                next_year = start_year + i + 1
                next_year_scaled = scaler_year.transform([[next_year]])[0, 0]
                new_row[-2] = next_year_scaled
                new_row[-1] = pred_mean_scaled
                current_sequence = np.vstack((current_sequence[1:], new_row))
            return predictions

        # Dự đoán cho 5 năm tới
        future_predictions = predict_future_lstm(model, last_seq, time_steps, 5, scaler_year, scaler_rate)
        
        # Tạo plot
        plt.figure(figsize=(10, 6))
        years = list(range(2014, 2025)) + list(range(2025, 2030))
        rates = historical_rates.tolist() + future_predictions
        
        # Plot historical data
        plt.plot(years[:11], rates[:11], 'b-', label='Historical Data')
        
        # Plot predictions
        plt.plot(years[10:], rates[10:], 'r--', label='Predictions')
        
        # Add confidence interval
        std_dev = np.std(future_predictions)
        plt.fill_between(years[10:], 
                        [r - 1.96*std_dev for r in rates[10:]],
                        [r + 1.96*std_dev for r in rates[10:]],
                        color='r', alpha=0.2)
        
        plt.title(f'Unemployment Rate Prediction for {country} - {sex} - {age_group}')
        plt.xlabel('Year')
        plt.ylabel('Unemployment Rate (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
        
        # Create plot URL
        plot_url = create_plot()
        
        return jsonify({
            'plot': plot_url,
            'predictions': future_predictions,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@main_bp.route('/explore')
def explore():
    """Explore page"""
    try:
        # Get available options for exploration
        countries = [country for country in label_encoders['country_name'].classes_]
        sexes = [sex for sex in label_encoders['sex'].classes_]
        age_groups = [age_group for age_group in label_encoders['age_group'].classes_]
        
        return render_template('explore.html',
                             countries=countries,
                             sexes=sexes,
                             age_groups=age_groups)
                             
    except Exception as e:
        flash(f"Error loading explore page: {str(e)}", "error")
        return render_template('explore.html')

@main_bp.route('/api/explore', methods=['POST'])
def explore_data():
    try:
        data = request.get_json()
        country = data.get('country')
        sex = data.get('sex')
        age_group = data.get('age_group')
        
        # Lấy dữ liệu
        df = pd.read_csv("global_unemployment_data.csv")
        df_filtered = df[(df['country_name'] == country) & 
                        (df['sex'] == sex) & 
                        (df['age_group'] == age_group)]
        
        # Tạo các biểu đồ
        historical_years = [str(year) for year in range(2014, 2025)]
        historical_rates = df_filtered[historical_years].values.flatten()
        
        # 1. Line plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(2014, 2025), historical_rates, 'b-', marker='o')
        plt.title(f'Unemployment Rate Trend for {country} - {sex} - {age_group}')
        plt.xlabel('Year')
        plt.ylabel('Unemployment Rate (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
        line_plot_url = create_plot()
        
        # 2. Box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_filtered[historical_years])
        plt.title(f'Unemployment Rate Distribution for {country} - {sex} - {age_group}')
        plt.xlabel('Year')
        plt.ylabel('Unemployment Rate (%)')
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
        box_plot_url = create_plot()
        
        # 3. Statistics
        stats = {
            'mean': float(np.mean(historical_rates)),
            'median': float(np.median(historical_rates)),
            'std': float(np.std(historical_rates)),
            'min': float(np.min(historical_rates)),
            'max': float(np.max(historical_rates))
        }
        
        return jsonify({
            'line_plot': line_plot_url,
            'box_plot': box_plot_url,
            'stats': stats,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@main_bp.route('/api/export-csv', methods=['POST'])
def export_csv():
    try:
        data = request.get_json()
        country = data.get('country')
        sex = data.get('sex')
        age_group = data.get('age_group')
        
        # Lấy dữ liệu
        df = pd.read_csv("global_unemployment_data.csv")
        df_filtered = df[(df['country_name'] == country) & 
                        (df['sex'] == sex) & 
                        (df['age_group'] == age_group)]
        
        # Tạo file CSV
        output = io.StringIO()
        df_filtered.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'unemployment_data_{country}_{sex}_{age_group}.csv'
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@main_bp.route('/quick-report')
def quick_report():
    """Quick report page"""
    try:
        # Get available options for report
        countries = [country for country in label_encoders['country_name'].classes_]
        sexes = [sex for sex in label_encoders['sex'].classes_]
        age_groups = [age_group for age_group in label_encoders['age_group'].classes_]
        
        return render_template('quick_report.html',
                             countries=countries,
                             sexes=sexes,
                             age_groups=age_groups)
                             
    except Exception as e:
        flash(f"Error loading quick report page: {str(e)}", "error")
        return render_template('quick_report.html')

@main_bp.route('/api/quick-report', methods=['POST'])
def generate_quick_report():
    try:
        data = request.get_json()
        country = data.get('country')
        sex = data.get('sex')
        age_group = data.get('age_group')
        
        # Lấy dữ liệu
        df = pd.read_csv("global_unemployment_data.csv")
        df_filtered = df[(df['country_name'] == country) & 
                        (df['sex'] == sex) & 
                        (df['age_group'] == age_group)]
        
        # Tạo các biểu đồ
        historical_years = [str(year) for year in range(2014, 2025)]
        historical_rates = df_filtered[historical_years].values.flatten()
        
        # 1. Line plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(2014, 2025), historical_rates, 'b-', marker='o')
        plt.title(f'Unemployment Rate Trend for {country} - {sex} - {age_group}')
        plt.xlabel('Year')
        plt.ylabel('Unemployment Rate (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
        line_plot_url = create_plot()
        
        # 2. Box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_filtered[historical_years])
        plt.title(f'Unemployment Rate Distribution for {country} - {sex} - {age_group}')
        plt.xlabel('Year')
        plt.ylabel('Unemployment Rate (%)')
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
        box_plot_url = create_plot()
        
        # 3. Statistics
        stats = {
            'mean': float(np.mean(historical_rates)),
            'median': float(np.median(historical_rates)),
            'std': float(np.std(historical_rates)),
            'min': float(np.min(historical_rates)),
            'max': float(np.max(historical_rates))
        }
        
        return jsonify({
            'line_plot': line_plot_url,
            'box_plot': box_plot_url,
            'stats': stats,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@main_bp.route('/api/export-report', methods=['POST'])
def export_report():
    try:
        data = request.get_json()
        country = data.get('country')
        sex = data.get('sex')
        age_group = data.get('age_group')
        line_plot = data.get('line_plot')
        box_plot = data.get('box_plot')
        stats = data.get('stats')
        
        # Tạo PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        elements.append(Paragraph(f'Unemployment Rate Report for {country}', title_style))
        elements.append(Paragraph(f'Sex: {sex}', styles['Normal']))
        elements.append(Paragraph(f'Age Group: {age_group}', styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Line Plot
        elements.append(Paragraph('Historical Trend', styles['Heading2']))
        img_data = base64.b64decode(line_plot)
        img = Image(BytesIO(img_data), width=400, height=300)
        elements.append(img)
        elements.append(Spacer(1, 20))
        
        # Box Plot
        elements.append(Paragraph('Distribution Analysis', styles['Heading2']))
        img_data = base64.b64decode(box_plot)
        img = Image(BytesIO(img_data), width=400, height=300)
        elements.append(img)
        elements.append(Spacer(1, 20))
        
        # Statistics
        elements.append(Paragraph('Statistical Summary', styles['Heading2']))
        stats_data = [
            ['Metric', 'Value'],
            ['Mean', f"{stats['mean']:.2f}%"],
            ['Median', f"{stats['median']:.2f}%"],
            ['Standard Deviation', f"{stats['std']:.2f}%"],
            ['Minimum', f"{stats['min']:.2f}%"],
            ['Maximum', f"{stats['max']:.2f}%"]
        ]
        table = Table(stats_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'unemployment_report_{country}_{sex}_{age_group}.pdf'
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@main_bp.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(current_app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon') 