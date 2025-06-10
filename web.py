import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(layout="wide")

# Đọc dữ liệu
@st.cache_data
def load_data():
    # Dữ liệu quốc gia
    file_path1 = "C:/Users/nguye/OneDrive/Máy tính/DNUuni/Do_an_tot_nghiep/Tylethatnghiep.xlsx"
    xls = pd.ExcelFile(file_path1)
    df_table1 = pd.read_excel(xls, sheet_name="Table 1")
    df_unemployment = df_table1.set_index("Country").transpose()
    
    # Dữ liệu dự đoán Việt Nam
    file_path2 = "WEO_Data.xlsx"
    df = pd.read_excel(file_path2, sheet_name="WEO_Data")
    unemp_df = df[
        (df["Country"] == "Vietnam") &
        (df["Subject Descriptor"].str.contains("Unemployment", case=False, na=False))
    ]
    year_cols = [col for col in unemp_df.columns if str(col).isdigit()]
    data = unemp_df[year_cols].T
    data.columns = ["Unemployment Rate"]
    data = data.dropna().reset_index()
    data.columns = ["Year", "Unemployment Rate"]
    data["Year"] = data["Year"].astype(int)
    
    return df_unemployment, data

df_unemployment, vietnam_data = load_data()

# Sidebar
st.sidebar.header('Chọn Quốc Gia')
top_countries = ['United States', 'Germany', 'China', 'Brazil', 'South Africa', 'Vietnam']
selected_countries = st.sidebar.multiselect('Chọn quốc gia để so sánh:', top_countries, default=top_countries)

# Tabs
tab1, tab2 = st.tabs(["Phân tích Quốc tế", "Dự đoán Việt Nam"])

with tab1:
    st.title('Phân tích Tỷ lệ Thất nghiệp Theo Quốc Gia')
    df_selected = df_unemployment[selected_countries]

    # Biểu đồ đường
    st.subheader('Biểu đồ Đường')
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for country in df_selected.columns:
        ax1.plot(df_selected.index, df_selected[country], marker='o', label=country)
    ax1.set_xlabel("Năm")
    ax1.set_ylabel("Tỷ lệ thất nghiệp (%)")
    ax1.set_title("Tỷ lệ thất nghiệp của một số quốc gia tiêu biểu")
    ax1.legend()
    ax1.grid()
    st.pyplot(fig1)

    # Biểu đồ khu vực
    st.subheader('Biểu đồ Khu Vực')
    regions = {
        'Châu Á': ['China', 'India', 'Japan', 'South Korea', 'Vietnam'],
        'Châu Âu': ['Germany', 'France', 'United Kingdom', 'Italy'],
        'Châu Mỹ': ['United States', 'Brazil', 'Argentina', 'Canada'],
        'Châu Phi': ['South Africa', 'Nigeria', 'Egypt', 'Kenya']
    }
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (region, countries) in enumerate(regions.items()):
        for country in countries:
            if country in df_unemployment.columns:
                axes[idx].plot(df_unemployment.index, df_unemployment[country], marker='o', label=country)
        axes[idx].set_title(region)
        axes[idx].legend()
        axes[idx].grid()
    plt.tight_layout()
    st.pyplot(fig2)

    # Heatmap
    st.subheader('Heatmap')
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_unemployment.T, cmap='coolwarm', linewidths=0.5, annot=False, ax=ax3)
    ax3.set_title("Heatmap tỷ lệ thất nghiệp theo quốc gia")
    ax3.set_xlabel("Năm")
    ax3.set_ylabel("Quốc gia")
    st.pyplot(fig3)

    # Top 10
    st.subheader('Top 10 Quốc gia')
    selected_year = st.selectbox('Chọn Năm', df_unemployment.index)
    top10 = df_unemployment.loc[selected_year].nlargest(10)
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.barh(top10.index, top10.values)
    ax4.set_xlabel("Tỷ lệ thất nghiệp (%)")
    ax4.set_ylabel("Quốc gia")
    ax4.set_title(f"Top 10 quốc gia có tỷ lệ thất nghiệp cao nhất năm {selected_year}")
    st.pyplot(fig4)