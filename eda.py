import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Thiết lập font chữ cho matplotlib
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

class UnemploymentEDA:
    def __init__(self, file_path="global_unemployment_data.csv"):
        self.file_path = file_path
        self.df = None
        self.df_long = None
        
    def load_data(self):
        """Load và chuyển đổi dữ liệu từ wide format sang long format"""
        try:
            # Đọc dữ liệu
            self.df = pd.read_csv(self.file_path)
            
            # Chuyển từ wide -> long format
            self.df_long = self.df.melt(
                id_vars=["country_name", "sex", "age_group", "age_categories"],
                value_vars=[str(year) for year in range(2014, 2025)],
                var_name="year",
                value_name="unemployment_rate"
            )
            self.df_long["year"] = self.df_long["year"].astype(int)
            
            print("Data loaded successfully!")
            print(f"\nShape of data: {self.df.shape}")
            print("\nFirst few rows:")
            print(self.df.head())
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def basic_statistics(self):
        """Tính toán các thống kê cơ bản"""
        if self.df_long is None:
            print("Please load data first")
            return
            
        print("\n=== Basic Statistics ===")
        
        # Thống kê tổng quan
        print("\nOverall statistics:")
        print(self.df_long["unemployment_rate"].describe())
        
        # Thống kê theo quốc gia
        print("\nStatistics by country:")
        country_stats = self.df_long.groupby("country_name")["unemployment_rate"].agg(
            ['mean', 'std', 'min', 'max']
        ).round(2)
        print(country_stats)
        
        # Thống kê theo giới tính
        print("\nStatistics by sex:")
        sex_stats = self.df_long.groupby("sex")["unemployment_rate"].agg(
            ['mean', 'std', 'min', 'max']
        ).round(2)
        print(sex_stats)
        
        # Thống kê theo nhóm tuổi
        print("\nStatistics by age group:")
        age_stats = self.df_long.groupby("age_group")["unemployment_rate"].agg(
            ['mean', 'std', 'min', 'max']
        ).round(2)
        print(age_stats)
    
    def plot_unemployment_trends(self):
        """Vẽ biểu đồ xu hướng thất nghiệp theo thời gian"""
        if self.df_long is None:
            print("Please load data first")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Tính trung bình tỷ lệ thất nghiệp theo năm
        yearly_avg = self.df_long.groupby("year")["unemployment_rate"].mean()
        
        # Vẽ biểu đồ đường
        plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2)
        
        plt.title("Global Unemployment Rate Trends (2014-2024)")
        plt.xlabel("Year")
        plt.ylabel("Unemployment Rate (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(yearly_avg.index)
        
        plt.tight_layout()
        plt.savefig("unemployment_trends.png")
        plt.close()
    
    def plot_country_comparison(self, top_n=10):
        """Vẽ biểu đồ so sánh tỷ lệ thất nghiệp giữa các quốc gia"""
        if self.df_long is None:
            print("Please load data first")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Tính trung bình tỷ lệ thất nghiệp theo quốc gia
        country_avg = self.df_long.groupby("country_name")["unemployment_rate"].mean()
        
        # Lấy top N quốc gia có tỷ lệ thất nghiệp cao nhất
        top_countries = country_avg.nlargest(top_n)
        
        # Vẽ biểu đồ cột
        plt.barh(top_countries.index, top_countries.values)
        
        plt.title(f"Top {top_n} Countries with Highest Average Unemployment Rate")
        plt.xlabel("Average Unemployment Rate (%)")
        plt.ylabel("Country")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("country_comparison.png")
        plt.close()
    
    def plot_age_group_distribution(self):
        """Vẽ biểu đồ phân phối tỷ lệ thất nghiệp theo nhóm tuổi"""
        if self.df_long is None:
            print("Please load data first")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Vẽ boxplot
        sns.boxplot(x="age_group", y="unemployment_rate", data=self.df_long)
        
        plt.title("Unemployment Rate Distribution by Age Group")
        plt.xlabel("Age Group")
        plt.ylabel("Unemployment Rate (%)")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig("age_group_distribution.png")
        plt.close()
    
    def plot_sex_comparison(self):
        """Vẽ biểu đồ so sánh tỷ lệ thất nghiệp theo giới tính"""
        if self.df_long is None:
            print("Please load data first")
            return
            
        plt.figure(figsize=(8, 6))
        
        # Tính trung bình tỷ lệ thất nghiệp theo giới tính
        sex_avg = self.df_long.groupby("sex")["unemployment_rate"].mean()
        
        # Vẽ biểu đồ cột
        plt.bar(sex_avg.index, sex_avg.values)
        
        plt.title("Average Unemployment Rate by Sex")
        plt.xlabel("Sex")
        plt.ylabel("Average Unemployment Rate (%)")
        
        plt.tight_layout()
        plt.savefig("sex_comparison.png")
        plt.close()
    
    def run_all_analysis(self):
        """Chạy tất cả các phân tích"""
        if self.load_data():
            self.basic_statistics()
            self.plot_unemployment_trends()
            self.plot_country_comparison()
            self.plot_age_group_distribution()
            self.plot_sex_comparison()
            print("\nAll analysis completed! Check the generated plots in the current directory.")

# Example usage
if __name__ == "__main__":
    eda = UnemploymentEDA()
    eda.run_all_analysis() 