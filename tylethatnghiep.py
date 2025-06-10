import pandas as pd
import numpy as np

# Đọc dữ liệu từ file Excel
file_path = "C:/Users/nguye/OneDrive/Máy tính/DNUuni/Do_an_tot_nghiep/Tylethatnghiep.xlsx"
xls = pd.ExcelFile(file_path)
df_table1 = pd.read_excel(xls, sheet_name="Table 1")

df_unemployment = df_table1.set_index("Country").transpose()