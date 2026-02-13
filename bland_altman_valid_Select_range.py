import pandas as pd

# ファイル読み込み
input_path = r"C:\Users\mpg\Desktop\python_rasio\Bland-Altman_analysis_spo2.xlsx"
df = pd.read_excel(input_path)

# SpO2の範囲（ここを変更すればOK）
spo2_max = 90   # 上限
spo2_min = 60   # 下限

# 指定範囲内のデータのみ抽出（体裁は維持）
df_filtered = df[(df["OxyTrue"] >= spo2_min) & (df["OxyTrue"] <= spo2_max)]

# 新しいファイルとして保存
output_path = fr"C:\Users\mpg\Desktop\python_rasio\Bland-Altman_analysis_spo2_{spo2_min}to{spo2_max}.xlsx"

df_filtered.to_excel(output_path, index=False)
