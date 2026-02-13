import pandas as pd

# ファイル読み込み
input_path = r"C:\Users\mpg\Desktop\python_rasio\Bland-Altman_analysis_spo2.xlsx"
df = pd.read_excel(input_path)

# 閾値
oxy_threshold = 95

# OxyTrue >= 95 の行を削除（体裁は維持）
df_filtered = df[df["OxyTrue"] < oxy_threshold]

# 新しいファイルとして保存
output_path = r"C:\Users\mpg\Desktop\python_rasio\Bland-Altman_analysis_spo2_OxyTrue_lt95.xlsx"
df_filtered.to_excel(output_path, index=False)
