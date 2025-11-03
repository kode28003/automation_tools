import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

##setting###
file_name = 'C:/Users/mpg/Desktop/python_rasio/peak.xlsx' #Peakのファイル名

peakAveragePoint = 3        #脈波ピークに対する隣接平均のポイント数
movingAveragePoint=13       #波形全体に対する隣接平均のポイント数
calibrationAveragePoint=10  #実験開始30秒間の隣接平均のポイント数
calibrationTimeStart=10     #キャリブレーション開始
calibrationTimeEnd=40       #キャリブレーション終了時間
slope_num=116.04            #推定式の傾き
base_slope_num=117.56       #この実験のデータの傾き
k = 1.5                     #±2SD=95% , ±1.5SD = 86.6% , ±1SD = 68.8%
heartRateAve = 15           #脈拍に対する隣接平均のポイント数
min_hr = 35                 #下限 (bpm)
max_hr = 220                #上限 (bpm)
##setting###

df = pd.read_excel(file_name)
excel_row_count = len(df)

# 時間1と振幅1のデータ
time1_col = 'A'
amplitude1_col = 'B'
data1 = df[[time1_col, amplitude1_col]].sort_values(by=time1_col)
data1['peak_number1'] = range(1, excel_row_count + 1)
data1 = data1.reset_index(drop=True)

# 時間2と振幅2のデータ
time2_col = 'C'
amplitude2_col = 'D'
data2 = df[[time2_col, amplitude2_col]].sort_values(by=time2_col)
data2['peak_number2'] = range(1, excel_row_count + 1)
data2 = data2.reset_index(drop=True)

# ==========================================
# ▼ 正のピークのみを抽出
# ==========================================
data1_pos = data1[data1[amplitude1_col] > 0].reset_index(drop=True)
data2_pos = data2[data2[amplitude2_col] > 0].reset_index(drop=True)

# ==========================================
# ▼ 脈拍数（bpm）を算出
# ==========================================
data1_pos['peak_interval'] = data1_pos[time1_col].diff()
data1_pos['heart_rate_bpm'] = 60 / data1_pos['peak_interval']

data2_pos['peak_interval'] = data2_pos[time2_col].diff()
data2_pos['heart_rate_bpm'] = 60 / data2_pos['peak_interval']

# NaN除去
data1_hr = data1_pos.dropna(subset=['heart_rate_bpm']).reset_index(drop=True)
data2_hr = data2_pos.dropna(subset=['heart_rate_bpm']).reset_index(drop=True)

# ==========================================
# ▼ 人間の脈波としてありえない値を除去
# ==========================================
data1_hr = data1_hr[(data1_hr['heart_rate_bpm'] >= min_hr) &
                    (data1_hr['heart_rate_bpm'] <= max_hr)].reset_index(drop=True)
data2_hr = data2_hr[(data2_hr['heart_rate_bpm'] >= min_hr) &
                    (data2_hr['heart_rate_bpm'] <= max_hr)].reset_index(drop=True)

# ==========================================
# ▼ スムージング（移動平均）
# ==========================================
data1_hr['heart_rate_bpm_smooth'] = data1_hr['heart_rate_bpm'].rolling(window=heartRateAve).mean()
data2_hr['heart_rate_bpm_smooth'] = data2_hr['heart_rate_bpm'].rolling(window=heartRateAve).mean()

# ==========================================
# ▼ 折れ線グラフ（横軸＝時間）
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(data1_hr[time1_col], data1_hr['heart_rate_bpm_smooth'], color='red', label='760nm', linewidth=3)
plt.plot(data2_hr[time2_col], data2_hr['heart_rate_bpm_smooth'], color='green', label='940nm', linewidth=3)
plt.plot(df['OxyTime'], df['HR'], color='blue', label='HR (bpm)', linewidth=3)

plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Heart Rate [bpm]', fontsize=16)
plt.xlim(0, 540)
plt.ylim(50, 115)
plt.title('Heart Rate', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/comp_HR.png")
plt.show()


# ==========================================
# ▼ Excel出力用データ統合
# ==========================================
merged_data1 = pd.merge(
    data1,
    data1_hr[[time1_col, 'heart_rate_bpm', 'heart_rate_bpm_smooth']],
    on=time1_col, how='left'
)
merged_data1.rename(columns={
    amplitude1_col: 'Amplitude',
    time1_col: 'Time [s]',
    'heart_rate_bpm': 'HeartRate[bpm]',
    'heart_rate_bpm_smooth': f'Smoothed({heartRateAve}-pt)'
}, inplace=True)

merged_data2 = pd.merge(
    data2,
    data2_hr[[time2_col, 'heart_rate_bpm', 'heart_rate_bpm_smooth']],
    on=time2_col, how='left'
)
merged_data2.rename(columns={
    amplitude2_col: 'Amplitude',
    time2_col: 'Time [s]',
    'heart_rate_bpm': 'HeartRate[bpm]',
    'heart_rate_bpm_smooth': f'Smoothed({heartRateAve}-pt)'
}, inplace=True)

# ==========================================
# ▼ 接触型HRを秒単位で対応付け
# ==========================================
def safe_int_seconds(series):
    series = pd.to_numeric(series, errors='coerce')
    series = series.replace([np.inf, -np.inf], np.nan)
    return series.dropna().astype(int)

# 接触型HR（基準）
df['sec'] = safe_int_seconds(df['OxyTime'])

# 各波長の整数秒
merged_data1['sec'] = safe_int_seconds(merged_data1['Time [s]'])
merged_data2['sec'] = safe_int_seconds(merged_data2['Time [s]'])

# 秒単位で接触型HRを追加
merged_data1 = pd.merge(merged_data1, df[['sec', 'HR']], on='sec', how='left')
merged_data2 = pd.merge(merged_data2, df[['sec', 'HR']], on='sec', how='left')

merged_data1.rename(columns={'HR': 'Reference_HR(bpm)'}, inplace=True)
merged_data2.rename(columns={'HR': 'Reference_HR(bpm)'}, inplace=True)

merged_data1.drop(columns=['sec'], inplace=True)
merged_data2.drop(columns=['sec'], inplace=True)

# ==========================================
# ▼ HR正ピークのみデータ
# ==========================================
hr_pos_760 = data1_hr[[time1_col, 'heart_rate_bpm', 'heart_rate_bpm_smooth']].copy()
hr_pos_760.rename(columns={
    time1_col: 'Time [s]',
    'heart_rate_bpm': 'HeartRate[bpm]',
    'heart_rate_bpm_smooth': f'Smoothed({heartRateAve}-pt)'
}, inplace=True)
hr_pos_760 = hr_pos_760.sort_values('Time [s]').reset_index(drop=True)

hr_pos_940 = data2_hr[[time2_col, 'heart_rate_bpm', 'heart_rate_bpm_smooth']].copy()
hr_pos_940.rename(columns={
    time2_col: 'Time [s]',
    'heart_rate_bpm': 'HeartRate[bpm]',
    'heart_rate_bpm_smooth': f'Smoothed({heartRateAve}-pt)'
}, inplace=True)
hr_pos_940 = hr_pos_940.sort_values('Time [s]').reset_index(drop=True)


# ==========================================
# ▼ HR_Positive_* シートに接触型HRを対応付け
# ==========================================

# 接触型HR（整数秒）
hr_contact = pd.DataFrame({
    'OxyTime': df['OxyTime'],
    'HR_contact': df['HR']
}).dropna()
hr_contact['sec'] = hr_contact['OxyTime'].astype(int)

# --- 760nm ---
hr_pos_760['sec'] = hr_pos_760['Time [s]'].astype(int)
hr_pos_760 = pd.merge(
    hr_pos_760,
    hr_contact[['sec', 'HR_contact']],
    on='sec',
    how='left'
)
hr_pos_760.rename(columns={'HR_contact': 'Contact_HR(bpm)'}, inplace=True)
hr_pos_760.drop(columns=['sec'], inplace=True)

# --- 940nm ---
hr_pos_940['sec'] = hr_pos_940['Time [s]'].astype(int)
hr_pos_940 = pd.merge(
    hr_pos_940,
    hr_contact[['sec', 'HR_contact']],
    on='sec',
    how='left'
)
hr_pos_940.rename(columns={'HR_contact': 'Contact_HR(bpm)'}, inplace=True)
hr_pos_940.drop(columns=['sec'], inplace=True)
# ==========================================
# ▼ Excel出力
# ==========================================
output_path = 'C:/Users/mpg/Desktop/python_rasio/heart_rate_result.xlsx'
with pd.ExcelWriter(output_path) as writer:
    merged_data1.to_excel(writer, sheet_name='Wavelength_760nm', index=False)
    merged_data2.to_excel(writer, sheet_name='Wavelength_940nm', index=False)
    hr_pos_760.to_excel(writer, sheet_name='HR_Positive_760nm', index=False)
    hr_pos_940.to_excel(writer, sheet_name='HR_Positive_940nm', index=False)

print(f"結果をExcelに保存しました: {output_path}")