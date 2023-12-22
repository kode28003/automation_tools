
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

file_name = 'C:/Users/mpg/Desktop/python_rasio/peak.xlsx'
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


# 2つのデータセットの時間が等しいのみを抽出
merged_data = pd.merge(data1, data2, left_on=time1_col, right_on=time2_col, how='inner', suffixes=('_1', '_2'))

continuous_values1 = merged_data['peak_number1'][(merged_data['peak_number1'].diff() == 1) | (merged_data['peak_number1'].diff(-1) == -1)]
continuous_values2 = merged_data['peak_number2'][(merged_data['peak_number2'].diff() == 1) | (merged_data['peak_number2'].diff(-1) == -1)]
merged_data['continueNum'] = continuous_values1
merged_data['continueTime'] = merged_data['A'][merged_data['peak_number1'].isin(continuous_values1)]
merged_data['800nm'] = merged_data['B'][merged_data['peak_number1'].isin(continuous_values1)]
merged_data['940nm'] = merged_data['D'][merged_data['peak_number1'].isin(continuous_values1)]


# 連続する'continueNum'の組み合わせを特定
continuous_combinations = [(merged_data['continueNum'].iloc[i], merged_data['continueNum'].iloc[i+1]) for i in range(len(merged_data) - 1) if merged_data['continueNum'].iloc[i+1] - merged_data['continueNum'].iloc[i] == 1]


# '940nm'列の前後の数が連続の場合、'continueTime'の前後の平均を計算して'940nm_time'列に保存
merged_data['Peak_time_ave'] = np.nan
for num1, num2 in continuous_combinations:
    merged_data.loc[(merged_data['continueNum'] == num1) & (merged_data['continueNum'].shift(-1) == num2), 'Peak_time_ave'] = (merged_data['continueTime'] + merged_data['continueTime'].shift(-1)) / 2
# '940nm'列の前後の数が連続の場合、計算して'940nm_diff'列に保存
merged_data['800nm_Peak-Peak'] = np.nan
for num1, num2 in continuous_combinations:
    merged_data.loc[(merged_data['continueNum'] == num1) & (merged_data['continueNum'].shift(-1) == num2), '800nm_Peak-Peak'] = abs(merged_data['800nm'].shift(-1)) + abs(merged_data['800nm'])

merged_data['940nm_Peak-Peak'] = np.nan
for num1, num2 in continuous_combinations:
    merged_data.loc[(merged_data['continueNum'] == num1) & (merged_data['continueNum'].shift(-1) == num2), '940nm_Peak-Peak'] = abs(merged_data['940nm'].shift(-1)) + abs(merged_data['940nm'])

merged_data['ratio_Peak-Peak'] = np.where(
    merged_data['800nm_Peak-Peak'] != 0,
    merged_data['940nm_Peak-Peak'] / merged_data['800nm_Peak-Peak'],
    np.nan # 800nm_Peak-Peakが0の場合はNaNを割り当てる
)

print(merged_data)
# cleaned_merged_data = merged_data.dropna()

df2 = merged_data[['peak_number1', 'A', 'B', 'D']]
df3 = merged_data[['continueNum', 'continueTime', '800nm', '940nm']]
df4 = merged_data[['continueNum', 'continueTime', '800nm', '940nm', 'Peak_time_ave', '800nm_Peak-Peak', '940nm_Peak-Peak', 'ratio_Peak-Peak']]

# df_rasio=merged_data[['Peak_time_ave','ratio_Peak-Peak']]+df[['OxyTime','Spo2']]
df_rasio = pd.concat([merged_data[['Peak_time_ave','ratio_Peak-Peak']], df[['OxyTime','Spo2']]], axis=1)

output_file_name = 'C:/Users/mpg/Desktop/python_rasio/change_date_rasio.xlsx'
with pd.ExcelWriter(output_file_name) as writer:
    # 同じデータフレームを2つの異なるシートに保存
    df.to_excel(writer, sheet_name='original')
    df2.to_excel(writer, sheet_name='sameTimePeak')
    df3.to_excel(writer, sheet_name='continuePeak')
    df4.to_excel(writer, sheet_name='rasioPeak')
    df_rasio.to_excel(writer, sheet_name='s')
    merged_data.to_excel(writer, sheet_name='Sheet1')

# plt.scatter(merged_data['Peak_time_ave'], merged_data['ratio_Peak-Peak'])
# plt.xlabel('time [s]')
# plt.ylabel('rasio (940nm/800nm)')
# plt.title('rasio & time with Ratios')
# plt.legend()

# plt.savefig("C:/Users/mpg/Desktop/python_rasio/graph_image.png")
# plt.show()

def poly2(x, a, b, c):
    return a*x**2 + b*x + c

# NaNを除外したデータを用意する
cleaned_data = merged_data.dropna(subset=['Peak_time_ave', 'ratio_Peak-Peak'])
xdata = cleaned_data['Peak_time_ave']
ydata = cleaned_data['ratio_Peak-Peak']

# curve_fitを使用してパラメータを推定
params, params_covariance = curve_fit(poly2, xdata, ydata)
a, b, c = params
print("Fitting Curve Equation: y = {:.10f}x² + {:.5f}x + {:.5f}".format(a, b, c))

# 推定したパラメータを使ってフィッティングカーブをプロット
plt.scatter(xdata, ydata, label='Data')

# プロットの装飾
plt.xlabel('Peak Time Average [s]')
plt.ylabel('Ratio (940nm/800nm)')
plt.title('plot')
plt.legend()
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/plot.png")
plt.title('Curve Fitting')
plt.plot(xdata, poly2(xdata, *params), label="y = {:.6f}x² + {:.3f}x + {:.3f}".format(a, b, c), color='red')
plt.legend()
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/rasio_time_fitting.png")
plt.show()

plt.xlabel('Time [s]')
plt.ylabel('SpO2 [%]')
plt.title('SpO2 fluctuation')
plt.legend()
plt.plot( df['OxyTime'], df['Spo2'],color='red')
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/spo2.png")
plt.show()