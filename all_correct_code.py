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
merged_data = pd.DataFrame()


# data1 と data2 の行をループして、条件を満たす行を見つけてマージ
for i, row1 in data1.iterrows():
    for j, row2 in data2.iterrows():
        # time1_col と time2_col の差の絶対値が 0.025 未満の場合 < 0.025:
        if abs(row1[time1_col] - row2[time2_col]) < 0.030:
            merged_row = pd.DataFrame([row1.tolist() + row2.tolist()])
            merged_data = pd.concat([merged_data, merged_row], ignore_index=True)

# カラム名の再設定
merged_data.columns = list(data1.columns) + list(data2.columns)

print(merged_data.head())

continuous_values1 = merged_data['peak_number1'][(merged_data['peak_number1'].diff() == 1) | (merged_data['peak_number1'].diff(-1) == -1)]
continuous_values2 = merged_data['peak_number2'][(merged_data['peak_number2'].diff() == 1) | (merged_data['peak_number2'].diff(-1) == -1)]
merged_data['continueNum'] = continuous_values1
merged_data['continueTime'] = merged_data['A'][merged_data['peak_number1'].isin(continuous_values1)]
merged_data['800nm'] = merged_data['B'][merged_data['peak_number1'].isin(continuous_values1)]
merged_data['940nm'] = merged_data['D'][merged_data['peak_number1'].isin(continuous_values1)]

# 連続する'continueNum'の組み合わせを特定
continuous_combinations = [(merged_data['continueNum'].iloc[i], merged_data['continueNum'].iloc[i+1]) for i in range(len(merged_data) - 1) if merged_data['continueNum'].iloc[i+1] - merged_data['continueNum'].iloc[i] == 1]

merged_data['Peak_time_ave'] = np.nan
for num1, num2 in continuous_combinations:
    merged_data.loc[(merged_data['continueNum'] == num1) & (merged_data['continueNum'].shift(-1) == num2), 'Peak_time_ave'] = (merged_data['continueTime'] + merged_data['continueTime'].shift(-1)) / 2

merged_data['800nm_Peak-Peak'] = np.nan
for num1, num2 in continuous_combinations:
    merged_data.loc[(merged_data['continueNum'] == num1) & (merged_data['continueNum'].shift(-1) == num2), '800nm_Peak-Peak'] = abs(merged_data['800nm'].shift(-1)) + abs(merged_data['800nm'])

merged_data['940nm_Peak-Peak'] = np.nan
for num1, num2 in continuous_combinations:
    merged_data.loc[(merged_data['continueNum'] == num1) & (merged_data['continueNum'].shift(-1) == num2), '940nm_Peak-Peak'] = abs(merged_data['940nm'].shift(-1)) + abs(merged_data['940nm'])

merged_data['ratio_Peak-Peak'] = np.where(
    merged_data['800nm_Peak-Peak'] != 0,
    merged_data['940nm_Peak-Peak'] / merged_data['800nm_Peak-Peak'],
    np.nan 
)
df2 = merged_data[['peak_number1', 'A', 'B', 'D']]
df3 = merged_data[['continueNum', 'continueTime', '800nm', '940nm']]
df4 = merged_data[['continueNum', 'continueTime', '800nm', '940nm', 'Peak_time_ave', '800nm_Peak-Peak', '940nm_Peak-Peak', 'ratio_Peak-Peak']]

merged_data = merged_data.dropna(subset=['Peak_time_ave', 'ratio_Peak-Peak'])

df.reset_index(drop=True, inplace=True)
merged_data.reset_index(drop=True, inplace=True)
df_rasio = pd.concat([merged_data[['Peak_time_ave', 'ratio_Peak-Peak']], df[['OxyTime', 'Spo2']]], axis=1)


df_rasio_int = df_rasio.copy()
df_rasio_int['Peak_time_ave'] = df_rasio_int['Peak_time_ave'].dropna().replace([np.inf, -np.inf], np.nan).astype(float).astype(int)
##
##
##
##setting###
movingAveragePoint=30 #隣接平均のポイント数 (default:30)

calibrationAveragePoint=13
calibrationTimeStart=10 #キャリブレーション開始 (defalult:10)
calibrationTimeEnd=40 #キャリブレーション終了時間 (defalult:40)
slope_num=116.04 #推定式の傾き
base_slope_num=110.47 #この実験のデータの傾き
k = 1.5       #±2SD=95% , ±1.5SD = 86.6% , ±1SD = 68.8%　が格納される範囲
##setting###
##
##
##
ratio_median = df_rasio_int.loc[
    (df_rasio_int['Peak_time_ave'] >= calibrationTimeStart) & (df_rasio_int['Peak_time_ave'] <= calibrationTimeEnd),
    'ratio_Peak-Peak'
].median()

subset = df_rasio_int.loc[
    (df_rasio_int['Peak_time_ave'] >= calibrationTimeStart) & (df_rasio_int['Peak_time_ave'] <= calibrationTimeEnd),
    'ratio_Peak-Peak'
]

subset = subset.rolling(window=calibrationAveragePoint).mean().dropna()

lower_limit = ratio_median * 0.8  
upper_limit = ratio_median * 1.2

# print(f"10〜40秒のratio中央値: {ratio_median:.2f}")
# print(f"lower_limit: {lower_limit:.2f}")
# print(f"upper_limit: {upper_limit:.2f}")


mean_ratio = subset.mean()
std_ratio = subset.std()
lower_limit_2sd = mean_ratio - k * std_ratio
upper_limit_2sd = mean_ratio + k * std_ratio

print(" ")
print(f"10〜40秒のratio平均: {mean_ratio:.2f}")
print(f"lower_limit_{k}sd (平均-{k}SD): {lower_limit_2sd:.2f}")
print(f"upper_limit_{k}sd (平均+{k}SD): {upper_limit_2sd:.2f}")
print(" ")
##
##
##現在使用している範囲##
mad = np.median(np.abs(subset - ratio_median))  # MAD計算
sigma_est = 1.4826 * mad                       # 補正して標準偏差相当
lower_limit_mad = ratio_median - k * sigma_est
upper_limit_mad = ratio_median + k * sigma_est
width = upper_limit_mad-lower_limit_mad

print(" ")
print(f"10〜40秒のratio中央値:{ratio_median:.2f}")
print(f"lower_limit_mad(中央値-{k}*MAD*1.48): {lower_limit_mad:.2f}")
print(f"upper_limit_mad(中央値+{k}*MAD*1.48): {upper_limit_mad:.2f}")
print(f"1.5SDによる幅: {width:.2f}")

##
##
##
def calc_spo2_ratio_range(slope, intercept, delta_R, spo2_min=75, spo2_max=96):
    results = []
    spo2_values = np.arange(spo2_max, spo2_min - 1, -1)
    for spo2 in spo2_values:
        R_center = (spo2 - intercept) / slope
        # ΔR の幅を使って上下限を算出
        R_lower = R_center - delta_R / 2
        R_upper = R_center + delta_R / 2
        # 参考としてSpO2の上下限も計算
        SpO2_lower = slope * R_lower + intercept
        SpO2_upper = slope * R_upper + intercept
        results.append({
            "SpO2": spo2,
            "R_center": R_center,
            "R_lower": R_lower,
            "R_upper": R_upper,
            "SpO2_lower": SpO2_lower,
            "SpO2_upper": SpO2_upper
        })
    return results

##
##
##

# 新しいデータフレームを格納するためのリスト
new_data = []
all_new_data=[]

    
def plot_ratio_and_spo2(df,name):
    fig, ax1 = plt.subplots(figsize=(10, 6))  # 図のサイズを指定
    color1 = 'tab:red'
    ax1.set_xlabel('time',fontsize=16)
    ax1.set_ylabel('ratio', color=color1,fontsize=16)
    ax1.plot(
        df['Peak_time_ave'],
        df['ratio_Peak-Peak'],
        color='lightgray',
        alpha=0.9,
        label='raw ratio'
    )
    df['ratio_Peak-Peak_MA'] = df['ratio_Peak-Peak'].rolling(window=movingAveragePoint).mean()
    ax1.plot(df['Peak_time_ave'], df['ratio_Peak-Peak_MA'], color=color1, label='ratio')
    ax1.tick_params(axis='y', labelcolor=color1,labelsize=14)
    ax1.set_ylim(0.95, 1.25)
    ax2 = ax1.twinx()  # 2つ目の縦軸を作成
    color2 = 'tab:blue'
    ax2.set_ylabel('Spo2', color=color2 ,fontsize=16)
    ax2.plot(df['Peak_time_ave'], df['Spo2'], color=color2, label='Pulse Oximeter')
    ax2.tick_params(axis='y', labelcolor=color2,labelsize=14)
    ax2.set_ylim(75, 102.5)
    plt.title('ratio & Spo2')
    ax1.legend(loc='upper left')  # Camera のラベルを左上に表示
    ax2.legend(loc='upper right')  # OxyTrue のラベルを右上に表示
    fig.tight_layout()  # レイアウトの調整
    plt.savefig(f"C:/Users/mpg/Desktop/python_rasio/output_image/spo2_ratio_line_{name}.png")  # 画像を保存
    plt.show()




for i in range(len(df_rasio_int)):
    for j in range(len(df_rasio_int)):
        if df_rasio_int.at[i, 'Peak_time_ave'] == df_rasio_int.at[j, 'OxyTime']: 
            ratio = df_rasio_int.at[i, 'ratio_Peak-Peak']
            spo2 = df_rasio_int.at[j, 'Spo2']
            
            all_new_data.append({
                'Peak_time_ave': df_rasio_int.at[i, 'Peak_time_ave'],
                'ratio_Peak-Peak': ratio,
                'Spo2': spo2
            })
            
            if 97 <= spo2 <= 100 and (ratio < lower_limit_mad or ratio > upper_limit_mad):
            # if 97 <= spo2 <= 100 and (ratio < lower_limit_2sd or ratio > upper_limit_2sd):
            # if 97 <= spo2 <= 100 and (ratio < lower_limit or ratio > upper_limit):
                continue
            # spo2が97未満なら upper_limit以上のみスキップ
            elif spo2 < 97 and ratio > upper_limit_mad:
            # elif spo2 < 97 and ratio > upper_limit_2sd:
            # elif spo2 < 97 and ratio > upper_limit:
                continue
            
            new_data.append({
                'Peak_time_ave': df_rasio_int.at[i, 'Peak_time_ave'],
                'ratio_Peak-Peak': ratio,
                'Spo2': spo2
            })

new_df = pd.DataFrame(new_data)
all_new_df = pd.DataFrame(all_new_data)
plot_ratio_and_spo2(new_df,"only_upper")
new_df["Spo2_int"] = np.floor(new_df["Spo2"]).astype(int)
sample_counts = new_df.groupby("Spo2_int").size().reset_index(name="n_samples")
sample_counts["weight"] = np.sqrt(sample_counts["n_samples"])
new_df_include_sample_num = new_df.merge(sample_counts, on="Spo2_int", how="left")
subset_10_40 = new_df[(new_df['Peak_time_ave'] >= calibrationTimeStart) & (new_df['Peak_time_ave'] <= calibrationTimeEnd)].copy()
subset_10_40['b'] = subset_10_40['Spo2'] - slope_num * subset_10_40['ratio_Peak-Peak']


# 切片bの中央値
b_median = subset_10_40['b'].median()
print(" ")
print(f"10〜40秒の切片中央値 b: {b_median:.2f}")
print(f"キャリブレーション後の近似式: SpO2 = {slope_num:.2f} * ratio + {b_median:.2f}")

##
##
##
###

results = calc_spo2_ratio_range(slope_num, b_median, width)
new_data_dyn = []      # フィルタ後データ（動的上限下限）
all_new_data_dyn = []  # 全データ（除外含む）

for i in range(len(df_rasio_int)):
    for j in range(len(df_rasio_int)):
        if df_rasio_int.at[i, 'Peak_time_ave'] == df_rasio_int.at[j, 'OxyTime']:
            
            ratio = df_rasio_int.at[i, 'ratio_Peak-Peak']
            spo2 = df_rasio_int.at[j, 'Spo2']

            # 全データを保存
            all_new_data_dyn.append({
                'Peak_time_ave': df_rasio_int.at[i, 'Peak_time_ave'],
                'ratio_Peak-Peak': ratio,
                'Spo2': spo2
            })

            # ===== SpO₂ごとの上限・下限を取得 =====
            # SpO₂に最も近い結果を検索
            matched_row = min(results, key=lambda x: abs(x["SpO2"] - spo2))
            upper_limit = matched_row["R_upper"]
            lower_limit = matched_row["R_lower"]
            # =======================================

            # ===== 除外条件 =====
            # SpO₂が97〜100% → 上下両側チェック
            if 97 <= spo2 <= 100:
                if ratio < lower_limit or ratio > upper_limit:
                    continue
                
            elif ratio < lower_limit or ratio > upper_limit:
                continue

            # 条件を通過したデータを保存
            new_data_dyn.append({
                'Peak_time_ave': df_rasio_int.at[i, 'Peak_time_ave'],
                'ratio_Peak-Peak': ratio,
                'Spo2': spo2#,
                # 'R_upper_used': upper_limit,
                # 'R_lower_used': lower_limit
            })

df_new_cutted = pd.DataFrame(new_data_dyn)
df_all_cutted = pd.DataFrame(all_new_data_dyn)
df_new_cutted["Spo2_int"] = np.floor(df_new_cutted["Spo2"]).astype(int)
sample_counts = df_new_cutted.groupby("Spo2_int").size().reset_index(name="n_samples")
sample_counts["weight"] = np.sqrt(sample_counts["n_samples"])
new_df_include_sample_num_cutted = df_new_cutted.merge(sample_counts, on="Spo2_int", how="left")
subset_10_40_cutted = df_new_cutted[(df_new_cutted['Peak_time_ave'] >= calibrationTimeStart) & (df_new_cutted['Peak_time_ave'] <= calibrationTimeEnd)].copy()
subset_10_40_cutted['b'] = subset_10_40_cutted['Spo2'] - base_slope_num * subset_10_40_cutted['ratio_Peak-Peak']


# 切片bの中央値
cutted_median = subset_10_40_cutted['b'].median()
print(" ")
print(f"10〜40秒の切片中央値 cutted b: {cutted_median:.2f}")
print(f"キャリブレーション後の近似式 cutted: SpO2 = {base_slope_num:.2f} * ratio + {cutted_median:.2f}")
##
##
##

plot_ratio_and_spo2(df_new_cutted,"upper_lowwer")

n_alls = len(df_all_cutted)
n_news = len(df_new_cutted)
diff = n_alls - n_news  # 除外された件数
# plt.figure(figsize=(8,6))
plt.scatter(df_all_cutted['ratio_Peak-Peak'], df_all_cutted['Spo2'], 
            label=f'raw data (cutted: {diff} points)', 
            color='lightgray', alpha=0.7)
plt.scatter(df_new_cutted['ratio_Peak-Peak'], df_new_cutted['Spo2'], 
            label='cutted data')
plt.xlabel('ratio', fontsize=16)
plt.ylabel('SpO2 [%]', fontsize=16)
plt.legend()
plt.xlim(0.8, 2.1)
plt.ylim(75, 102.5)
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/cutted_plot.png")
plt.show()

# --- 推定SpO2を新規データフレームとして作成 ---
new_df_with_est = new_df.copy()
new_df_with_est['intercept'] = new_df_with_est['Spo2'] - slope_num * new_df_with_est['ratio_Peak-Peak']
new_df_with_est['intercept_median'] = float('nan')
new_df_with_est.loc[new_df_with_est.index[0], 'intercept_median'] = "intercept_median"
new_df_with_est.loc[new_df_with_est.index[1], 'intercept_median'] = b_median
new_df_with_est['Spo2_est'] = slope_num * new_df_with_est['ratio_Peak-Peak'] + b_median
new_df_with_est['Spo2_est'] = new_df_with_est['Spo2_est'].clip(upper=100)
new_df_with_est['Spo2_est_smooth'] = new_df_with_est['Spo2_est'].rolling(window=movingAveragePoint).mean()


plt.figure(figsize=(10, 6))
plt.plot(new_df_with_est['Peak_time_ave'], new_df_with_est['Spo2_est_smooth'], color='red', label='Camera', linewidth=2)
plt.plot(new_df_with_est['Peak_time_ave'], new_df_with_est['Spo2'], color='blue', label='PulseOxy', linewidth=2)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('SpO₂ (%)', fontsize=16)
plt.title('Compare SpO2')
plt.ylim(75, 103)          # 縦軸固定
plt.xlim(0, 540)           # 横軸固定
plt.xticks(fontsize=18)               # x軸の目盛サイズ
plt.yticks(fontsize=18)    
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"C:/Users/mpg/Desktop/python_rasio/output_image/compSpO2_slope_{slope_num}.png")
plt.show()

#
# DataFrame のコピー
df_new_cutted_est = df_new_cutted.copy()
df_new_cutted_est['intercept'] = df_new_cutted_est['Spo2'] - slope_num * df_new_cutted_est['ratio_Peak-Peak']
df_new_cutted_est['intercept_median'] = float('nan')
df_new_cutted_est.loc[df_new_cutted_est.index[0], 'intercept_median'] = "intercept_median"
df_new_cutted_est.loc[df_new_cutted_est.index[1], 'intercept_median'] = b_median
df_new_cutted_est['Spo2_est'] = slope_num * df_new_cutted_est['ratio_Peak-Peak'] + b_median
df_new_cutted_est['Spo2_est'] = df_new_cutted_est['Spo2_est'].clip(upper=100)
df_new_cutted_est['Spo2_est_smooth'] = df_new_cutted_est['Spo2_est'].rolling(window=movingAveragePoint).mean()


plt.figure(figsize=(10, 6))
plt.plot(df_new_cutted_est['Peak_time_ave'], df_new_cutted_est['Spo2_est_smooth'], color='red', label='Camera', linewidth=2)
plt.plot(df_new_cutted_est['Peak_time_ave'], df_new_cutted_est['Spo2'], color='blue', label='PulseOxy', linewidth=2)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('SpO₂ (%)', fontsize=16)
plt.title('Compare SpO2')
plt.ylim(75, 103)          # 縦軸固定
plt.xlim(0, 540)           # 横軸固定
plt.xticks(fontsize=18)               # x軸の目盛サイズ
plt.yticks(fontsize=18)    
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"C:/Users/mpg/Desktop/python_rasio/output_image/compSpO2_slope_{slope_num}_cutted.png")
plt.show()

##
##
##
##
df_new_cutted_est_base_slope_num = df_new_cutted.copy()
df_new_cutted_est_base_slope_num['intercept'] = df_new_cutted_est_base_slope_num['Spo2'] - base_slope_num * df_new_cutted_est_base_slope_num['ratio_Peak-Peak']
df_new_cutted_est_base_slope_num['intercept_median'] = float('nan')
df_new_cutted_est_base_slope_num.loc[df_new_cutted_est_base_slope_num.index[0], 'intercept_median'] = "intercept_median"
df_new_cutted_est_base_slope_num.loc[df_new_cutted_est_base_slope_num.index[1], 'intercept_median'] = cutted_median
df_new_cutted_est_base_slope_num['Spo2_est'] = base_slope_num * df_new_cutted_est_base_slope_num['ratio_Peak-Peak'] + cutted_median
df_new_cutted_est_base_slope_num['Spo2_est'] = df_new_cutted_est_base_slope_num['Spo2_est'].clip(upper=100)
df_new_cutted_est_base_slope_num['Spo2_est_smooth'] = df_new_cutted_est_base_slope_num['Spo2_est'].rolling(window=movingAveragePoint).mean()


plt.figure(figsize=(10, 6))
plt.plot(df_new_cutted_est['Peak_time_ave'], df_new_cutted_est['Spo2'], color='blue', label='PulseOxy', linewidth=3)
plt.plot(df_new_cutted_est['Peak_time_ave'], df_new_cutted_est['Spo2_est_smooth'], color='red', label=f'Camera (slope:{slope_num})', linewidth=3)
plt.plot(df_new_cutted_est_base_slope_num['Peak_time_ave'], df_new_cutted_est_base_slope_num['Spo2_est_smooth'], color='green', label=f'Camera (base slope:{base_slope_num})', linewidth=2,linestyle='--' , alpha=0.7)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('SpO₂ (%)', fontsize=16)
plt.title('Compare SpO2 base slope')
plt.ylim(75, 103)          # 縦軸固定
plt.xlim(0, 540)           # 横軸固定
plt.xticks(fontsize=18)               # x軸の目盛サイズ
plt.yticks(fontsize=18)    
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"C:/Users/mpg/Desktop/python_rasio/output_image/compSpO2_base_slope_{base_slope_num}_cutted.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_new_cutted_est['Peak_time_ave'], df_new_cutted_est['Spo2'], color='blue', label='PulseOxy', linewidth=3)
plt.plot(df_new_cutted_est['Peak_time_ave'], df_new_cutted_est['Spo2_est_smooth'], color='red', label=f'Camera (slope:{slope_num})', linewidth=3)
plt.plot(df_new_cutted_est_base_slope_num['Peak_time_ave'], df_new_cutted_est_base_slope_num['Spo2_est_smooth'], color='green', label=f'Camera (base slope:{base_slope_num})', linewidth=2,linestyle='--' , alpha=0.7)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('SpO₂ (%)', fontsize=16)
plt.title('Compare SpO2 base slope')
# plt.ylim(75, 103)          # 縦軸固定
plt.xlim(0, 540)           # 横軸固定
plt.xticks(fontsize=18)               # x軸の目盛サイズ
plt.yticks(fontsize=18)    
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"C:/Users/mpg/Desktop/python_rasio/output_image/compSpO2_base_slope_{base_slope_num}_cutted_non_ylim.png")
plt.show()


df_plot_data = pd.DataFrame({
    'Peak_time_ave_1': df_new_cutted_est['Peak_time_ave'],
    'Pulse_Oxy_Spo2': df_new_cutted_est['Spo2'],
    'Peak_time_ave_2': df_new_cutted_est['Peak_time_ave'],
    f'approximated_slope_{slope_num}': df_new_cutted_est['Spo2_est_smooth'],
    'Peak_time_ave_3': df_new_cutted_est_base_slope_num['Peak_time_ave'],
    f'this_median_base_slope_{base_slope_num}': df_new_cutted_est_base_slope_num['Spo2_est_smooth']
})
##
##
##
##

def random_data(new_df, n=3): #random作成
    result = pd.DataFrame()
    for spo2_value in range(100, 75, -1):  # SpO2 values 100, 99, 98
        spo2_df = new_df[new_df['Spo2'] == spo2_value]
        if len(spo2_df) >= n:
            sampled_df = spo2_df.sample(n)
            result = pd.concat([result, sampled_df])
    return result

sampled_df = random_data(new_df)
random_df=sampled_df[['ratio_Peak-Peak', 'Spo2']]


def median_ratio_for_integer_spo2(df):
    df_copy = df.copy()
    # Spo2を小数も含めて、そのまま使用
    df_copy['Spo2_int'] = np.floor(df_copy['Spo2']).astype(int)  # 整数部分に注目
    # 各整数Spo2ごとに中央値を計算
    median_ratio_per_integer_spo2 = df_copy.groupby('Spo2_int')['ratio_Peak-Peak'].median().reset_index()
    median_ratio_per_integer_spo2.rename(columns={'Spo2_int': 'Spo2'}, inplace=True)
    return median_ratio_per_integer_spo2

def median_ratio_for_integer_spo2_weigh(df):
    # Spo2を切り捨てて整数化
    df = df.copy()
    df['Spo2'] = np.floor(df['Spo2']).astype(int)  # 元の列名をそのまま上書き
    # 各整数Spo2ごとに中央値とサンプル数を計算
    grouped = df.groupby('Spo2').agg(
        median_ratio=('ratio_Peak-Peak', 'median'),
        n_samples=('ratio_Peak-Peak', 'size')
    ).reset_index()
    grouped['weight'] = np.sqrt(grouped['n_samples'])
    return grouped

def mean_ratio_for_integer_spo2(df):
    integer_spo2_df = df[df['Spo2'] % 1 == 0]
    mean_ratio_per_integer_spo2 = integer_spo2_df.groupby('Spo2')['ratio_Peak-Peak'].mean().reset_index()
    mean_ratio_per_integer_spo2 = mean_ratio_per_integer_spo2[['ratio_Peak-Peak', 'Spo2']]
    return mean_ratio_per_integer_spo2

median_df = median_ratio_for_integer_spo2(new_df)
median_df_upper_lowwer = median_ratio_for_integer_spo2(df_new_cutted)

median_df_weigh = median_ratio_for_integer_spo2_weigh(new_df)
cutted_median_df_weigh=median_ratio_for_integer_spo2_weigh(df_new_cutted)


mean_df = mean_ratio_for_integer_spo2(new_df)
output_file_name = 'C:/Users/mpg/Desktop/python_rasio/change_date_rasio.xlsx'




with pd.ExcelWriter(output_file_name) as writer:
    # random_df.to_excel(writer, sheet_name='random')
    # mean_df.to_excel(writer, sheet_name='mean')
    # filtered_df.to_excel(writer, sheet_name='filtered')
    df_plot_data.to_excel(writer, sheet_name='result_graph')
    new_df_include_sample_num_cutted.to_excel(writer, sheet_name='weigh_with_cutted')
    cutted_median_df_weigh.to_excel(writer, sheet_name='cutted_weigh')
    median_df_upper_lowwer.to_excel(writer, sheet_name='cutted_median')
    df_new_cutted_est_base_slope_num.to_excel(writer, sheet_name='comp_spo2_cutted_base_slope')
    df_new_cutted_est.to_excel(writer, sheet_name='comp_spo2_cutted')
    median_df_weigh.to_excel(writer, sheet_name='weigh')
    new_df_include_sample_num.to_excel(writer, sheet_name='weigh_with')
    median_df.to_excel(writer, sheet_name='median')
    new_df_with_est.to_excel(writer, sheet_name='comp_spo2') 
    new_df.to_excel(writer, sheet_name='result_relation_spo2')
    df_rasio.to_excel(writer, sheet_name='result')
    df_rasio_int.to_excel(writer, sheet_name='correct_time_result')
    df.to_excel(writer, sheet_name='original')
    df2.to_excel(writer, sheet_name='sameTimePeak')
    df3.to_excel(writer, sheet_name='continuePeak')
    df4.to_excel(writer, sheet_name='rasioPeak')
    merged_data.to_excel(writer, sheet_name='AllDate')


cleaned_data = merged_data.dropna(subset=['Peak_time_ave', 'ratio_Peak-Peak'])
xdata = cleaned_data['Peak_time_ave']
ydata = cleaned_data['ratio_Peak-Peak']

n_all = len(all_new_df)
n_new = len(new_df)


plt.scatter(all_new_df['ratio_Peak-Peak'], all_new_df['Spo2'])
plt.xlabel('ratio', fontsize=16)
plt.ylabel('SpO2 [%]', fontsize=16)
plt.legend()
plt.xlim(0.8, 2.1) 
plt.ylim(75, 102.5) 
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/all_plot.png")
plt.show()


# 差を計算
diff = abs(n_all - n_new)
plt.scatter(all_new_df['ratio_Peak-Peak'], all_new_df['Spo2'], label=f'raw data(cutted: {diff} points)',color='lightgray',alpha=0.9)###################
plt.scatter(new_df['ratio_Peak-Peak'], new_df['Spo2'], label='cutted data')
plt.xlabel('ratio', fontsize=16)
plt.ylabel('SpO2 [%]', fontsize=16)
plt.legend()
plt.xlim(0.8, 2.1) 
plt.ylim(75, 102.5) 
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/cal_cutted_plot.png")
plt.show()


#median
plt.scatter(median_df['ratio_Peak-Peak'], median_df['Spo2'],label='Data')
plt.xlabel('ratio',fontsize=16)
plt.ylabel('SpO2 [%]',fontsize=16)
plt.title('median')
plt.legend()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0.8, 2.1) 
plt.ylim(75, 102.5) 
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/median.png")
plt.show()
##
##median2
##
plt.figure(figsize=(8,6))
plt.scatter(median_df_upper_lowwer['ratio_Peak-Peak'], median_df_upper_lowwer['Spo2'], 
            label='cutted Data')
# 軸ラベル・タイトル
plt.xlabel('ratio', fontsize=16)
plt.ylabel('SpO2 [%]', fontsize=16)
plt.title('cutted Data (new_data_dyn)', fontsize=16)
plt.legend()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0.8, 2.1)
plt.ylim(75, 102.5)
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/new_data_dyn.png")
plt.show()
##
##

x = median_df['ratio_Peak-Peak']
y = median_df['Spo2']

# 線形回帰（1次近似）を計算
slope, intercept = np.polyfit(x, y, 1)
y_fit = slope * x + intercept
print(" ")
print(f"Linear fit normal: SpO2 = {slope:.2f} * ratio + {intercept:.2f}")
# プロット
plt.figure()
plt.scatter(x, y, label='Data')
plt.plot(x, y_fit, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('ratio', fontsize=16)
plt.ylabel('SpO2 [%]', fontsize=16)
plt.title('median')
plt.legend()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0.8, 2.1)
plt.ylim(75, 102.5)
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/median_fit.png")
plt.show()


x1 = median_df_weigh['median_ratio'].values
y1 = median_df_weigh['Spo2'].values
w1 = median_df_weigh['weight'].values   # √n_samples を重みとして使用

# 重み付き線形回帰
slope1, intercept1 = np.polyfit(x1, y1, 1, w=w1)

# フィット直線
y_fit1 = slope1 * x1 + intercept1
print(f"Linear fit & weight: SpO2 = {slope1:.2f} * ratio + {intercept1:.2f}")
plt.figure()
plt.scatter(x1, y1, label='Data')
plt.plot(x1, y_fit1, color='red', label=f'Fit: y = {slope1:.2f}x + {intercept1:.2f}')
plt.xlabel('ratio', fontsize=16)
plt.ylabel('SpO2 [%]', fontsize=16)
plt.title('with weight')
plt.legend()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0.8, 2.1)
plt.ylim(75, 102.5)
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/median_fit_with_weight.png")
plt.show()
##
##
##
x2 = cutted_median_df_weigh['median_ratio'].values
y2 = cutted_median_df_weigh['Spo2'].values
w2 = cutted_median_df_weigh['weight'].values   # √n_samples を重みとして使用

# 重み付き線形回帰
slope2, intercept2 = np.polyfit(x2, y2, 1, w=w2)

# フィット直線
y_fit2 = slope2 * x2 + intercept2
print(f"Linear fit & weight & cutted: SpO2 = {slope2:.2f} * ratio + {intercept2:.2f}")
plt.figure()
plt.scatter(x2, y2, label='Data')
plt.plot(x2, y_fit2, color='red', label=f'Fit: y = {slope2:.2f}x + {intercept2:.2f}')
plt.xlabel('ratio', fontsize=16)
plt.ylabel('SpO2 [%]', fontsize=6)
plt.title('cutted with weight')
plt.legend()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0.8, 2.1)
plt.ylim(75, 102.5)
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/cutted_median_fit_with_weight.png")
plt.show()

##
##
##
#average
# plt.scatter(mean_df['ratio_Peak-Peak'], mean_df['Spo2'],label='Data')
# plt.xlabel('ratio')
# plt.ylabel('SpO2 [%]')
# plt.title('average')
# plt.legend()
# plt.xlim(0.8, 2.1) 
# plt.ylim(75, 102.5) 
# plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/average.png")
# plt.show()

#random
# plt.scatter(sampled_df['ratio_Peak-Peak'], sampled_df['Spo2'],label='Data')
# plt.xlabel('ratio')
# plt.ylabel('SpO2 [%]')
# plt.title('random extraction')
# plt.legend()
# plt.xlim(0.8, 2.1) 
# plt.ylim(75, 102.5) 
# plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/random.png")
# plt.show()
