import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# ファイルパスを設定してExcelファイルを読み込む
file_name = 'C:/Users/mpg/Desktop/python_rasio/all_plot_data.xlsx'
df = pd.read_excel(file_name)

# 各subjectの列をチェックしてデータを取得
subject_count = len([col for col in df.columns if 'ratio' in col])  # 各subjectの数を取得
print(f"Number of subjects found: {subject_count}")
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 使用する色のリスト

name=input("グラフのタイトルを入力してください \n")

# 各subjectのデータをプロット
for i in range(1, subject_count + 1):
    # 各subjectの`ratio`と`spo2`列を取得
    ratio_column = f'subject{i}_ratio'
    spo2_column = f'subject{i}_spo2'

    # NaN値を除去してデータが存在するかをチェック
    subject_data = df[[ratio_column, spo2_column]].dropna()

    if not subject_data.empty:  # データが存在する場合にのみプロット
        plt.scatter(
            subject_data[ratio_column], 
            subject_data[spo2_column], 
            color=colors[i % len(colors)], 
            label=f'Subject {i}', 
            s=50,  # 点のサイズを設定
            alpha=0.7  # 点の透明度を設定
        )
    else:
        print(f"{ratio_column} or {spo2_column} does not have valid data and will be skipped.")


names= "median"
plt.xlabel('ratio')
plt.ylabel('SpO2 [%]')
plt.title(names, fontsize=20)
plt.legend(title='Subjects', loc='upper right')
plt.xlim(0.8, 2.1)
plt.ylim(75, 102.5)
output_file_path = f"C:/Users/mpg/Desktop/python_rasio/output_image/all_subject_plot_{names}.png"
plt.savefig(output_file_path)
plt.show()

# plt.figure(figsize=(12, 8))
for i in range(1, subject_count + 1):
    # 各subjectの ratio と spo2 列を取得
    ratio_column = f'subject{i}_ratio'
    spo2_column = f'subject{i}_spo2'
    
    # NaNを除去してデータを取得
    subject_data = df[[ratio_column, spo2_column]].dropna()

    if not subject_data.empty:
        X = subject_data[ratio_column].values.reshape(-1, 1)
        y = subject_data[spo2_column].values
        reg = LinearRegression().fit(X, y)
        
        # 傾きと切片を取得
        slope = reg.coef_[0]
        intercept = reg.intercept_
        # 散布図のプロット
        plt.scatter(
            subject_data[ratio_column], subject_data[spo2_column],
            color=colors[i % len(colors)], label=f'Subject {i}\ny = {slope:.2f}x + {intercept:.2f}',
            s=50, alpha=0.7
        )

        # 線形回帰を適用
        

        # 傾きと切片を2桁にフォーマットして表示
        print(f"Subject {i}: y = {slope:.2f}x + {intercept:.2f}")

        # 回帰直線の範囲を設定
        x_fit = np.linspace(X.min(), X.max(), 100)
        y_fit = reg.predict(x_fit.reshape(-1, 1))

        # 回帰直線をプロット
        plt.plot(x_fit, y_fit, color=colors[i % len(colors)], linestyle='--', linewidth=2)

# グラフの設定

plt.xlabel('ratio')
plt.ylabel('SpO2 [%]')
plt.title(names, fontsize=20)
plt.legend(title='Subjects', loc='upper right')
plt.xlim(0.8, 2.1)
plt.ylim(75, 102.5)

# 画像を保存
output_file_path = f"C:/Users/mpg/Desktop/python_rasio/output_image/all_subject_plot_fit_{name}_linear.png"
plt.savefig(output_file_path)
plt.show()
##
##
##
##
##
##
# ===============================
# √sample 重み付き回帰（同じ形式）
# ===============================

plt.figure()

for i in range(1, subject_count + 1):
    ratio_column  = f'subject{i}_ratio'
    spo2_column   = f'subject{i}_spo2'
    sample_column = f'sample{i}'

    # 必要な列がなければスキップ
    if not all(col in df.columns for col in [ratio_column, spo2_column, sample_column]):
        continue

    subject_data = df[[ratio_column, spo2_column, sample_column]].dropna()

    if not subject_data.empty:
        X = subject_data[ratio_column].values.reshape(-1, 1)
        y = subject_data[spo2_column].values
        weights = np.sqrt(subject_data[sample_column].values)  # ★重み

        # ★重み付き線形回帰
        reg = LinearRegression()
        reg.fit(X, y, sample_weight=weights)

        # 傾きと切片
        slope = reg.coef_[0]
        intercept = reg.intercept_

        # 散布図（元データ）
        plt.scatter(
            subject_data[ratio_column],
            subject_data[spo2_column],
            color=colors[i % len(colors)],
            label=f'Subject {i}(w)\n y = {slope:.2f}x + {intercept:.2f}',
            s=50,
            alpha=0.7
        )

        print(f"Subject {i} weighted: y = {slope:.2f}x + {intercept:.2f}")

        # 回帰直線
        x_fit = np.linspace(X.min(), X.max(), 100)
        y_fit = reg.predict(x_fit.reshape(-1, 1))

        plt.plot(
            x_fit,
            y_fit,
            color=colors[i % len(colors)],
            linestyle='--',
            linewidth=2
        )

# グラフ設定（元コードと完全一致）

plt.xlabel('ratio')
plt.ylabel('SpO2 [%]')
plt.title("fit with weight", fontsize=16)
plt.legend(title='Subjects', loc='upper right')
plt.xlim(0.8, 2.1)
plt.ylim(75, 102.5)

output_file_path = (
    f"C:/Users/mpg/Desktop/python_rasio/output_image/"
    f"all_subject_plot_fit_{names}_weighted.png"
)
plt.savefig(output_file_path)
plt.show()
