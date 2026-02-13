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

# プロットの設定
#plt.figure(figsize=(12, 8))  # グラフのサイズを設定
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 使用する色のリスト

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
output_file_path = f"C:/Users/mpg/Desktop/python_rasio/output_image/all_subject_plot_fit_{names}.png"
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
name=input("グラフのタイトルを入力してください \n")
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

# 全体のデータを格納するリスト
all_ratios = []
all_spo2 = []


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

        # 全体のリストにデータを追加
        all_ratios.extend(subject_data[ratio_column].values)
        all_spo2.extend(subject_data[spo2_column].values)


all_ratios = np.array(all_ratios)
all_spo2 = np.array(all_spo2)

all_ratios_reshaped = all_ratios.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(all_ratios_reshaped, all_spo2)
slope = linear_regressor.coef_[0]
intercept = linear_regressor.intercept_
x_fit = np.linspace(all_ratios.min(), all_ratios.max(), 100)  # X軸の範囲を設定
y_fit = linear_regressor.predict(x_fit.reshape(-1, 1))

equation_text = f"SpO2 = {slope:.2f} * ratio + {intercept:.2f}"
plt.plot(x_fit, y_fit, color='black', linestyle='-', linewidth=2, label='Linear Fit')
print(f"Linear fit equation: SpO2 = {slope:.2f} * ratio + {intercept:.2f}")
plt.xlabel('ratio')
plt.ylabel('SpO2 [%]')
plt.title(equation_text,fontsize=15)
plt.legend(title='Subjects', loc='upper right')  # 凡例を右上に配置
plt.xlim(0.8, 2.1)  # X軸の範囲を指定
plt.ylim(75, 102.5)  # Y軸の範囲を指定
plt.savefig("C:/Users/mpg/Desktop/python_rasio/output_image/linear_fit_plot.png")  # 画像を保存
plt.show()


data = pd.DataFrame({'ratio': all_ratios, 'spo2': all_spo2})
median_data = data.groupby('spo2').median().reset_index()  # SpO2ごとにratioの中央値を求める
print(median_data)
#plt.figure(figsize=(12, 8))
plt.scatter(all_ratios, all_spo2, color='lightgray', label='Original Data', alpha=0.5)
plt.scatter(median_data['ratio'], median_data['spo2'], color='red', s=30, label='Median Points')
plt.xlabel('ratio')
plt.ylabel('SpO2 [%]')
plt.title("all Subjects "+name, fontsize=20)
plt.legend(title='Data Points', loc='upper right')
plt.xlim(0.8, 2.1) 
plt.ylim(75, 102.5)
output_file_path = f"C:/Users/mpg/Desktop/python_rasio/output_image/all_subject_plot_median_{name}.png"
plt.savefig(output_file_path) 
plt.show()


# 線形回帰モデルをフィッティング
x_median = median_data['spo2'].values.reshape(-1, 1)  # SpO2をX軸のデータに設定
y_median = median_data['ratio'].values  # ratioをY軸のデータに設定

linear_regressors = LinearRegression()
linear_regressors.fit(x_median, y_median)

# 回帰直線の係数と切片を取得
slope_median = linear_regressors.coef_[0]
intercept_median = linear_regressors.intercept_

# 回帰直線の方程式を変形して SpO2 = (1/slope) * ratio - (intercept/slope) の形にする
slope_inverse = 1 / slope_median
intercept_transformed = -intercept_median / slope_median

# 新しい回帰直線の式を作成
equation_texts = f"SpO2 = {slope_inverse:.2f} * ratio + {intercept_transformed:.2f}"

# 回帰直線をプロットするデータを作成
x_fit_median = np.linspace(median_data['spo2'].min(), median_data['spo2'].max(), 100)
y_fit_median = linear_regressors.predict(x_fit_median.reshape(-1, 1))

# グラフをプロット
plt.scatter(all_ratios, all_spo2, color='lightgray', label='Original Data', alpha=0.5)  # 元のデータをプロット
plt.scatter(median_data['ratio'], median_data['spo2'], color='red', s=30, label='Median Points')  # 中央値をプロット
plt.plot(y_fit_median, x_fit_median, color='black', linestyle='-', linewidth=2, label='Linear Fit (Median Points)')

# タイトルを SpO2 の式に変更
plt.title(equation_texts, fontsize=15)
plt.xlabel('ratio')
plt.ylabel('SpO2 [%]')
plt.legend(title='Data Points', loc='upper right')
plt.xlim(0.8, 2.1)  # X軸の範囲を指定
plt.ylim(75, 102.5)  # Y軸の範囲を指定

# 画像を保存して表示
output_file_path = f"C:/Users/mpg/Desktop/python_rasio/output_image/all_subject_plot_fit_{name}.png"
plt.savefig(output_file_path)
plt.show()


median_data_swapped = median_data[['ratio','spo2']]
output_file_path = "C:/Users/mpg/Desktop/python_rasio/output_image/all_plot_data_result_"+name+".xlsx"
with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
    median_data_swapped.to_excel(writer, sheet_name='subjects_median', index=False)
    