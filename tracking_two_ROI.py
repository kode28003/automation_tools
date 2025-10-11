import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# 動画と白色板画像のパス
video_path = 'C:/Users/mpg/Desktop/python_rasio/yoneyama.mp4'
white_img_path = 'C:/Users/mpg/Desktop/python_rasio/normal_white.bmp'

# 白色板画像の読み込み（グレースケール）
white_img = cv2.imread(white_img_path, cv2.IMREAD_GRAYSCALE)
if white_img is None:
    print("白色板画像が読み込めません")
    exit()

# 動画読み込み
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("動画が開けません")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
max_history_frames = int(fps * 6)  # 表示範囲6秒

# 最初のフレーム取得とROI選択
ret, frame = cap.read()
if not ret:
    print("最初のフレームが読み込めません")
    exit()

roi = cv2.selectROI('Select ROI', frame, showCrosshair=True, fromCenter=False)
cv2.destroyWindow('Select ROI')

roi_x, roi_y, roi_w, roi_h = map(int, roi)

# 2つ目のROIは左に980px、下に2pxずらす（座標計算）
roi2_x = max(roi_x - 980, 0)
roi2_y = min(roi_y + 2, frame.shape[0] - roi_h)  # 画面外にはみ出さないように調整

# トラッカー初期化
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, (roi_x, roi_y, roi_w, roi_h))

# データ格納用
brightness_list = []
absorbance_list = []
brightness2_list = []
absorbance2_list = []
time_list = []
frame_count = 0

# ROI移動しきい値
prev_cx = roi_x + roi_w // 2
prev_cy = roi_y + roi_h // 2
move_threshold = 4

# matplotlibで2つの別ウィンドウを用意
plt.ion()

fig1, ax1 = plt.subplots()
line1, = ax1.plot([], [], lw=2, color='green')
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('Absorbance ROI 1')
ax1.set_title('Real-time Absorbance ROI 1 (Last 6 sec)')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

fig2, ax2 = plt.subplots()
line2, = ax2.plot([], [], lw=2, color='blue')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Absorbance ROI 2')
ax2.set_title('Real-time Absorbance ROI 2 (Last 6 sec)')
ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    success, box = tracker.update(frame)

    if success:
        cx = int(box[0] + box[2] / 2)
        cy = int(box[1] + box[3] / 2)

        # ROI移動しきい値判定
        if abs(cx - prev_cx) <= move_threshold and abs(cy - prev_cy) <= move_threshold:
            cx, cy = prev_cx, prev_cy
        else:
            prev_cx, prev_cy = cx, cy

        # ROI1の座標計算
        x1 = max(cx - roi_w // 2, 0)
        y1 = max(cy - roi_h // 2, 0)
        x1_end = min(x1 + roi_w, gray.shape[1])
        y1_end = min(y1 + roi_h, gray.shape[0])

        # ROI2はROI1の座標から左980、下2ずらす（画面範囲内に制限）
        x2 = max(x1 - 980, 0)
        y2 = min(y1 + 2, gray.shape[0] - roi_h)
        x2_end = x2 + roi_w
        y2_end = y2 + roi_h

        # ROI1輝度計算
        roi_img1 = gray[y1:y1_end, x1:x1_end]
        current_brightness1 = np.mean(roi_img1)

        # ROI2輝度計算
        roi_img2 = gray[y2:y2_end, x2:x2_end]
        current_brightness2 = np.mean(roi_img2)

        # 白色板から同じROI切り出し＆平均輝度
        white_roi1 = white_img[y1:y1_end, x1:x1_end]
        white_roi2 = white_img[y2:y2_end, x2:x2_end]
        white_brightness1 = np.mean(white_roi1)
        white_brightness2 = np.mean(white_roi2)

        # 吸光度計算
        if current_brightness1 > 0 and white_brightness1 > 0:
            reflectance1 = current_brightness1 / (white_brightness1 * 2)
            absorbance1 = np.log10(1 / reflectance1)
        else:
            absorbance1 = np.nan

        if current_brightness2 > 0 and white_brightness2 > 0:
            reflectance2 = current_brightness2 / (white_brightness2 * 2)
            absorbance2 = np.log10(1 / reflectance2)
        else:
            absorbance2 = np.nan

    else:
        current_brightness1 = np.nan
        current_brightness2 = np.nan
        absorbance1 = np.nan
        absorbance2 = np.nan

    elapsed_time = frame_count / fps
    time_list.append(elapsed_time)
    brightness_list.append(current_brightness1)
    absorbance_list.append(absorbance1)
    brightness2_list.append(current_brightness2)
    absorbance2_list.append(absorbance2)

    # 表示用フレーム描画
    display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if success:
        # ROI1
        cv2.rectangle(display_img, (x1, y1), (x1_end, y1_end), (0, 255, 0), 2)
        cv2.putText(display_img, f'Abs ROI1: {absorbance1:.3f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ROI2
        cv2.rectangle(display_img, (x2, y2), (x2_end, y2_end), (255, 0, 0), 2)
        cv2.putText(display_img, f'Abs ROI2: {absorbance2:.3f}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        cv2.putText(display_img, 'Tracking failure', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    cv2.putText(display_img, f'Time: {minutes:02d}:{seconds:02d}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Video with ROI', display_img)

    # グラフ1更新 (ROI1)
    recent_times = time_list[-max_history_frames:]
    recent_absorbance = absorbance_list[-max_history_frames:]
    line1.set_data(recent_times, recent_absorbance)
    if len(recent_times) >= 2:
        ax1.set_xlim(recent_times[0], recent_times[-1])
        valid_abs1 = [a for a in recent_absorbance if not np.isnan(a)]
        if valid_abs1:
            mean_val1 = np.mean(valid_abs1)
            margin = 0.0012
            ax1.set_ylim(mean_val1 - margin, mean_val1 + margin)
        else:
            ax1.set_ylim(0, 1)
        ax1.relim()
        ax1.autoscale_view()
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    # グラフ2更新 (ROI2)
    recent_absorbance2 = absorbance2_list[-max_history_frames:]
    line2.set_data(recent_times, recent_absorbance2)
    if len(recent_times) >= 2:
        ax2.set_xlim(recent_times[0], recent_times[-1])
        valid_abs2 = [a for a in recent_absorbance2 if not np.isnan(a)]
        if valid_abs2:
            mean_val2 = np.mean(valid_abs2)
            margin = 0.0012
            ax2.set_ylim(mean_val2 - margin, mean_val2 + margin)
        else:
            ax2.set_ylim(0, 1)
        ax2.relim()
        ax2.autoscale_view()
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

# Excel保存
df = pd.DataFrame({
    'Time (sec)': time_list,
    'Brightness ROI1': brightness_list,
    'Absorbance ROI1': absorbance_list,
    'Brightness ROI2': brightness2_list,
    'Absorbance ROI2': absorbance2_list
})
save_path = 'C:/Users/mpg/Desktop/python_rasio/absorbance_log_twoROIs.xlsx'
df.to_excel(save_path, index=False)
print(f"Excelに保存しました → {save_path}")
