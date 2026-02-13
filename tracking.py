
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import pandas as pd

# video_path = 'C:/Users/mpg/Desktop/python_rasio/hirohata.mp4'
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("動画が開けません")
#     exit()

# fps = cap.get(cv2.CAP_PROP_FPS)
# max_history_frames = int(fps * 10)  # 直近10秒分

# ret, frame = cap.read()
# if not ret:
#     print("最初のフレームが読み込めません")
#     exit()

# roi = cv2.selectROI('Select ROI', frame, showCrosshair=True, fromCenter=False)
# cv2.destroyWindow('Select ROI')

# roi_x, roi_y, roi_w, roi_h = map(int, roi)
# tracker = cv2.TrackerCSRT_create()
# tracker.init(frame, (roi_x, roi_y, roi_w, roi_h))

# brightness_list = []
# time_list = []
# frame_count = 0

# # ROI中心の初期値
# prev_cx = roi_x + roi_w // 2
# prev_cy = roi_y + roi_h // 2
# move_threshold = 4 # ROIを動かすしきい値（px）########################################################

# plt.ion()
# fig, ax = plt.subplots()
# line, = ax.plot([], [], lw=2)
# ax.set_xlabel('Time (sec)')
# ax.set_ylabel('Brightness')
# ax.set_title('Real-time Brightness (Last 10 sec)')
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # 横軸1秒刻み

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_count += 1

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
#     success, box = tracker.update(frame)

#     if success:
#         cx = int(box[0] + box[2] / 2)
#         cy = int(box[1] + box[3] / 2)

#         # 前のROI中心と比較して移動が小さい場合、更新しない
#         if abs(cx - prev_cx) <= move_threshold and abs(cy - prev_cy) <= move_threshold:
#             cx, cy = prev_cx, prev_cy
#         else:
#             prev_cx, prev_cy = cx, cy  # 明確に動いたら更新

#         x = max(cx - roi_w // 2, 0)
#         y = max(cy - roi_h // 2, 0)
#         x_end = min(x + roi_w, gray.shape[1])
#         y_end = min(y + roi_h, gray.shape[0])
#         roi_img = gray[y:y_end, x:x_end]
#         mean_brightness = np.mean(roi_img)##
#     else:
#         mean_brightness = np.nan

#     elapsed_time = frame_count / fps
#     brightness_list.append(mean_brightness)
#     time_list.append(elapsed_time)

#     display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#     if success:
#         cv2.rectangle(display_img, (x, y), (x_end, y_end), (0, 255, 0), 2)
#         cv2.putText(display_img, f'Brightness: {mean_brightness:.2f}', (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     else:
#         cv2.putText(display_img, 'Tracking failure', (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     minutes = int(elapsed_time // 60)
#     seconds = int(elapsed_time % 60)
#     cv2.putText(display_img, f'Time: {minutes:02d}:{seconds:02d}', (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#     cv2.imshow('Video with ROI', display_img)

#     recent_times = time_list[-max_history_frames:]
#     recent_brightness = brightness_list[-max_history_frames:]

#     line.set_data(recent_times, recent_brightness)

#     if len(recent_times) >= 2:
#         ax.set_xlim(recent_times[0], recent_times[-1])

#         valid_brightness = [b for b in recent_brightness if b is not None and not np.isnan(b)]
#         if valid_brightness:
#             mean_val = np.mean(valid_brightness)
#             margin = 2
#             ymin = max(mean_val - margin, 0)
#             ymax = min(mean_val + margin, 255)
#             ax.set_ylim(ymin, ymax)
#         else:
#             ax.set_ylim(0, 255)

#         ax.relim()
#         ax.autoscale_view()

#     fig.canvas.draw()
#     fig.canvas.flush_events()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# plt.ioff()
# plt.show()

# df = pd.DataFrame({
#     'Time (sec)': time_list,
#     'Brightness': brightness_list
# })
# df.to_excel('C:/Users/mpg/Desktop/python_rasio/brightness_log.xlsx', index=False)
# print("Excelに保存しました → brightness_log_thresholded_roi.xlsx")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# 動画と白色板画像のパス
video_path = 'C:/Users/mpg/Desktop/python_rasio/hirohata.mp4'
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
max_history_frames = int(fps * 6)

# 最初のフレーム取得とROI選択
ret, frame = cap.read()
if not ret:
    print("最初のフレームが読み込めません")
    exit()

roi = cv2.selectROI('Select ROI', frame, showCrosshair=True, fromCenter=False)
cv2.destroyWindow('Select ROI')

roi_x, roi_y, roi_w, roi_h = map(int, roi)

# トラッカー初期化
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, (roi_x, roi_y, roi_w, roi_h))

# データ格納用
brightness_list = []
absorbance_list = []
white_brightness_list = []
time_list = []
frame_count = 0

# ROI移動しきい値
prev_cx = roi_x + roi_w // 2
prev_cy = roi_y + roi_h // 2
move_threshold = 4

# プロット初期化
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Absorbance')
ax.set_title('Real-time Absorbance (Last 10 sec)')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

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

        if abs(cx - prev_cx) <= move_threshold and abs(cy - prev_cy) <= move_threshold:
            cx, cy = prev_cx, prev_cy
        else:
            prev_cx, prev_cy = cx, cy

        x = max(cx - roi_w // 2, 0)
        y = max(cy - roi_h // 2, 0)
        x_end = min(x + roi_w, gray.shape[1])
        y_end = min(y + roi_h, gray.shape[0])

        # 対象ROIから平均輝度
        roi_img = gray[y:y_end, x:x_end]
        current_brightness = np.mean(roi_img)

        # 白色板画像の同じROIを切り出し
        white_roi_img = white_img[y:y_end, x:x_end]
        white_brightness = np.mean(white_roi_img)

        white_brightness_list.append(white_brightness)

        # 吸光度計算
        if current_brightness > 0 and white_brightness > 0:
            reflectance = current_brightness / (white_brightness * 2)
            absorbance = np.log10(1 / reflectance)
        else:
            absorbance = np.nan
    else:
        current_brightness = np.nan
        white_brightness = np.nan
        absorbance = np.nan

    elapsed_time = frame_count / fps
    time_list.append(elapsed_time)
    brightness_list.append(current_brightness)
    absorbance_list.append(absorbance)

    # 表示用フレーム
    display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if success:
        cv2.rectangle(display_img, (x, y), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(display_img, f'Abs: {absorbance:.3f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_img, 'Tracking failure', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    cv2.putText(display_img, f'Time: {minutes:02d}:{seconds:02d}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Video with ROI', display_img)

    # グラフ更新
    recent_times = time_list[-max_history_frames:]
    recent_absorbance = absorbance_list[-max_history_frames:]
    line.set_data(recent_times, recent_absorbance)

    if len(recent_times) >= 2:
        ax.set_xlim(recent_times[0], recent_times[-1])

        valid_absorbance = [a for a in recent_absorbance if not np.isnan(a)]
        if valid_absorbance:
            mean_val = np.mean(valid_absorbance)
            margin = 0.008
            ax.set_ylim(mean_val - margin, mean_val + margin)
        else:
            ax.set_ylim(0, 1)

        ax.relim()
        ax.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

# Excel出力
df = pd.DataFrame({
    'Time (sec)': time_list,
    'Brightness': brightness_list,
    'White Brightness': white_brightness_list,
    'Absorbance': absorbance_list
})
save_path = 'C:/Users/mpg/Desktop/python_rasio/absorbance_log_tracking_whiteROI.xlsx'
df.to_excel(save_path, index=False)
print(f"Excelに保存しました → {save_path}")
