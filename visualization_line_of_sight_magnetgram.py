import numpy as np
import cv2
import os

# Load Data
data_mag = np.load('train_mag_0_499.npy')
label = np.load('train_label_0_499.npy')

# Check Data Shape
print(data_mag.shape)
print(label.shape)

# カウントダウンを開始するフレアのインデックスを特定
flare_indices = [idx for idx, val in enumerate(label) if val == 1]
print("Flare indices:", flare_indices)

# OpenCVウィンドウの準備
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

i = 0
pause = False
while i < data_mag.shape[2]:
    frame = data_mag[:, ::-1, i].copy()  # フレームをコピーして操作

    # カウントダウン表示
    if any(idx - 10 <= i <= idx for idx in flare_indices):
        next_flare_index = min(idx for idx in flare_indices if idx - 10 <= i <= idx)
        countdown = next_flare_index - i
        if countdown >= 0:  # マイナスになる場合は表示しない
            cv2.putText(frame, f'Countdown: {countdown}', (1, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'frame: {i}', (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)


    #print('label[i]',label[i])
    # キーボード入力に応じた制御
    key = cv2.waitKey(1000)  # おおよそ30fpsの速度で再生
    if key == ord(' '):  # スペースキーで一時停止・再生
        pause = not pause
    while pause:
        key = cv2.waitKey(100)
        if key == ord(' '):
            pause = False
        elif key == ord('g'):  # 'g' キーで10コマ早送り
            i += 10
            break
        elif key == ord('f'):  # 'f' キーで1コマ早送り
            i += 1
            break
        elif key == ord('d'):  # 'd' キーで1コマ巻き戻し
            i = max(0, i - 1)
            break
        elif key == ord('s'):  # 'd' キーで10コマ巻き戻し
            i = max(0, i - 10)
            break

    if not pause:
        if key == ord('g'):  # 'g' キーで10コマ早送り
            i += 10
        elif key == ord('f'):  # 'f' キーで1コマ早送り
            i += 1
        elif key == ord('d'):  # 'd' キーで1コマ巻き戻し
            i = max(0, i - 1)
        elif key == ord('s'):  # 'd' キーで10コマ巻き戻し
            i = max(0, i - 10)
        else:
            i += 1

cv2.destroyAllWindows()
