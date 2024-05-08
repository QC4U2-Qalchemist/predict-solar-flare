import numpy as np
import cv2
import os
from PIL import Image, ImageStat
from skimage import feature, filters
import matplotlib.pyplot as plt

def get_resolution(image):
    return image.shape[0] * image.shape[1]

def get_brightness(image):
    img = Image.fromarray(image)
    stat = ImageStat.Stat(img)
    return stat.mean[0]

def get_contrast(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_gray.std()

def get_subject_clarity(image):
    scaled_image = np.clip(image * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(scaled_image, 100, 200)
    return np.mean(edges)

def get_background_complexity(image):
    scaled_image = np.clip(image * 255, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    entropy = filters.rank.entropy(gray, np.ones((5, 5)))
    return np.mean(entropy)

def get_motion_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var


# Load Data
data_mag = np.load('train_mag.npy')
label = np.load('train_label.npy')

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
gauss_thres=140
while i < data_mag.shape[2]:
    frame = data_mag[:, ::-1, i].copy()  # フレームをコピーして操作
    large_mag = np.where(frame >= gauss_thres, frame, 0)
    large_mag_pixels = np.sum(np.where(frame >= gauss_thres, 1, 0))
    small_mag = np.where(frame < -gauss_thres, frame, 0)
    zero_mag = np.where((frame > -1) & (frame < 1), frame, 0)
    #zero_mag = np.where(frame == 0, frame, 0)
    scaled_image = np.clip(frame * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(scaled_image, 190, 200)
    scaled_image2 = np.clip(large_mag * 255, 0, 255).astype(np.uint8)
    edges2 = cv2.Canny(scaled_image2, 190, 200)

    #print('frame.shape',frame.shape)
    #print('type(frame)',type(frame))
    cv2.imshow('large_mag', large_mag)
    cv2.imshow('scaled_image', scaled_image)
    cv2.imshow('edges', edges)
    cv2.imshow('scaled_image2', scaled_image2)
    cv2.imshow('edges2', edges2)
    #cv2.imshow('small_mag', small_mag)
    #print(zero_mag)
    cv2.imshow('zero_mag', zero_mag)

    resolution = get_resolution(frame)
    brightness = get_brightness(frame)
    #contrast = get_contrast(frame)
    subject_clarity = get_subject_clarity(frame)
    subject_clarity_large_mag = get_subject_clarity(large_mag)
    print('subject_clarity',subject_clarity,subject_clarity_large_mag,large_mag_pixels,label[i],)
    #background_complexity = get_background_complexity(frame)
    #print('background_complexity',background_complexity)
    #motion_blur = get_motion_blur(frame)
    #print('',resolution,brightness,contrast,subject_clarity,background_complexity,motion_blur)





    # カウントダウン表示
    if any(idx - 10 <= i <= idx for idx in flare_indices):
        next_flare_index = min(idx for idx in flare_indices if idx - 10 <= i <= idx)
        countdown = next_flare_index - i
        if countdown >= 0:  # マイナスになる場合は表示しない
            cv2.putText(frame, f'Countdown: {countdown}', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)

    #print('label[i]',label[i])
    # キーボード入力に応じた制御
    key = cv2.waitKey(1)  # おおよそ30fpsの速度で再生
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
