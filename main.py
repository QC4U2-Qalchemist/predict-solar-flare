import argparse
import sys
import numpy as np
import cv2
from tqdm import tqdm

from features_extractor import FeaturesExtractor
#from plot_features import FeaturesPlotor

pause = False
def key_handler(wait_in_ms):
    global pause

    key = cv2.waitKey(wait_in_ms)
    if key == ord(' '):  # スペースキーで一時停止・再生
        pause = not pause
    while pause:
        key = cv2.waitKey(100)
        if key == ord(' '):
            pause = False
            break

def show_imgs(imgs, titles, wait_in_ms):
    for img, title in zip(imgs, titles):
        cv2.imshow(title, img)

    key_handler(wait_in_ms)

def find_min_max_coordinates(frame):
    # 0でないピクセルの座標を取得
    non_zero_coords = cv2.findNonZero(frame)
    
    if non_zero_coords is not None:
        # X軸とY軸で座標を分割
        x_coords = non_zero_coords[:,:,0]
        y_coords = non_zero_coords[:,:,1]
        
        # X軸とY軸の最小値と最大値を求める
        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)
        
        return (min_x, max_x, min_y, max_y)
    else:
        return None

def mask_outside_circle(frame, circumference_width=8):
    # 0でないピクセルの座標の最小値と最大値を求める
    bounds = find_min_max_coordinates(frame)
    if bounds is None:
        return None
    
    min_x, max_x, min_y, max_y = bounds
    
    # 円の中心座標を計算
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # 円の半径を計算
    radius = max(max_x - center_x, max_y - center_y) - circumference_width
    
    # 出力用の画像を準備
    masked_image = np.zeros_like(frame)
    
    # マスク処理
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            if np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) > radius:
                masked_image[y, x] = 0
            else:
                masked_image[y, x] = frame[y, x]
    
    return masked_image

#DEBUG=True
DEBUG=False
def main(line_of_sight_mag_filepath, label_solar_flare_filepath, out_dir, gauss_thresh, is_show_imgs, save_circumferential_denoising_npy, save_active_regions):
    global DEBUG
    # Load Data
    mag = np.load(line_of_sight_mag_filepath)
    label = np.load(label_solar_flare_filepath)

    # Check Data Shape and type
    print(mag.shape)
    print(label.shape)
    print(type(mag.shape))
    print(type(label.shape))

    f_ext = FeaturesExtractor(out_dir=out_dir, gauss_thresh=gauss_thresh, is_show_imgs=is_show_imgs, is_save_circumferential_denoising_npy=save_circumferential_denoising_npy, is_save_active_regions=save_active_regions)
    features = []
    for i in tqdm(range(mag.shape[2])):

        if DEBUG and i >= 5:
            break

        # i日目の画像を左右反転して表示
        frame = mag[:, ::-1, i].copy()


        # 特徴量抽出
        f_ext.append(i, frame)
        #features.append([fext.get_subject_clarity(frame),fext.get_subject_clarity(masked_frame),fext.get_subject_clarity(high_strength_gauss)])

        # 各種加工画像を描画する
        #if is_show_imgs:
        #    show_imgs([frame,masked_frame,high_strength_gauss], ['frame','masked','high strength gauss'], 100)


    # 特徴量を保存
    f_ext.save_features('features.npy')

    # 円柱ノイズ除去映像npy出力
    f_ext.save_circumferential_denoising('train_mag_circumferential_denoised.npy')

    denoized = np.load('train_mag_circumferential_denoised.npy')
    print('denoized',denoized)

    # 特徴量プロット
    if DEBUG:
        f_ext.plot(label[:i])
    else:
        f_ext.plot(label)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--line-of-sight-mag-filepath', default='train_mag.npy', help='npy file path for line-of-sight magnetogram')
    parser.add_argument('--label-solar-flare-filepath', default='train_label.npy', help='npy file path for label of solar flare')
    parser.add_argument('--out-dir', default='active_regions', help='output directory')
    parser.add_argument('--gauss-thresh', default=200, type=int, help='Threshold of Gauss')
    parser.add_argument('--show-imgs', action='store_true',default=False)
    parser.add_argument('--save-circumferential-denoising-npy', action='store_true',default=True)
    parser.add_argument('--save-active-regions', action='store_true',default=True)

    args = parser.parse_args()
    print('args',args)
    line_of_sight_mag_filepath = args.line_of_sight_mag_filepath
    label_solar_flare_filepath = args.label_solar_flare_filepath
    gauss_thresh = args.gauss_thresh
    is_show_imgs = args.show_imgs
    out_dir = args.out_dir
    save_circumferential_denoising_npy = args.save_circumferential_denoising_npy
    save_active_regions = args.save_active_regions
    sys.exit(main(line_of_sight_mag_filepath, label_solar_flare_filepath, out_dir, gauss_thresh, is_show_imgs, save_circumferential_denoising_npy, save_active_regions))
