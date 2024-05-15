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

    # 特徴量を保存
    f_ext.save_features('features.npy')

    # Active regionsのみの太陽球データ保存
    f_ext.save_only_active_regions_image('only_active_regions.npy')

    # 円柱ノイズ除去映像npy出力
    if save_circumferential_denoising_npy:
        f_ext.save_circumferential_denoising('train_mag_circumferential_denoised.npy')

        #denoized = np.load('train_mag_circumferential_denoised.npy')
        #print('denoized',denoized)

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
    parser.add_argument('--save-circumferential-denoising-npy', action='store_true',default=False)
    parser.add_argument('--save-active-regions', action='store_true',default=False)

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
