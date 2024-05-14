import argparse
import sys
import numpy as np
import cv2
from tqdm import tqdm
import re
import os

from read_active_regions import list_npy_files_recursively, extract_values_from_filename

def extract_features(active_region):
    #cv2.imshow('active region',active_region)
    #cv2.waitKey(0)
    return 0,0,0

class ExtractActiveRegionFeatures:

    def __init__(self, gauss_thresh=200):
        self.gauss_thresh = gauss_thresh
        return

    def get_magnetic_neural_line(self, image):

        cv2.imshow('image',image)

        scaled_image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # ガウシアンブラーを適用して画像のノイズを低減
        blurred = cv2.GaussianBlur(scaled_image, (5, 5), 0)

        # Cannyエッジ検出器を適用（グレースケール変換もここで行う）
        #gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 高い閾値を設定して勾配の大きいエッジのみを抽出
        high_threshold = 200  # この値を調整することで勾配の大きいエッジのみ抽出

        # Cannyエッジ検出器を適用
        return cv2.Canny(blurred, high_threshold, high_threshold * 2)

        
        
    def get_features_of_magnetic_neural_line(self, image):

        scaled_image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # ガウシアンブラーを適用して画像のノイズを低減
        blurred = cv2.GaussianBlur(scaled_image, (5, 5), 0)

        cv2.imshow('blurred',blurred)

        # 高い閾値を設定して勾配の大きいエッジのみを抽出
        high_threshold = 250  # この値を調整することで勾配の大きいエッジのみ抽出

        # Cannyエッジ検出器を適用
        edge = cv2.Canny(blurred, high_threshold, high_threshold * 2)


        # findContoursを使用してエッジの輪郭を見つける
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 各輪郭の面積を計算
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        #print("Areas of detected contours:", contour_areas)

        if False:
            cv2.imshow('edge', edge)
            # 表示用
            color = cv2.cvtColor(np.clip(blurred * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # 元の画像に輪郭を描画（緑色で、線の太さは2）
            cv2.drawContours(color, contours, -1, (0, 255, 0), 1)

            cv2.imshow('contours',color)
            cv2.waitKey(0)

        num_of_magnetic_neural_lines = len(contours)
        total_length_of_magnetic_neural_lines = np.sum(contour_areas)
        complexity_of_magnetic_neural_lines = self.get_complexity(edge)

        return [num_of_magnetic_neural_lines, total_length_of_magnetic_neural_lines, complexity_of_magnetic_neural_lines]


    def get_complexity(self, image):
        scaled_image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # Cannyエッジ検出を適用
        edges = cv2.Canny(scaled_image, 100, 200)

        # エッジのピクセル数を計算
        edges_pixels = np.sum(edges == 255)

        # 全ピクセル数
        total_pixels = image.size

        # 複雑度の計算（エッジの割合）
        complexity = edges_pixels / total_pixels

        return complexity

    def get_features(self, image):
        avg = np.mean(image)
        std = np.std(image) 
        max_gauss = np.max(image)
        min_gauss = np.min(image)
        strong_gauss = np.sum(image >= self.gauss_thresh)
        week_gauss = np.sum(-image >= -self.gauss_thresh)
        complexity = self.get_complexity(image)
        return [avg, std, max_gauss, min_gauss, strong_gauss, week_gauss, complexity] + self.get_features_of_magnetic_neural_line(image)

def main(active_regions_dir):
    ext_ar_features = ExtractActiveRegionFeatures(gauss_thresh=200)
    features_of_active_regions = {}
    active_region_npy_files = list_npy_files_recursively(active_regions_dir)

    for active_region_file in active_region_npy_files:
        basename = os.path.basename(active_region_file)
        
        print('processing',basename)

        # 座標データなどをファイル名から抽出
        idx, centor_x, centor_y, width, height = extract_values_from_filename(basename)
        #print('idx, centor_x, centor_y, width, height',idx, centor_x, centor_y, width, height)

        # Load Data
        active_region = np.load(active_region_file)

        key='{:05}'.format(idx)
        if idx not in features_of_active_regions:
            features_of_active_regions[key]=[]

        features_of_active_regions[key].append([idx, centor_x, centor_y, width, height, width * height] + ext_ar_features.get_features(active_region))


    print(features_of_active_regions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--active-regions-dir', default='active_regions', help='input dir which include active regions npy')

    args = parser.parse_args()
    print('args',args)
    active_regions_dir = args.active_regions_dir
    sys.exit(main(active_regions_dir))
