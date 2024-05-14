import argparse
import sys
import numpy as np
import cv2
from tqdm import tqdm
import re
import os



def list_npy_files_recursively(directory):
    """
    指定されたディレクトリおよびそのサブディレクトリ内の全ての.npyファイルのリストを返す関数。
    
    Args:
    directory (str): 検索するディレクトリのパス。
    
    Returns:
    list: .npyファイルのパスのリスト。
    """
    npy_files = []
    # os.walkでディレクトリを再帰的に走査
    for dirpath, dirnames, filenames in os.walk(directory):
        # .npyファイルをフィルタリングし、リストに追加
        for filename in filenames:
            if filename.endswith('.npy'):
                # ファイルの完全なパスをリストに追加
                full_path = os.path.join(dirpath, filename)
                npy_files.append(full_path)
    
    return npy_files

def extract_values_from_filename(filename):
    # 正規表現パターンを定義（各パラメータの数値をキャプチャ）
    pattern = r'idx(\d+)_x(\d+)_y(\d+)_w(\d+)_h(\d+)\.npy'
    
    # 正規表現でマッチングを実行
    match = re.search(pattern, filename)
    if match:
        # マッチングした各グループの値を取得
        idx, x, y, w, h = match.groups()
        return int(idx), int(x), int(y), int(w), int(h)
    else:
        # マッチング失敗の場合はエラーを表示
        raise ValueError("ファイル名が正しい形式ではありません。")

def main(active_regions_dir):

    active_region_npy_files = list_npy_files_recursively(active_regions_dir)
    #print(active_region_npy_files)

    for active_region_file in active_region_npy_files:
        basename = os.path.basename(active_region_file)
        print('basename',basename)

        idx, centor_x, centor_y, width, height = extract_values_from_filename(basename)
        print('idx, centor_x, centor_y, width, height',idx, centor_x, centor_y, width, height)

        # Load Data
        active_region = np.load(active_region_file)
        cv2.imshow('active region',active_region)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--active-regions-dir', default='active_regions', help='input dir which include active regions npy')

    args = parser.parse_args()
    print('args',args)
    active_regions_dir = args.active_regions_dir
    sys.exit(main(active_regions_dir))
