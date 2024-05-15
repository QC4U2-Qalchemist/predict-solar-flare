import pandas as pd
import argparse
import sys



def main(input_file, output_file):
    # CSVファイルの読み込み
    df = pd.read_csv(input_file)

    # サマライズの処理
    summary_df = df.groupby('frame_id').agg(
        num_of_active_regions=('frame_id', 'size'),
        area=('area', 'sum'),
        avg=('area', lambda x: (df.loc[x.index, 'avg'] * x).sum() / x.sum()),
        max_gauss=('max_gauss', 'max'),
        min_gauss=('min_gauss', 'min'),
        strong_gauss=('strong_gauss', 'sum'),
        week_gauss=('week_gauss', 'sum'),
        complexity=('complexity', 'sum'),
        num_of_magnetic_neural_lines=('num_of_magnetic_neural_lines', 'sum'),
        total_length_of_magnetic_neural_lines=('total_length_of_magnetic_neural_lines', 'sum'),
        complexity_of_magnetic_neural_lines=('complexity_of_magnetic_neural_lines', 'sum')
    ).reset_index()

    # 新しいCSVファイルとして出力
    output_file = 'summarized_active_regions_features.csv'
    summary_df.to_csv(output_file, index=False)

    print(f'Summarized CSV file has been saved to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--features-active-regions-csv-filepath', default='active_regions_features.csv', help='csv file path for features of active regions')
    parser.add_argument('--out-csv-filepath', default='summarize_active_regions_features.csv', help='output csv path')

    args = parser.parse_args()
    print('args',args)

    features_active_regions_csv_filepath = args.features_active_regions_csv_filepath
    out_csv_filepath = args.out_csv_filepath
    sys.exit(main(features_active_regions_csv_filepath, out_csv_filepath))
