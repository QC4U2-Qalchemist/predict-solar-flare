import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# 再現性のためのランダムシードの設定
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# データの読み込み
data = pd.read_csv('summarized_active_regions_features.csv')
data = data.loc[:, data.columns != 'frame_id']

labels = np.load('train_label.npy')

# 特徴量とラベルの分離
X = data[['num_of_active_regions', 'area', 'avg', 'max_gauss', 'min_gauss', 'strong_gauss', 'week_gauss', 'complexity', 'num_of_magnetic_neural_lines', 'total_length_of_magnetic_neural_lines', 'complexity_of_magnetic_neural_lines']]
y = labels

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# データを時系列形式に変換
# ここでは仮にシーケンス長を10とする
sequence_length = 10
def create_sequences(data, labels, sequence_length):
    sequences = []
    labels_seq = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        labels_seq.append(labels[i+sequence_length])
    return np.array(sequences), np.array(labels_seq)

X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.1, random_state=seed)

# モデルの構築
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(sequence_length, X_train.shape[2]), return_sequences=True),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# モデルのコンパイル
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# モデルのトレーニング
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.1)

# 予測
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# 混同行列の計算
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# TSSの計算
TSS = tp / (tp + fn) - fp / (fp + tn)

# 結果の表示
print("Accuracy:", accuracy_score(y_test, y_pred))
print("True Skill Statistic (TSS):", TSS)
