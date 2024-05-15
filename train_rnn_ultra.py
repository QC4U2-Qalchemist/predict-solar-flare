import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

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

# カスタムコールバックの作成
class TSSCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.tss = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        TSS = tp / (tp + fn) - fp / (fp + tn)
        logs['val_tss'] = TSS  # TSSをログに追加
        self.tss.append(TSS)
        print(f'Epoch {epoch+1}: TSS = {TSS:.4f}')

tss_callback = TSSCallback()

# モデルの構築
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(sequence_length, X_train.shape[2]), return_sequences=True),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# モデルの構築（LSTMレイヤーと全結合層を追加）
#model = tf.keras.Sequential([
#    tf.keras.layers.LSTM(64, input_shape=(sequence_length, X_train.shape[2]), return_sequences=True),
#    tf.keras.layers.LSTM(64, return_sequences=True),
#    tf.keras.layers.LSTM(64),
#    tf.keras.layers.Dense(32, activation='relu'),
#    tf.keras.layers.Dense(16, activation='relu'),
#    tf.keras.layers.Dense(1, activation='sigmoid')
#])

# モデルのコンパイル
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# モデルチェックポイントの作成
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_tss', save_best_only=True, mode='max', verbose=1)

# モデルのトレーニング
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.1, callbacks=[tss_callback, checkpoint])

# ベストモデルのロード
best_model = tf.keras.models.load_model('best_model.h5', custom_objects={'tss_metric': tss_callback})

# ベストモデルで予測とTSSの計算
y_pred_prob = best_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
best_TSS = tp / (tp + fn) - fp / (fp + tn)
print("Best TSS from saved model:", best_TSS)

# 最終モデルの予測とTSSの計算
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
final_TSS = tp / (tp + fn) - fp / (fp + tn)
print("Final Accuracy:", accuracy_score(y_test, y_pred))
print("Final True Skill Statistic (TSS):", final_TSS)

# Accuracy, Val_accuracy, TSSの推移をプロット
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(tss_callback.tss, label='TSS')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.show()
