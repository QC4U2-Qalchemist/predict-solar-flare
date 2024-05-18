import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from keras_tuner import HyperModel, RandomSearch, Objective

# 再現性のためのランダムシードの設定
seed = 659
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
sequence_length = 27
def create_sequences(data, labels, sequence_length):
    sequences = []
    labels_seq = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        labels_seq.append(labels[i+sequence_length])
    return np.array(sequences), np.array(labels_seq)

X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)

# 時系列データの前半90%を訓練データ、後半10%をテストデータとして分割
train_size = int(len(X_seq) * 0.9)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# クラスの分布を確認してクラス重みを計算
#class_weights = {0: (1 / np.sum(y_train == 0)) * (len(y_train) / 2.0),
#                 1: (1 / np.sum(y_train == 1)) * (len(y_train) / 2.0)}

# カスタムTSSメトリックの定義
def tss_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.int32)
    tn = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 0), tf.float32))
    fp = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 1), tf.float32))
    fn = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 0), tf.float32))
    tp = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32))
    tss = tp / (tp + fn) - fp / (fp + tn)
    return tss

# ハイパーモデルの定義
class RNNHyperModel(HyperModel):
    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32),
                                       input_shape=(sequence_length, X_train.shape[2]),
                                       return_sequences=True))
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(tf.keras.layers.LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=128, step=32),
                                           return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 else False))
        for i in range(hp.Int('num_dense_layers', 1, 3)):
            model.add(tf.keras.layers.Dense(units=hp.Int('dense_units_' + str(i), min_value=16, max_value=128, step=16),
                                            activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tss_metric])
        return model

# Keras Tuner のインスタンスを作成
tuner = RandomSearch(
    RNNHyperModel(),
    objective=Objective('val_tss_metric', direction='max'),
    max_trials=10000,
    executions_per_trial=1,
    directory='my_dir',  # ディレクトリ名を指定
    project_name='rnn_tuning'
)

# 早期停止のコールバック
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_tss_metric', patience=10, restore_best_weights=True)
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_tss_metric', patience=10, restore_best_weights=True, mode='max')

# チューニングの実行
#tuner.search(X_train, y_train, epochs=50, validation_split=0.1, class_weight=class_weights, callbacks=[early_stopping])
#tuner.search(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[early_stopping])
tuner.search(X_train, y_train, epochs=50, validation_split=0.1)

# チューニング結果の表示
tuner.results_summary()

# 最適なハイパーパラメータの取得
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")

# 最適なモデルの取得
best_model = tuner.get_best_models(num_models=1)[0]

# モデルの可視化
from tensorflow.keras.utils import plot_model
plot_model(best_model, to_file='best_model.png', show_shapes=True, show_layer_names=True)

# 訓練データと検証データの損失と精度のプロット
#history = best_model.fit(X_train, y_train, epochs=50, validation_split=0.1, class_weight=class_weights, callbacks=[early_stopping])
#history = best_model.fit(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[early_stopping])
history = best_model.fit(X_train, y_train, epochs=50, validation_split=0.1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['tss_metric'], label='Train TSS')
plt.plot(history.history['val_tss_metric'], label='Validation TSS')
plt.xlabel('Epoch')
plt.ylabel('TSS')
plt.legend()
plt.show()

# テストデータでの評価
test_data = pd.read_csv('test_summarized_active_regions_features.csv')
test_data = test_data.loc[:, test_data.columns != 'frame_id']
test_labels = np.load('test_label.npy')
test_data_scaled = scaler.transform(test_data)
X_test_final, y_test_final = create_sequences(test_data_scaled, test_labels, sequence_length)

# ベストモデルでテストデータを予測
y_test_pred_prob = best_model.predict(X_test_final)
y_test_pred = (y_test_pred_prob > 0.5).astype(int)

# テストデータの評価
tn, fp, fn, tp = confusion_matrix(y_test_final, y_test_pred).ravel()
test_TSS = tp / (tp + fn) - fp / (fp + tn)
test_accuracy = accuracy_score(y_test_final, y_test_pred)
test_precision = precision_score(y_test_final, y_test_pred)
test_recall = recall_score(y_test_final, y_test_pred)
test_f1 = f1_score(y_test_final, y_test_pred)
print("Test Confusion matrix", confusion_matrix(y_test_final, y_test_pred))
print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)
print("Test True Skill Statistic (TSS):", test_TSS)

