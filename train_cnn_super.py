import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import backend as K

def tss_metric(y_true, y_pred):
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    
    tp = K.sum(K.cast(y_true * y_pred, 'float32'))
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'))
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'))
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float32'))
    
    tss = tp / (tp + fn + K.epsilon()) - fp / (fp + tn + K.epsilon())
    return tss

def custom_loss(y_true, y_pred):
    tss = tss_metric(y_true, y_pred)
    return 1 - tss  # TSSを最大化するために損失としては1-TSSを使用

# データの読み込み
data = pd.read_csv('summarized_active_regions_features.csv')
data = data.loc[:, data.columns != 'frame_id']

labels = np.load('train_label.npy')



def train(seed):
    # 再現性のためのランダムシードの設定
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # 特徴量とラベルの分離
    X = data[['num_of_active_regions', 'area', 'avg', 'max_gauss', 'min_gauss', 'strong_gauss', 'week_gauss', 'complexity', 'num_of_magnetic_neural_lines', 'total_length_of_magnetic_neural_lines', 'complexity_of_magnetic_neural_lines']]
    y = labels

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # データの標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # モデルの構築
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # モデルのコンパイル
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tss_metric])

    # モデルのトレーニング
    history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, validation_split=0.1)

    # 予測
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # TSSの計算
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    TSS = tp / (tp + fn) - fp / (fp + tn)

    # 結果の表示
    print(seed, "Accuracy:", accuracy_score(y_test, y_pred))
    print(seed, "True Skill Statistic (TSS):", TSS)

    # Epoch vs Accuracy
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['val_tss_metric'])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

train(22)
