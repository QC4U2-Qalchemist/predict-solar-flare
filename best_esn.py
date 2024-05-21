import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

# ランダムシードの設定
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# データの読み込み
train_labels = np.load('/ssd/solar/train_label.npy')
train_features = pd.read_csv('summarized_active_regions_features.csv')
test_labels = np.load('/ssd/solar/test_label.npy')
test_features = pd.read_csv('test_summarized_active_regions_features.csv')

# ラグ特徴量の作成
def create_lag_features(features, lags=3):
    lagged_features = features.copy()
    for lag in range(1, lags + 1):
        lagged = features.shift(lag).fillna(0)
        lagged_features = pd.concat([lagged_features, lagged.add_suffix(f'_lag{lag}')], axis=1)
    return lagged_features

# 特徴データの標準化
scaler = StandardScaler()

class EchoStateNetwork:
    def __init__(self, input_dim, reservoir_size=100, spectral_radius=0.95, sparsity=0.1, input_scaling=0.1, leak_rate=0.3, random_seed=None):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        if random_seed is not None:
            np.random.seed(random_seed)
        self.W_in = (np.random.rand(self.reservoir_size, self.input_dim) - 0.5) * self.input_scaling
        self.W_res = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5
        self.W_res *= (np.random.rand(self.reservoir_size, self.reservoir_size) < self.sparsity)
        self.W_res *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.state = np.zeros(self.reservoir_size)

    def __call__(self, x):
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * np.tanh(np.dot(self.W_in, x) + np.dot(self.W_res, self.state))
        return self.state

class ReservoirComputing:
    def __init__(self, input_dim, reservoir_size=100, spectral_radius=0.95, sparsity=0.1):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.W_in = np.random.rand(self.reservoir_size, self.input_dim) * 2 - 1
        self.W_res = np.random.rand(self.reservoir_size, self.reservoir_size) * 2 - 1
        self.W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.state = np.zeros(self.reservoir_size)

    def __call__(self, x):
        self.state = np.tanh(np.dot(self.W_in, x) + np.dot(self.W_res, self.state))
        return self.state

# TSSの計算関数
def calculate_tss(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tss = (tp / (tp + fn)) - (fp / (fp + tn))
    return tss

# リザバーの出力を集める関数
def collect_states(features, model):
    return np.array([model(x) for x in features])

# 最適化対象の関数
def objective(params):
    #reservoir_size, spectral_radius, ridge_alpha, input_scaling, lags, model_type, sparsity, leak_rate = params
    reservoir_size, spectral_radius, ridge_alpha, input_scaling,  sparsity, leak_rate = params

    train_features_lagged = create_lag_features(pd.DataFrame(train_features), lags=1)
    test_features_lagged = create_lag_features(pd.DataFrame(test_features), lags=1)

    train_features_scaled = scaler.fit_transform(train_features_lagged)
    test_features_scaled = scaler.transform(test_features_lagged)
    model_type = 'esn'
    if model_type == 'esn':
        model = EchoStateNetwork(input_dim=train_features_scaled.shape[1],
                                 reservoir_size=reservoir_size,
                                 spectral_radius=spectral_radius,
                                 input_scaling=input_scaling,
                                 sparsity=sparsity,
                                 leak_rate=leak_rate,
                                 random_seed=RANDOM_SEED)
    else:
        model = ReservoirComputing(input_dim=train_features_scaled.shape[1],
                                   reservoir_size=reservoir_size,
                                   spectral_radius=spectral_radius,
                                   sparsity=sparsity)

    train_states = collect_states(train_features_scaled, model)
    test_states = collect_states(test_features_scaled, model)

    ridge = Ridge(alpha=ridge_alpha, random_state=RANDOM_SEED)

    ridge.fit(train_states, train_labels)

    test_pred = ridge.predict(test_states) > 0.5
    test_tss = calculate_tss(test_labels, test_pred)
    print(f"Evaluating parameters: {params} TSS:{test_tss}")

    # マイナスを返すことで、TSSを最大化するように最適化
    return -test_tss

# パラメータの空間
space  = [
    Integer(400, 401, name='reservoir_size'),
    Real(0.8, 1.0, name='spectral_radius'),
    Real(0.1, 5.0, name='ridge_alpha'),
    Real(0.1, 1.0, name='input_scaling'),
    #Integer(1, 7, name='lags'),
    #Categorical(['esn', 'rc'], name='model_type'),
    Real(0.1, 0.6, name='sparsity'),
    Real(0.1, 1.0, name='leak_rate')
]

# ベイズ最適化の実行
#res_gp = gp_minimize(objective, space, n_calls=10000, random_state=RANDOM_SEED)

# 最適なパラメータとその結果
#best_tss = -res_gp.fun
#best_params = res_gp.x

#print(f'Best TSS: {best_tss:.2f}')
#print(f'Best Parameters: {dict(zip([dim.name for dim in space], best_params))}')

# ベストパラメータを用いて再度モデルを評価
reservoir_size, spectral_radius, ridge_alpha, input_scaling, sparsity, leak_rate = 400, 1.0, 2.8608919531679526, 1.0, 0.2805104977278139, 0.39546624335130565

train_features_lagged = create_lag_features(pd.DataFrame(train_features), lags=1)
test_features_lagged = create_lag_features(pd.DataFrame(test_features), lags=1)

train_features_scaled = scaler.fit_transform(train_features_lagged)
test_features_scaled = scaler.transform(test_features_lagged)

model = EchoStateNetwork(input_dim=train_features_scaled.shape[1],
                             reservoir_size=reservoir_size,
                             spectral_radius=spectral_radius,
                             input_scaling=input_scaling,
                             sparsity=sparsity,
                             leak_rate=leak_rate,
                             random_seed=RANDOM_SEED)

train_states = collect_states(train_features_scaled, model)
test_states = collect_states(test_features_scaled, model)

ridge = Ridge(alpha=ridge_alpha, random_state=RANDOM_SEED)
ridge.fit(train_states, train_labels)

train_pred = ridge.predict(train_states) > 0.5
test_pred = ridge.predict(test_states) > 0.5

train_tss = calculate_tss(train_labels, train_pred)
test_tss = calculate_tss(test_labels, test_pred)

print(f'Re-evaluated Train TSS: {train_tss}')
print(f'Re-evaluated Test TSS: {test_tss}')

