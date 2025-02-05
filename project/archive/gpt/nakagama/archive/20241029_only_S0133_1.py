import time
import scipy.io
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from scipy.signal import butter, lfilter

start_time = time.time()

# EEGデータの読み込み
fp = '../../../../dataset/Aygun2024/physio/'
eeg_data_1 = scipy.io.loadmat(fp + 'S0113/EEG_S0113_1.mat')['rawEEG']
#eeg_data_2 = scipy.io.loadmat(fp + 'S0116/EEG_S0116_1.mat')['rawEEG']
#eeg_data_3 = scipy.io.loadmat(fp + 'S0116/EEG_S0116_2.mat')['rawEEG']

# SIMlslデータの読み込み
simlsl_data_1 = scipy.io.loadmat(fp + 'S0113/SIMlsl_S0113_1.mat')['SIM_lsl']
#simlsl_data_2 = scipy.io.loadmat(fp + 'S0116/SIMlsl_S0116_1.mat')['SIM_lsl']
#simlsl_data_3 = scipy.io.loadmat(fp + 'S0116/SIMlsl_S0116_2.mat')['SIM_lsl']

# バンドパスフィルタの定義（ベータ波：13-30Hz）
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# サンプリング周波数（仮に256Hzとする）
fs = 256

# ベータ波の抽出
eeg_beta_1 = bandpass_filter(eeg_data_1[1, :], 13, 30, fs)

# ベータ波のパワーを計算し、ラベルを生成（クラスごとのサンプル数の割合が均等になるように）
power_threshold_1 = np.percentile(eeg_beta_1**2, 50)
y_1 = (eeg_beta_1**2 > power_threshold_1).astype(int)
y = y_1

# lat加速度とlong加速度の抽出
lat_acc_1 = simlsl_data_1[18, :]
long_acc_1 = simlsl_data_1[19, :]

# 特徴量行列の作成
X_1 = np.vstack((lat_acc_1, long_acc_1)).T
X = X_1

# 最小のサンプル数に合わせてトリミング
min_samples = min(X.shape[0], y.shape[0])

X = X[:min_samples, :]
y = y[:min_samples]

# NaN 値の処理 - 平均値で補完
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVMモデルの作成と訓練
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# 訓練精度とテスト精度の計算
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("train_accuracy   ",train_accuracy)
print("test_accuracy    ",test_accuracy)

# 混同行列の計算と表示
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

print("conf_matrix_train    \n",conf_matrix_train)
print("conf_matrix_test     \n",conf_matrix_test)

print("elapse time ",time.time()-start_time)
