import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('aapl_stock.csv')
data = df[['Close']].values

# 标准化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 30天序列预测1天
seq_len = 30
X, y = [], []
for i in range(len(data_scaled)-seq_len):
    X.append(data_scaled[i:i+seq_len])
    y.append(data_scaled[i+seq_len])

X = np.array(X)
y = np.array(y)

# 划分数据集
train_size = int(len(X)*0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 保存数据
np.savez('stock_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, max_close=scaler.data_max_)
print('数据已预处理完成并保存到stock_data.npz')
