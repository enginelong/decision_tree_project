import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms

raw_samples = [] # 保存样本数据
with open('D:\python\data\car.txt', 'r') as f:
    for line in f.readlines():
        raw_samples.append(line.replace('\n', '').split(','))
data = np.array(raw_samples).transpose()
print(data.shape)

encoders = [] # 记录标签编码器
train_x = [] # 编码后的数据

# 对样本进行标签编码
for row in range(len(data)):
    encoder = sp.LabelEncoder() # 创建标签编码器
    encoders.append(encoder)
    if row < len(data) - 1: # 不是最后一行，因此为样本特征
        coder = encoder.fit_transform(data[row]) # 编码
        train_x.append(coder)
    else: # 最后一行为样本标签
        train_y = encoder.fit_transform(data[row])
train_x = np.array(train_x).transpose()
train_y = np.array(train_y)
print(train_x.shape)
print(train_y.shape)

# 创建随机森林分类器
model = se.RandomForestClassifier(max_depth=8,
                                  n_estimators=150,
                                  random_state=1)
# 训练模型
model.fit(train_x, train_y)
print('accuracy: ', model.score(train_x, train_y)) # 打印出平均精度

# 预测
pred_data = [['high', 'med', '5more', '4', 'big', 'low'],
             ['high', 'high', '4', '4', 'med', 'med'],
             ['low', 'low', '2', '2', 'small', 'high'],
             ['low', 'med', '3', '4', 'med', 'high']]
pred_data = np.array(pred_data).transpose()
pred_x = []
for row in range(len(pred_data)):
    encoder = encoders[row] # 获取用于原始训练数据的编码器
    pred_x.append(encoder.fit_transform(pred_data[row]))
pred_x = np.array(pred_x).transpose()
print(pred_x.shape)
pred_y = model.predict(pred_x)
pred_data_reverse = encoders[-1].inverse_transform(pred_y) # 反向解析出数据
print('pred_data_reverse:\n ', pred_data_reverse)



