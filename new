import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import log_loss, accuracy_score

df = pd.read_csv('建模 - 副本240610.csv')
# 划分特征和目标变量
X = df.drop(['Fail'], axis=1)
y = df['Fail']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=72, stratify=df['Fail'])
df.head()

# 初始化SVM分类模型（使用固定的超参数）
model_svm = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
# 保存每次bootstrap的log loss和accuracy
log_loss_list = []
accuracy_list = []

# Bootstrap 重抽样1000次
n_iterations = 1000
for i in range(n_iterations):
    # 重抽样数据
    X_resampled, y_resampled = resample(X_train, y_train, random_state=i)
    
    # 训练模型
    model_svm.fit(X_resampled, y_resampled)
    
    # 在训练集上进行预测（也可以使用独立的验证集）
    y_pred_proba = model_svm.predict_proba(X_resampled)
    y_pred = model_svm.predict(X_resampled)
    
    # 计算log loss和accuracy
    log_loss_value = log_loss(y_resampled, y_pred_proba)
    accuracy_value = accuracy_score(y_resampled, y_pred)
    
    # 保存结果
    log_loss_list.append(log_loss_value)
    accuracy_list.append(accuracy_value)

# 计算log loss和accuracy的平均值和标准差
mean_log_loss = np.mean(log_loss_list)
std_log_loss = np.std(log_loss_list)
mean_accuracy = np.mean(accuracy_list)
std_accuracy = np.std(accuracy_list)

# 输出结果
print(f"Mean Log Loss: {mean_log_loss:.4f} (± {std_log_loss:.4f})")
print(f"Mean Accuracy: {mean_accuracy:.4f} (± {std_accuracy:.4f})")

