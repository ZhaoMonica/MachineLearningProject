from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
print(iris.keys())
# 150行，4种特征
print(iris.data.shape)
print(iris.target.shape)
print(iris.target)
# 一元线性回归  监督学习  用于数据预测  要求数据线性


# 数量   行
n = np.arange(12).reshape(3,4)
print(len(n))