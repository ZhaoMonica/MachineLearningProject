from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


x = [20, 16, 34, 23, 27, 32, 18, 22]
y = [64, 61, 84, 70, 88, 92, 72, 77]
# 展示样本分布
plt.title('训练样本分布')
plt.plot(x, y,'b.')
plt.show()

#
# model =LinearRegression()
# model.fit(x, y)

# plt.plot(x,y, 'k-')
# plt.show()
