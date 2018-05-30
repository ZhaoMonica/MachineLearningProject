from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 获取数据
boston = load_boston()
# 获取训练集 训练样本个数是60
x = boston.data[:60, 5].reshape(-1, 1)
# target 中只有价格，不需要再取列数
y = boston.target[:60].reshape(-1, 1)  # 房屋价格
# 展示样本分布
plt.title('训练样本分布')
plt.plot(x, y,'b.')
plt.show()

# 调用 一元线性回归函数
# 获取线性回归模型
model = LinearRegression()
# 将训练数据放入模型中
model.fit(x, y)

# 获取预测数据（预测集）  目的： 对比
# ,5  第5列  reshape(-1, 1) ：转化为实际二维数组
x1 = boston.data[60:100,5].reshape(-1, 1)  # 实际自变量x
y1 = boston.target[60:100].reshape(-1, 1)    # 实际因变量y
y2 = model.predict(x1)  # y2是预测的数据集

plt.plot(x1, y2, 'r-')  # 预测图
plt.plot(x1, y1,'y.') # 实际图
plt.title('预测与实际对比误差情况')
plt.show()
