# 导入所使用的包

import numpy as np
import pandas as pd
import tensorflow as tf


# 输入空间
iris = pd.read_csv('iris.txt', header=None)  # 数据特点：没有头
x = iris.iloc[:, :4].values   # 获取样本的特征值
y = iris.iloc[:, 4].values    # 样本的类别（最后一列）
y = np.array([[1 if i == 'Iris-setosa' else -1 for i in y]])  # 'Iris-setosa' 品种的设为1类别
# 特征空间
xGaussian = []  # 保存高维空间的特征值

for i in range(x.shape[0]):    #  x.shape[0] 所有行
	# 高斯核函数：exp(-\\x - L\\/d)
	# axis=-1：根据行来求
	xGaussian.append(np.exp(-np.sum((x[i] - x)**2, axis=-1)/3))
x = np.array(xGaussian).T

# 建立算法模型（较难）
Lx = x.shape[0]    # 表示每个样本有多少个特征值
xHolder = tf.placeholder(shape=[Lx, None], dtype=tf.float32)   # 可以暂时理解为函数的一个形参
yHolder = tf.placeholder(shape=[1, None], dtype=tf.float32)   # 1行多列
w = tf.Variable(tf.random_normal(shape=[1, Lx]))   # w是1行Lx列的矩阵
b = tf.Variable(tf.random_normal(shape=[1]))

# 目标函数：loss
lossTheta = tf.reduce_sum(tf.square(w))  # （先平方再累计求和）：\\w\\^2-->间隔公式
# 损失函数； max(1-y*(wx+b),0)  正常的小于0，异常的大于零；maximum即正常的返回0，异常的返回其本身
lossLabels = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(yHolder, tf.add(tf.matmul(w, xHolder), b)))))
loss = tf.add(tf.multiply(0.01, lossTheta), lossLabels)

opt = tf.train.GradientDescentOptimizer(0.01)   # 梯度下降法，学习率为0.01
train = opt.minimize(loss)    # min(loss)
# 训练算法
init = tf.global_variables_initializer()  # 初始化
sess = tf.Session()   # 会话
sess.run(init)

for i in range(1000):
	sess.run(train, feed_dict={xHolder: x, yHolder: y})   # 喂数据；训练1000次

# 测试验证
w = np.array(sess.run(w))
b = sess.run(b)
y_predict = np.dot(w, x) + b
# 将浮点值转换为正负
y_predict = y_predict.flatten() >= 0
# y_predict == (y > 0) 相等为真，返回1，否则返回0
print('预测的准确率：{}%'.format((y_predict == (y > 0)).sum()/len(y[0])*100))