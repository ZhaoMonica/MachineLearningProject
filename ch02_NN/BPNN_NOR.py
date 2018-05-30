import math
import random

random.seed(0)


# 创建随机函数
def rand(a, b):
    return (b - a) * random.random() + a


# 创建m*n矩阵
def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


# 求解sigmoid函数值
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# 求解sigmoid导数值
def sigmoid_derivative(x):
    return x * (1 - x)


# 定义BPNeuralNetwork类， 使用三个列表维护输入层，隐含层和输出层神经元，
# 列表中的元素代表对应神经元当前的输出值.使用两个二维列表以邻接矩阵的形式维
# 护输入层与隐含层， 隐含层与输出层之间的连接权值， 通过同样的形式保存矫正矩阵.
class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    # 定义setup方法初始化神经网络
    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # 初始化各层节点
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # 创建输入层--隐藏层和隐藏层--输出层之间的Theta权重矩阵
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # 随机初始化Theta
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # 初始化矫正矩阵
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    # 定义predict方法进行一次前馈，并返回输出
    def predict(self, inputs):
        # 激活输入层
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # 激活隐含层
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # 激活输出层
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    # 定义back_propgate方法定义一次反向传播和更新权值的过程，并返回最终预测误差
    def back_propagate(self, case, label, learn, correct):
        # 进行前馈
        self.predict(case)
        # 获得该次前馈输出层的误差
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # 获得该次前馈隐藏层的误差
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # 更新隐藏层和输出层间的Theta权重矩阵和矫正矩阵
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # 更新输入层和隐藏层间的Theta权重矩阵和矫正矩阵
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # 获得全局误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    # 定义train方法控制迭代，该方法可以修改最大迭代次数、学习率、矫正率三个参数
    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    # 编写test方法，演示如何使用神经网络学习异或逻辑
    def test(self):
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        labels = [[0], [1], [1], [0]]
        self.setup(2, 5, 1)
        self.train(cases, labels, 10000, 0.05, 0.1)
        for case in cases:
            print(self.predict(case))


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()
