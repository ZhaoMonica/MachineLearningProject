# 2.	针对testSetRBF2.txt中数据，使用高斯核函数，进行SVM分类，画图并标记出支持向量。

from numpy import *
import matplotlib.pyplot as plt
from sklearn import svm


# 加载数据集
def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr=open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return array(dataMat), array(labelMat)


data, target = loadDataSet("testSetRBF2.txt")
index1 = where(target == 1)
X1 = data[index1]
index2 = where(target == -1)
X2 = data[index2]
# 二维空间画图
plt.plot(X1[:,0],X1[:,1],'ro')
plt.plot(X2[:,0],X2[:,1],'bx')
plt.show()
# # 高斯核函数
sigma = 0.01
clf = svm.SVC(C=1, kernel='rbf', gamma=1 / sigma)
clf.fit(data, target)
plt.scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.Paired)  # [:，0]列切片，第0列
plt.axis('tight')
plt.show()
