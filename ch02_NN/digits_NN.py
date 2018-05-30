import numpy as np
import matplotlib.pyplot as plt
from sklearn import neural_network
from sklearn.datasets import load_digits


# 加载数据集
digits = load_digits()
# # print(digits)
# print(digits.keys())
# n_samples, n_features = digits.data.shape
# print("Numble of sample:", n_samples)   # 1797个样本个数
# print("Number of feature", n_features)  # 64个特征
# # 第一个样例
# print(digits.data[0])     #
# print(digits.data.shape)   # （1797,64）
# print(digits.target.shape)   # （1797，）
# print(digits.target[0])        # 样本分类
x = digits.data
y = digits.target
plt.gray()
plt.imshow(digits.images[0])   # 十分类问题
plt.show()
fig = plt.figure(figsize=(6, 6))
for i in range(64):
	ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
	ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
	ax.text(0, 7, str(digits.target[i]))
plt.show()

# 模型训练
 # PCA 降维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(digits.data)
plt.scatter(proj[:, 0], proj[:, 1], c=digits.target)
plt.colorbar()
plt.show()
 #  训练模型
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
clf = GaussianNB()
clf.fit(X_train, y_train)

# 结果分析
predicted = clf.predict(X_test)
expected = y_test
matches = (predicted == expected)
print(matches.sum() / float(len(matches)))
print(clf.score(X_test, y_test))
from sklearn.metrics import accuracy_score
print(accuracy_score(predicted, expected))

from sklearn.metrics import classification_report
print(classification_report(expected, predicted))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(expected, predicted))

fig = plt.figure(figsize=(6, 6))
for i in range(64):
	ax = fig.add_subplot(8, 8, i + 1,xticks=[],yticks=[])
	ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary, interpolation='nearest')
	# ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
	if predicted[i] == expected[i]:
		ax.text(0, 7, str(predicted[i]), color="green")
	else:
		ax.text(0, 7, str(predicted[i]), color='red')
plt.show()

# X = digits.data[:, 0:2]; Y = digits.target
# data = np.hstack((X, Y.reshape(Y.size, 1)))
# np.random.shuffle(data)		# 打乱样本顺序
# X = data[:, :-1]; Y = data[:, -1]
# train_x = X[:-30]; test_x = X[-30:]
# train_y = Y[:-30]; test_y = Y[-30:]
#
#
# def plot_samples(ax, x, y):
# 	n_class = 10
# 	plot_colors = "bry"
# 	for i, color in zip(range(n_class), plot_colors):
# 		idx = np.where(y == i)
# 		ax.scatter(x[idx, 0], x[idx, 1], c=color, label=digits.target_names[i], cmap=plt.cm.Paired)
#
#
# def plot_classifier_predict_meshgrid(ax, clf, x_min, x_max, y_min, y_max):
# 	plot_step = 0.02
# 	xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
# 	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# 	Z = Z.reshape(xx.shape)
# 	ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#
#
# def mlpclassifier_digits():
# 	fig = plt.figure()
# 	ax = fig.add_subplot(1, 1, 1)
# 	classifier = neural_network.MLPClassifier(activation='logistic',
# 											  max_iter=10000, hidden_layer_sizes=(30,))
# 	classifier.fit(train_x, train_y)
# 	train_score = classifier.score(train_x, train_y)
# 	test_score = classifier.score(test_x, test_y)
# 	x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
# 	y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
# 	plot_classifier_predict_meshgrid(ax, classifier, x_min, x_max, y_min, y_max)
# 	plot_samples(ax, train_x, train_y)
# 	ax.legend(loc='best')
#
# 	ax.set_title("train score:%f; test score:%f" % (train_score, test_score))
# 	plt.show()
#
#
# mlpclassifier_digits()