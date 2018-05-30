import matplotlib.pyplot as plt
from sklearn import neural_network
from sklearn.datasets import load_digits


# 加载数据集
digits = load_digits()
X = digits.data
Y = digits.target
#  过拟合：增加训练样本
train_x = X[:-100]; test_x = X[-100:]
train_y = Y[:-100]; test_y = Y[-100:]


def mlpclassifier_digits():
	classifier = neural_network.MLPClassifier(activation='logistic',
											  max_iter=10000, hidden_layer_sizes=(100,))
	classifier.fit(train_x, train_y)
	train_score = classifier.score(train_x, train_y)
	test_score = classifier.score(test_x, test_y)
	print(train_score)
	print(test_score)
	tes=test_y[:100]
	print(tes)
	res = classifier.predict(test_x[:100])
	print(res)
	fig = plt.figure(figsize=(6, 6))
	for i in range(100):
		ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
		ax.imshow(test_x.reshape(-1, 8, 8)[i], cmap=plt.cm.binary, interpolation='nearest')
		if tes[i] == res[i]:
			ax.text(0, 7, str(res[i]), color="green")
		else:
			ax.text(0, 7, str(res[i]), color='red')
	plt.show()


mlpclassifier_digits()