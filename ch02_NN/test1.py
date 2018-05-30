import numpy as np
from sklearn import neural_network, datasets

np.random.seed(0)
digits = datasets.load_digits()
print(digits.data.shape)

X = digits.data[:-100]; Y = digits.target[:-100]
X_test = digits.data[-100:]; Y_test = digits.target[-100:]

def mlpclassifier_iris():
	classifier = neural_network.MLPClassifier(activation='logistic',
											  max_iter=10000, hidden_layer_sizes=(100,))
	classifier.fit(X, Y)
	train_score = classifier.score(X, Y)
	print('train score: {}'.format(train_score))
	test_score = classifier.score(X_test, Y_test)
	print('test score: {}'.format(test_score))
	print(Y_test[:100])
	res = classifier.predict(X_test[:100])
	print(res)
#
#
#
mlpclassifier_iris()