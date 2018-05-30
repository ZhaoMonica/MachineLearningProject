from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split


# 加载手写识别体
def load_sklearn_digits():
	digits = datasets.load_digits()
	x_train = digits.data
	y_train = digits.target
	return train_test_split(x_train, y_train, test_size=0.25, random_state=0,stratify=y_train)


def test_knn(*data):
	x_train, x_test, y_train, y_test = data
	model = neighbors.KNeighborsClassifier()
	model.fit(x_train, y_train)
	# 测试数据与实际数据进行对比
	print(model.predict(x_test))
	print(y_test)
	# 测试的准确率
	print(model.score(x_test, y_test))


if __name__ == '__main__':
	x_train, x_test, y_train, y_test = load_sklearn_digits()
	test_knn(x_train, x_test, y_train, y_test)


