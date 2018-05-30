from sklearn import datasets,svm
from sklearn.model_selection import train_test_split


# 加载数据
def load_data():
	iris = datasets.load_iris()
	X_train = iris.data
	y_train = iris.target
	return train_test_split(X_train, y_train, test_size=0.25, random_state=0,stratify=y_train)


# 测试线性分类支持向量机
def test_LinearSVC(*data):
	X_train, X_test, y_train, y_test = data
	cls = svm.LinearSVC()
	cls.fit(X_train, y_train)
	print('Coefficients:%s, intercept %s' % (cls.coef_,cls.intercept_) )
	print('Score: %.2f' % cls.score(X_test, y_test))


if __name__ == "__main__":
	X_train, X_test, y_train, y_test = load_data()
	test_LinearSVC(X_train, X_test, y_train, y_test)