from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model


# 鸢尾花有三种分类
# 加载数据
def load_data():
	iris = datasets.load_iris()
	x_train = iris.data  # 所有数据
	y_train = iris.target  # 所有目标
	return train_test_split(x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)  # 两个y_train不一样

# 定义逻辑回归的函数
def test_logistic(*data):
	x_train, x_test, y_train, y_test = data
	model = linear_model.LogisticRegression()
	model.fit(x_train, y_train)
	print(y_test)
	print(model.predict(x_test))
	print(model.score(x_test, y_test))


# 测试
if 	__name__== '__main__':
	x_train, x_test, y_train, y_test = load_data()
	test_logistic(x_train, x_test,y_train,y_test)