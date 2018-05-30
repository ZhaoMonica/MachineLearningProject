from sklearn import datasets
from sklearn import decomposition


def load_data():
	iris = datasets.load_iris()
	return iris.data, iris.target


def test_PCA(*data):
	x, y = data
	# n_components=2 :降成二维
	pca = decomposition.PCA(n_components=None)
	pca.fit(x)
	print(pca.explained_variance_ratio_)


if __name__ == '__main__':
	x, y = load_data()
	test_PCA(x, y)