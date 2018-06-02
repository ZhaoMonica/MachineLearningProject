import tushare as ts
import matplotlib.pyplot as plt
import mpl_finance as mpf
import numpy as np
from sklearn import neural_network
import pandas as pd
import pickle

'''
	pickle模块是对Python对象结构进行二进制序列化和反序列化的协议实现，
	简单说就是把Python数据变成流的形式。像上面的例子，把数据保存或者读入。

'''

# 获取并保存数据
def load_datas():
	datas = ts.get_hist_data('600848', start='2018-01-01', end='2018-06-01')
	datas.to_csv('data1.csv', columns=['open', 'high', 'close', 'low', 'volume', 'p_change', 'ma5','ma10', 'ma20', 'v_ma5', 'v_ma20'])
	print('数据已下载')
# 从data1.csv中读取数据
def lable_datas():
	f = open('data1.csv')
	df=pd.read_csv(f)     #读入股票数据
	#分类 1 为赚 ；-1 表示赔;
	lable = []
	for i in range(len(df['open'])):
		if df['close'][i] >= df['open'][i]:
			lable.append(1)
		else:
			lable.append(-1)
	# 将分完类的数据保存到 data2.csv 中去
	new_data = np.column_stack((df, lable))
	np.savetxt('data2.csv', new_data, fmt="%s", delimiter=",")
	print('分类完成')

# 从data2.csv中读取数据
def get_datas():
	df=pd.read_csv('data2.csv',names=['date', 'open', 'high', 'close', 'low', 'volume', 'p_change', 'ma5','ma10', 'ma20', 'v_ma5', 'v_ma20', 'label'])
	# data=data[::-1]      #反转，使数据按照日期先后顺序排列
	df = df [::-1]
	X = df.iloc[:, 1:].values
	Y = np.array(df['label'])
	train_x = X[:-70]; test_x = X[-70:]
	train_y = Y[:-70]; test_y = Y[-70:]
	return train_x, test_x, train_y, test_y

def mlpclassifier_finance(train_x, test_x, train_y, test_y):

	classifier = neural_network.MLPClassifier(activation='logistic',
											  max_iter=10000, hidden_layer_sizes=(90,80))
	classifier.fit(train_x, train_y)
	train_score = classifier.score(train_x, train_y)
	test_score = classifier.score(test_x, test_y)
	print('train_score:', train_score)
	print('test_score:', test_score)
	if train_score >= 0.70:
		# 第二种方法
		# dump和load 函数能一个接着一个地将几个对象转储到同一个文件。随后调用 load() 来以同样的顺序检索这些对象
		output = open('data.pkl', 'wb')
		input = open('data.pkl', 'rb')
		s = pickle.dump(classifier, output)
		output.close()
		clf2 = pickle.load(input)
		input.close()
		print(clf2.predict(train_x[0:1]))


# 数据图
def plot_data():
	data = pd.read_csv('data1.csv')
	prices = data[['open', 'high', 'low', 'close']]
	dates = data['date']
	candleData = np.column_stack([list(range(len(dates))), prices])
	fig = plt.figure(figsize=(10, 6))
	ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
	mpf.candlestick_ohlc(ax, candleData, width=0.5, colorup='r', colordown='b')
	plt.show()


# load_datas()
# lable_datas()
# plot_data()
train_x, test_x, train_y, test_y = get_datas()
mlpclassifier_finance(train_x, test_x, train_y, test_y)




