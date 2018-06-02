import tushare as ts
import matplotlib.pyplot as plt
import mpl_finance as mpf
import numpy as np
from sklearn import neural_network
import pandas as pd


# datas = ts.get_hist_data('600848', ktype='5') #获取5分钟k线数据
# # print(datas)
# datas.to_csv('data.csv',columns=[ 'open', 'high', 'close', 'low' ])

# datas = ts.get_hist_data('600848', start='2018-01-05',end='2018-01-09')
# datas.to_csv('data1.csv',columns=[ 'date', 'open', 'high', 'close', 'low', 'volume', 'p_change', 'ma5','ma10', 'ma20', 'v_ma5', 'v_ma20', 'turnover'])
# # 从data.csv中读取数据

f = open('data.csv')
df = pd.read_csv(f)   # 读入股票数据
# data = df.iloc[:, 1:].values  # 取2列往后的数据
df = df[::-1]
X = df.iloc[:, 1:].values
Y = np.array(df['label'])   #

train_x = X[:-200]; test_x = X[-200:]
train_y = Y[:-200]; test_y = Y[-200:]



# df=pd.read_csv(f)     #读入股票数据
# data=np.array(df['open'])   #获取最高价序列
# data=data[::-1]      #反转，使数据按照日期先后顺序排列
# #以折线图展示data
#
# plt.figure()
# plt.plot(data)
# plt.show()

def mlpclassifier_finance():
	classifier = neural_network.MLPClassifier(activation='logistic',
											  max_iter=10000, hidden_layer_sizes=(100,))
	classifier.fit(train_x, train_y)
	train_score = classifier.score(train_x, train_y)
	test_score = classifier.score(test_x, test_y)
# 在这里保存最好的数据
# 	if test_score >0.90:
# 		classifier.jump()
	print(train_score)
	print(test_score)

#
#

mlpclassifier_finance()
# prices = np.array(['open', 'high', 'close', 'low'])
# dates = np.array(['date'])
# candleData = np.column_stack([list(range(len(dates))), prices])
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
# mpf.candlestick_ohlc(ax, candleData, width=0.5, colorup='r', colordown='b')
# plt.show()
