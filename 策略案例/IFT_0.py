import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
T+0：日内交易
参考：《股票被套：史上最全T+0操作技巧，成功解套！.docx》
'''
x = pd.read_csv('IFT_0.csv', index_col='datetime')
lows = x['low'].values
highs = x['high'].values
closes = x['close'].values

Ndays = len(closes)//48         # 每天48个数据
Re = []
holds = 15
for i in range(Ndays):
    lowT = lows[i*48:(i+1)*48]
    highT = highs[i*48:(i+1)*48]
    closeT = closes[i*48:(i+1)*48]
    startBar = holds//5
    upline = max(highT[:startBar])
    downline = min(lowT[:startBar])
    for i2 in range(startBar, len(lowT)):
        if lowT[i2] < downline:
            # 低于下线买入
            Re.append(1-closeT[-1]/downline)
            break
        elif highT[i2] > upline:
            # 高于上线卖出
            Re.append(closeT[-1]/upline-1)
            break
        if i2 > len(lowT)-2:
            Re.append(0.0)

Re = np.array(Re)
plt.plot(Re.cumsum(), label='Strategy')
reRaw = closes[47::48]/closes[::48]-1
plt.plot(reRaw.cumsum(), label='HS300')     # 沪深300指数
plt.plot([0, 500], [0, 0], c='r')
plt.grid()
plt.legend()
plt.show()









