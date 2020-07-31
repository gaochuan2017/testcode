import numpy as np
import matplotlib.pyplot as plt
import math
'''
a=np.interp(1.5,[0,1,2],[2,4,8])
print(a)
''' 
#模拟yolov3-r的学习率设置，训练方式为在线更新BatchNumber * BatchSize = tolal number of 1 ECPOCH
# 总计8张图，分为8个batch 每个batch 1 张图
#训练效果最好的模型来自epoch = 30000 iters = 684
epochs = 300
BatchNumber = 8
iters = 300*BatchNumber
lr0 = 0.01
lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
x =[int(i/BatchNumber) for i in range(iters)]
y =[lf(i)*lr0 for i in range(iters)]
burn_in = [np.interp(i,[0,500],[0,lr0*lf(int(i/BatchNumber))]) if i<=500 else lr0*lf(int(i/BatchNumber)) for i in range(iters)]
plt.plot(x,burn_in)
plt.xlabel('burn_in 500 epoch')
plt.ylabel('learning rate')
plt.show()
