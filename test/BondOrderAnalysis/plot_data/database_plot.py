import numpy as np
import matplotlib.pyplot as plt

# 从文本文件加载数据
data = np.loadtxt('all_q46.txt')
#data = np.loadtxt('Q4Q6_380K02ev.txt')
#data = np.loadtxt('Q4Q6_traj_300K02eV.txt')
# 提取x、y和z列
x = data[:, 0]
y = data[:, 1]
fx = data[:, 2]
fy = data[:,3]
fz = data[:,4]
#f = np.sqrt(fx*fx + fy*fy + fz*fz)
f = np.abs(fx)
len_x = len(x)
dpi=300
sizes = np.ones((1,len_x))*0.2
print(np.max(f))
plt.figure(dpi=dpi, figsize=(5,5))
# 创建散点图
plt.scatter(x, y, c=f, cmap='viridis', marker='o', s = sizes)
#plt.scatter(x, y, c="red", marker='o', s = sizes)
# 添加颜色条
plt.colorbar(label='Force Error')

# 添加轴标签
plt.xlabel('$Q_4$')
plt.ylabel('$Q_6$')
plt.xlim((0,1))
plt.ylim((0,1))

# 显示图形

plt.savefig(fname='ferror_train.png',dpi=dpi,format='png')
plt.show()

