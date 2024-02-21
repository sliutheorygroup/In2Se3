import numpy as np
import matplotlib.pyplot as plt

# 从文本文件加载数据
data_t = np.loadtxt('all_q46.txt')
data = np.loadtxt('Q4Q6_380K02ev.txt')
#data = np.loadtxt('Q4Q6_traj_300K02eV.txt')
# 提取x、y和z列
x_t = data_t[:,0]
y_t = data_t[:,1]
x = data[:, 0]
y = data[:, 1]

len_t = len(x_t)
len_x = len(x)
dpi=300
sizes_t = np.ones((1,len_t))*0.2
sizes = np.ones((1,len_x))*0.2
plt.figure(dpi=dpi, figsize=(3.5,5))
# 创建散点图
#plt.scatter(x, y, c=f, cmap='viridis', marker='o', s = sizes)
plt.scatter(x_t, y_t, c="gray", marker='o', s = sizes_t)
plt.scatter(x, y, c="green", marker='o', s = sizes, alpha = 0.05)
# 添加颜色条
#plt.colorbar(label='Force Error')

# 添加轴标签
plt.xlabel('$Q_4$')
plt.ylabel('$Q_6$')
plt.xlim((0,1))
plt.ylim((0,1))

# 显示图形

plt.savefig(fname='ferror_380.png',dpi=dpi,format='png')
plt.show()

