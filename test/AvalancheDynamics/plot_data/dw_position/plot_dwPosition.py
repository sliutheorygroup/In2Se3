import matplotlib.pyplot as plt

# load data
data = []
with open('position.dat', 'r') as file:
    for line in file:
        values = [float(val) for val in line.split()]
        data.append(values)

x_values = [row[0] for row in data]
y_values = [row[1] for row in data]
error_values = [row[2] for row in data]

plt.errorbar(x_values[::2], y_values[::2], yerr=error_values[::2], fmt='o', markersize=5.0, capsize=3.0,color='green',zorder=0,label='300K 2.0V/nm')
plt.plot(x_values[::2], y_values[::2], ls='-', lw=2.0,color='green',zorder=1)

plt.xlabel(r'$t$ (ps)', fontsize=11)  
plt.ylabel(r'$x_{\mathrm{DW}}$ ($\mathrm{\AA}$)', fontsize=11)
plt.title('Domain Wall Position with Time')
plt.xlim(0,800)
plt.ylim(-1,110)

plt.legend(fontsize=11)
plt.savefig('dw_position.png')
plt.show()

