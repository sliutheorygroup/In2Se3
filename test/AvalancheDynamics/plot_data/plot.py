#!/usr/bin/env python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from numpy import polyfit, poly1d
from matplotlib.patches import Circle, Polygon

plt.style.use('nature.mplstyle')

# Font
mathfont = {'usetex': True}

# Color
clr1 = '#79aacf'  # In atom
clr2 = '#ffae34'  # Se atom
clr3 = 'C3'  # Postive polarization
clr4 = 'C0'  # Negative polarization
clr5 = 'C2'  # A-type arrow
clr6 = 'C0'  # Domain wall
clr7 = 'C4'  # Line-by-line
clr8 = 'C9'  # Stone-skipping
clr9 = 'C7'  # Atomic lines

# Layout
fig_w, fig_h = 6.0, 4.0
fig = plt.figure(figsize=(fig_w, fig_h))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#ax = fig.add_subplot()
hw_ratio = 0.47
spx = [0.00, 0.31, 0.00]  # space in inch
w = fig_h / 3 / hw_ratio
o1 = [spx[0]/fig_w, 0]
o2 = [(spx[0]+spx[1]+w)/fig_w, 0]
ax1 = fig.add_axes([o1[0], o1[1], w/fig_w, 1])
ax2 = fig.add_axes([o2[0], o2[1], w/fig_w, 1])
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
ax1.axis('off')
ax2.axis('off')
ax11 = ax1.inset_axes([0, 2/3, 1, 1/3], transform=ax1.transAxes)
ax12 = ax1.inset_axes([0, 1/3, 1, 1/3], transform=ax1.transAxes)
ax13 = ax1.inset_axes([0, 0/3, 1, 1/3], transform=ax1.transAxes)
ax21 = ax2.inset_axes([0, 2/3, 1, 1/3], transform=ax2.transAxes)
ax22 = ax2.inset_axes([0, 1/3, 1, 1/3], transform=ax2.transAxes)
ax23 = ax2.inset_axes([0, 0/3, 1, 1/3], transform=ax2.transAxes)
ax11.axis('off')
ax12.axis('off')
ax13.axis('off')
ax21.axis('off')
ax22.axis('off')
ax23.axis('off')
ax11i = ax11.inset_axes([0, 0, 1, 1/hw_ratio], transform=ax11.transAxes)
ax21i = ax21.inset_axes([0, 0, 1, 1/hw_ratio], transform=ax21.transAxes)
ax22i = ax22.inset_axes([0, 0, 1, 1/hw_ratio], transform=ax22.transAxes)
ax23i = ax23.inset_axes([0, 0, 1, 1/hw_ratio], transform=ax23.transAxes)
ax12i = ax12.inset_axes([0.125, 0.10, 0.875, 0.90], transform=ax12.transAxes)
ax13i = ax13.inset_axes([0.125, 0.16, 0.875, 0.50], transform=ax13.transAxes)
ax11i.set(xlim=(0,1), ylim=(0,1))
ax21i.set(xlim=(0,1), ylim=(0,1))
ax22i.set(xlim=(0,1), ylim=(0,1))
ax23i.set(xlim=(0,1), ylim=(0,1))
ax11i.axis('off')
ax21i.axis('off')
ax22i.axis('off')
ax23i.axis('off')
ax11.text( 0.012, 0.99, 'a', ha='left', va='top', fontsize=8.5, fontweight='bold', transform=ax11.transAxes)
ax12.text( 0.012, 1.02, 'b', ha='left', va='top', fontsize=8.5, fontweight='bold', transform=ax12.transAxes)
ax13.text( 0.012, 0.81, 'g', ha='left', va='top', fontsize=8.5, fontweight='bold', transform=ax13.transAxes)
ax21.text( 0.200, 0.99, 'c', ha='left', va='top', fontsize=8.5, fontweight='bold', transform=ax21.transAxes)
ax22.text( 0.200, 0.95, 'd', ha='left', va='top', fontsize=8.5, fontweight='bold', transform=ax22.transAxes)
ax23.text(-0.050, 0.85, 'e', ha='left', va='top', fontsize=8.5, fontweight='bold', transform=ax23.transAxes)
ax23.text( 0.505, 0.85, 'f', ha='left', va='top', fontsize=8.5, fontweight='bold', transform=ax23.transAxes)

# Subfigure A
def plot_axes(ax, origin, length, radius, label, zorient):
    ax.arrow(origin[0], origin[1], length, 0, length_includes_head=True, lw=0.6, head_width=0.01, head_length=0.01, overhang=0.3, color='k')
    ax.arrow(origin[0], origin[1], 0, length, length_includes_head=True, lw=0.6, head_width=0.01, head_length=0.01, overhang=0.3, color='k')
    ax.add_artist(plt.Circle((origin[0],origin[1]), radius, lw=0.6, facecolor='w', edgecolor='k'))
    if zorient == 'out':
        ax.add_artist(plt.Circle((origin[0],origin[1]), radius*0.2, lw=0.6, facecolor='k', edgecolor='k'))
    else:
        crossx  = [origin[0] - radius/np.sqrt(2), origin[0] + radius/np.sqrt(2)]
        crossy1 = [origin[1] - radius/np.sqrt(2), origin[1] + radius/np.sqrt(2)]
        crossy2 = [origin[1] + radius/np.sqrt(2), origin[1] - radius/np.sqrt(2)]
        ax.plot(crossx, crossy1, lw=0.6, c='k')
        ax.plot(crossx, crossy2, lw=0.6, c='k')
    labelpad = 0.015
    xlabel = origin + np.array([length+labelpad, -0.005])
    ylabel = origin + np.array([0, length+labelpad])
    zlabel = origin - np.ones(2)*(radius+labelpad*1.5)/np.sqrt(2)
    ax.text(xlabel[0], xlabel[1], label[0], ha='center', va='center')
    ax.text(ylabel[0], ylabel[1], label[1], ha='center', va='center')
    ax.text(zlabel[0], zlabel[1], label[2], ha='center', va='center')
    return ax
plot_axes(ax11i, [0.045, 0.105], 0.06, 0.011, [r'$x$',r'$y$',r'$z$'], 'out')

# Type A and B
# Custom
el = 0.062
rad1, rad2 = 0.225 * el, 0.125 * el  # radius of circles
pad = rad1 * 1.8
dsh = (0,(2.4,0.8))
c = np.sqrt(3)
# Coordinates
ori1 = np.array([[0.0, 1.0],
                 [1.5, 0.5], [1.5, 1.5],
                 [3.0, 0.0], [3.0, 1.0], [3.0, 2.0]])
ori2 = np.array([[0.5, 0.5], [0.5, 1.5],
                 [2.0, 0.0], [2.0, 1.0], [2.0, 2.0],
                 [3.5, 0.5], [3.5, 1.5]])
ori1[:,0] = ori1[:,0] * el * 1 + pad + 0.05
ori2[:,0] = ori2[:,0] * el * 1 + pad + 0.05
ori1[:,1] = ori1[:,1] * el * c + 0.25 * el * c + 0.133
ori2[:,1] = ori2[:,1] * el * c + 0.25 * el * c + 0.133
# Atoms
for i in range(len(ori1)):
    ax11i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad1, ls=dsh, lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range(len(ori2)):
    ax11i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad1, ls='-', lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
    ax11i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
# Bonds
for i in range(len(ori1)):
    for j in range(len(ori2)):
        distance = np.sqrt((ori1[i,0] - ori2[j,0])**2 + (ori1[i,1] - ori2[j,1])**2)
        if distance < el * 1.1:
            ax11i.plot([ori1[i,0], ori2[j,0]], [ori1[i,1], ori2[j,1]], ls='-', lw=0.6, c='C7', alpha=0.6, zorder=0)
#ax11i.add_patch(Circle((ori2[3,0], ori2[3,1]), el, ls='-', lw=0.6, fc='none',  ec='C7', alpha=1.0, zorder=1))
ax11i.arrow(ori2[3,0]+rad2*1.4, ori2[3,1], el-rad2*2.0, 0, length_includes_head=True, lw=0.8, head_width=0.011, head_length=0.012, overhang=0, color='C7')
ax11i.arrow(ori2[3,0]-rad2*1.4/2, ori2[3,1]+rad2*1.4*c/2, -(el-rad2*2.0)/2,  (el-rad2*2.0)*c/2, length_includes_head=True, lw=0.8, head_width=0.011, head_length=0.012, overhang=0, color=clr5)
ax11i.arrow(ori2[3,0]-rad2*1.4/2, ori2[3,1]-rad2*1.4*c/2, -(el-rad2*2.0)/2, -(el-rad2*2.0)*c/2, length_includes_head=True, lw=0.8, head_width=0.011, head_length=0.012, overhang=0, color=clr5)
ax11i.text(ori2[3,0]-rad2*4.5, ori2[3,1]-rad2*0.5, r'$A$', ha='center', va='center', color=clr5, transform=ax11i.transAxes)
ax11i.text(ori2[3,0]+rad2*5.5, ori2[3,1]+rad2*2.5, r'$B$', ha='center', va='center', color='C7', transform=ax11i.transAxes)

ax11i.arrow((ori2[4,0]+ori1[5,0])/2, ori2[4,1]+pad*1.1, -el/2, 0, length_includes_head=True, lw=0.6, head_width=0.01, head_length=0.01, overhang=0.3, color='k')
ax11i.arrow((ori2[4,0]+ori1[5,0])/2, ori2[4,1]+pad*1.1,  el/2, 0, length_includes_head=True, lw=0.6, head_width=0.01, head_length=0.01, overhang=0.3, color='k')
ax11i.text((ori2[4,0]+ori1[5,0])/2, ori2[4,1]+pad*1.3, r'$a$', ha='center', va='bottom', fontsize=6.5)

# Domain wall A
# Coordinates
ori1 = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0],
                 [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
                 [3.0, 0.0], [3.0, 1.0], [3.0, 2.0],
                 [4.5, 0.5], [4.5, 1.5], [4.5, 2.5],
                 [6.0, 0.0], [6.0, 1.0], [6.0, 2.0],
                 [7.5, 0.5], [7.5, 1.5], [7.5, 2.5],
                 [9.0, 0.0], [9.0, 1.0], [9.0, 2.0]])
ori2 = np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
                 [2.0, 0.0], [2.0, 1.0], [2.0, 2.0],
                 [3.5, 0.5], [3.5, 1.5], [3.5, 2.5],
                 [5.0, 0.0], [5.0, 1.0], [5.0, 2.0],
                 [6.5, 0.5], [6.5, 1.5], [6.5, 2.5],
                 [8.0, 0.0], [8.0, 1.0], [8.0, 2.0],
                 [9.5, 0.5], [9.5, 1.5], [9.5, 2.5]])
ori1[:,0] = ori1[:,0] * el * 1 + (1 - pad - 9.5 * el) 
ori2[:,0] = ori2[:,0] * el * 1 + (1 - pad - 9.5 * el)
ori1[:,1] = ori1[:,1] * el * c + 0.133
ori2[:,1] = ori2[:,1] * el * c + 0.133
# Atoms
for i in range(len(ori1)):
    ax11i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad1, ls=dsh, lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range(len(ori2)):
    ax11i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad1, ls='-', lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range( 0,  6, 1):
    ax11i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
for i in range( 6, 15, 1):
    ax11i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
for i in range(15, 21, 1):
    ax11i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
# Bonds
for i in range(len(ori1)):
    for j in range(len(ori2)):
        distance = np.sqrt((ori1[i,0] - ori2[j,0])**2 + (ori1[i,1] - ori2[j,1])**2)
        if distance < el * 1.1:
            ax11i.plot([ori1[i,0], ori2[j,0]], [ori1[i,1], ori2[j,1]], ls='-', lw=0.6, c='C7', alpha=0.6, zorder=0)
ax11i.fill_between([ori1[ 0,0]-pad, ori2[ 3,0]    ], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.10, fc=clr3, zorder=0)
ax11i.fill_between([ori1[ 6,0],     ori1[12,0]    ], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.10, fc=clr4, zorder=0)
ax11i.fill_between([ori2[15,0],     ori2[18,0]+pad], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.10, fc=clr3, zorder=0)
ax11i.text((ori2[ 3,0]+ori1[ 6,0])/2,     ori1[0,1]-pad*2.3, r'DW$_{A,s}$', ha='center', va='center', color=clr6, fontsize=6.5)
ax11i.text((ori1[12,0]+ori2[15,0])/2,     ori1[0,1]-pad*2.3, r'DW$_{A,l}$', ha='center', va='center', color=clr6, fontsize=6.5)
ax11i.text((ori1[ 0,0]-pad+ori2[ 3,0])/2, ori2[2,1]+pad*1.7, r'$\mathcal{P}^{+}$', ha='center', va='center', color=clr3, fontsize=6.5, fontdict=mathfont)
ax11i.text((ori1[ 6,0]+ori1[12,0])/2,     ori2[2,1]+pad*1.7, r'$\mathcal{P}^{-}$', ha='center', va='center', color=clr4, fontsize=6.5, fontdict=mathfont)
ax11i.text((ori2[15,0]+ori2[18,0]+pad)/2, ori2[2,1]+pad*1.7, r'$\mathcal{P}^{+}$', ha='center', va='center', color=clr3, fontsize=6.5, fontdict=mathfont)
ax11i.plot([(ori2[ 3,0]+ori1[ 6,0])/2, (ori2[ 3,0]+ori1[ 6,0])/2], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', lw=0.8, c=clr6, zorder=2)
ax11i.plot([(ori1[12,0]+ori2[15,0])/2, (ori1[12,0]+ori2[15,0])/2], [ori1[0,1]-pad, ori2[2,1]+pad], ls='-',  lw=0.8, c=clr6, zorder=2)
dsh0 = (0, (2.4,1.2))
ax11i.plot([ori2[ 3,0]+rad2, ori1[ 6,0]-rad2], [ori2[ 3,1], ori1[ 6,1]], ls=dsh0, lw=0.6, c=clr2, zorder=1)
ax11i.plot([ori2[ 4,0]+rad2, ori1[ 7,0]-rad2], [ori2[ 4,1], ori1[ 7,1]], ls=dsh0, lw=0.6, c=clr2, zorder=1)
ax11i.plot([ori2[ 5,0]+rad2, ori1[ 8,0]-rad2], [ori2[ 5,1], ori1[ 8,1]], ls=dsh0, lw=0.6, c=clr2, zorder=1)
ax11i.plot([ori1[12,0]+rad2, ori2[15,0]-rad2], [ori1[12,1], ori2[15,1]], ls=dsh0, lw=0.6, c=clr2, zorder=1)
ax11i.plot([ori1[13,0]+rad2, ori2[16,0]-rad2], [ori1[13,1], ori2[16,1]], ls=dsh0, lw=0.6, c=clr2, zorder=1)
ax11i.plot([ori1[14,0]+rad2, ori2[17,0]-rad2], [ori1[14,1], ori2[17,1]], ls=dsh0, lw=0.6, c=clr2, zorder=1)
for i in [6, 7, 8, 10, 11, 12, 13, 14]:
    ax11i.arrow(ori2[i,0]-rad2*0.25/2, ori2[i,1]-rad2*0.25*c/2, -(el-rad2*1.5)/2, -(el-rad2*1.5)*c/2, length_includes_head=True, lw=0.8, head_width=0.011, head_length=0.012, overhang=0, color=clr5)

# Subfigure B
dat1 = np.loadtxt('./data/position.dat', skiprows=0, usecols=(0,1,2))
dat2 = np.loadtxt('./data/460ps.dat',    skiprows=0, usecols=(0,1,2))
dat3 = np.loadtxt('./data/608ps.dat',    skiprows=0, usecols=(0,1,2))
ax12i.set_xlabel(r'$t$ (ps)')
ax12i.set_ylabel(r'$x_{\mathrm{DW}}$ ($\mathrm{\AA}$)')
ax12i.set(xlim=(0, 800), ylim=(0, 110))
ax12i.set_xticks(np.linspace(0, 800, 5))
ax12i.set_yticks(np.linspace(0, 100, 6))
ax12i.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
ax12i.errorbar(dat1[::4,0], dat1[::4,1], dat1[::4,2], fmt='o', markersize=1.8, capsize=1.6, c=clr5, zorder=0)
ax12i.plot(dat1[:,0], dat1[:,1], ls='-', lw=0.8, c=clr6, zorder=1)
ax12i.scatter(dat1[115,0], dat1[115,1], marker='s', fc='none', ec='C1', s=40, linewidths=0.8, alpha=0.8, zorder=2)
ax12i.scatter(dat1[152,0], dat1[152,1], marker='s', fc='none', ec='C1', s=40, linewidths=0.8, alpha=0.8, zorder=2)
ax12i.text(0.390, 0.920, r'$\mathcal{E}$', ha='right', va='center', transform=ax12i.transAxes, fontdict=mathfont)
ax12i.text(0.388, 0.914, r'$_{\mathrm{OP}}=2$ V/nm, $T=300$ K', ha='left', va='center', transform=ax12i.transAxes)
#ax12i.text(0.53, 0 904, r'$T=300$ K', ha='center', va='center', transform=ax12i.transAxes)

ap=dict(arrowstyle='simple,tail_width=0.15,head_width=0.7,head_length=0.7', shrinkA=6, shrinkB=0, facecolor='C1', edgecolor='none', connectionstyle='arc3,rad=0.35', alpha=0.8)
ax12i.annotate('', xy=(dat1[115,0]-100, dat1[115,1]+18), xycoords='data', xytext=(dat1[115,0], dat1[115,1]), textcoords='data', arrowprops=ap)
ap=dict(arrowstyle='simple,tail_width=0.15,head_width=0.7,head_length=0.7', shrinkA=6, shrinkB=0, facecolor='C1', edgecolor='none', connectionstyle='arc3,rad=-0.25', alpha=0.8)
ax12i.annotate('', xy=(dat1[152,0]+50, dat1[152,1]-30), xycoords='data', xytext=(dat1[152,0], dat1[152,1]), textcoords='data', arrowprops=ap)

padx, pady, w, h = 0.04, 0.12, 0.30, 0.33
ax12i1 = ax12i.inset_axes([padx+0.01, 1-pady-h, w, h], transform=ax12i.transAxes)
ax12i2 = ax12i.inset_axes([1-padx-w, pady-0.03, w, h], transform=ax12i.transAxes)
ax12i1.axis('off')
ax12i2.axis('off')
ax12i1.set(xlim=(3, 140), ylim=(2, 79))
ax12i2.set(xlim=(3, 140), ylim=(2, 79))
dsh1 = (0,(2.1,1))
cmap = mpl.colors.ListedColormap(mpl.colormaps['coolwarm'](np.linspace(0.4, 0.6, 2)))
levels = np.linspace(dat2[:,2].min(), dat2[:,2].max(), 3)
ax12i1.tricontourf(dat2[:,1], dat2[:,0], dat2[:,2], levels=levels, cmap=cmap)
ax12i1.tricontour(dat2[:,1], dat2[:,0], dat2[:,2], levels=levels, colors=['C0', 'C0', 'C0'], linewidths=[0, 0.8, 0], linestyles=[dsh1, dsh1, dsh1])
levels = np.linspace(dat3[:,2].min(), dat3[:,2].max(), 3)
ax12i2.tricontourf(dat3[:,1], dat3[:,0], dat3[:,2], levels=levels, cmap=cmap)
ax12i2.tricontour(dat3[:,1], dat3[:,0], dat3[:,2], levels=levels, colors=['C0', 'C0', 'C0'], linewidths=[0, 0.8, 0], linestyles=[dsh1, dsh1, dsh1])
ax12i1.text(0.5, 1.04, '460 ps', ha='center', va='bottom', transform=ax12i1.transAxes)
ax12i2.text(0.5, 1.04, '608 ps', ha='center', va='bottom', transform=ax12i2.transAxes)
ax12i1.arrow(0, (2+79)/2, 51.0, 0, length_includes_head=True, lw=0.8, head_width=5, head_length=4, overhang=0, color=clr6)
#ax12i1.plot([53.5, 53.5], [2, 79],   ls='--', lw=0.8, c=clr6)
#ax12i2.plot([92.0, 92.0], [2, 79],   ls='--', lw=0.8, c=clr6)
ax12i1.plot([114.1, 114.1], [2, 79], ls='-',  lw=0.8, c=clr6)
ax12i2.plot([114.1, 114.1], [2, 79], ls='-',  lw=0.8, c=clr6)
ax12i1.text(0.20,  0.68, r'$x_{\mathrm{DW}}$', ha='center', va='center', transform=ax12i1.transAxes)
ax12i1.text(0.360, -0.08, r'DW$_{A,s}$', ha='center', va='top', color=clr6, transform=ax12i1.transAxes)
ax12i1.text(0.795, -0.08, r'DW$_{A,l}$', ha='center', va='top', color=clr6, transform=ax12i1.transAxes)

# Subfigure D
ax13i.set_ylabel(r'$E$ (eV)', labelpad=4)
ax13i.set(xlim=(0, 10), ylim=(0, 0.4))
ax13i.set_yticks(np.linspace(0, 0.4, 5))
ax13i.set_xticks(np.linspace(0, 10, 6))
ax13i.set_xticklabels([])
ax13i.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
ax13i.tick_params(which='major', top=False, bottom=True, length=2.4)
ax13i.tick_params(which='minor', top=True, bottom=False, length=2.4)
Ea = 0.27
X1 = np.linspace(0, 10, 1000)
Y1 = Ea/2  * (np.sin((X1-0.5)*np.pi)+1)
X2 = np.linspace(1,  9, 1000)
Y2 = Ea/20 * (np.sin((X2+0.5)*np.pi)+1) + Ea
ax13i.plot(X1, Y1, ls='-',  c=clr7, lw=0.8)
ax13i.plot(X2, Y2, ls='--', c=clr8, lw=0.8)
ax13i.text(8.5, 0.155, 'Line-by-line',   ha='center', va='center', color=clr7)
ax13i.text(8.0, 0.340, 'Stone-skipping', ha='center', va='center', color=clr8)
ap=dict(arrowstyle='simple,tail_width=0.12,head_width=0.5,head_length=0.5', shrinkA=2, shrinkB=2, facecolor='C7', edgecolor='none', connectionstyle='arc3,rad=0', alpha=0.8)
ax13i.annotate('', xy=(1, 0.27), xycoords='data', xytext=(1, 0.4), textcoords='data', arrowprops=ap)
# Arrows
ax13ii = ax13i.inset_axes([-0.2, -0.5, 1.4, 2.0], transform=ax13i.transAxes)
ax13ii.set(xlim=(0,1), ylim=(0,1))
ax13ii.axis('off')
def plot_arrows1(ax, box_width, box_height, origin, nup, ndown):
    ntot = nup + ndown
    ax.add_patch(Polygon([[origin[0]-box_width/2, origin[1]-box_height/2], [origin[0]+box_width/2, origin[1]-box_height/2], [origin[0]+box_width/2, origin[1]+box_height/2], [origin[0]-box_width/2, origin[1]+box_height/2]], closed=True, ls='-', lw=0.4, fc='none', ec=clr7, alpha=0.5, transform=ax.transAxes, zorder=0))
    for i in range(nup):
        ax.arrow(origin[0]-box_width/2+box_width/(ntot+0.5)*(0.75+i),     origin[1]-box_height*0.4, 0,  box_height*0.8, length_includes_head=True, lw=0, width=0.004, head_width=0.011, head_length=0.024, overhang=0, ec='none', fc=clr3, alpha=0.7)
    for i in range(ndown):
        ax.arrow(origin[0]-box_width/2+box_width/(ntot+0.5)*(0.75+nup+i), origin[1]+box_height*0.4, 0, -box_height*0.8, length_includes_head=True, lw=0, width=0.004, head_width=0.011, head_length=0.024, overhang=0, ec='none', fc=clr4, alpha=0.7)
for i in range(6):
    plot_arrows1(ax13ii, 0.09, 0.09, [(0.2+i/5)/1.4, 0.25-0.10], i+1, 6-i)

def plot_arrows2(ax, box_width, box_height, origin, nup, ndown):
    ntot = nup + ndown
    ax.add_patch(Polygon([[origin[0]-box_width/2, origin[1]-box_height/2], [origin[0]+box_width/2, origin[1]-box_height/2], [origin[0]+box_width/2, origin[1]+box_height/2], [origin[0]-box_width/2, origin[1]+box_height/2]], closed=True, ls='-', lw=0.4, fc='none', ec=clr8, alpha=0.5, transform=ax.transAxes, zorder=0))
    for i in range(nup):
        ax.arrow(origin[0]-box_width/2+box_width/(ntot+0.5)*(0.75+i),     origin[1]-box_height*0.4, 0,  box_height*0.8, length_includes_head=True, lw=0, width=0.004, head_width=0.011, head_length=0.024, overhang=0, ec='none', fc=clr3, alpha=0.7)
    ax.scatter(origin[0]-box_width/2+box_width/(ntot+0.5)*(0.75+nup), origin[1], marker='o', s=7, color='C7', linewidths=0, alpha=0.9)
    ax.arrow(origin[0]-box_width/2+box_width/(ntot+0.5)*(0.75+nup+1), origin[1]+box_height*0.3, 0, -box_height*0.6, length_includes_head=True, lw=0, width=0.004, head_width=0.011, head_length=0.024, overhang=0, ec='none', fc='C7', alpha=0.9)
    for i in range(2, ndown):
        ax.arrow(origin[0]-box_width/2+box_width/(ntot+0.5)*(0.75+nup+i), origin[1]+box_height*0.4, 0, -box_height*0.8, length_includes_head=True, lw=0, width=0.004, head_width=0.011, head_length=0.024, overhang=0, ec='none', fc=clr4, alpha=0.7)
for i in range(5):
    plot_arrows2(ax13ii, 0.09, 0.09, [(0.2+0.1+i/5)/1.4, 0.75+0.10], i+1, 6-i)
# legend
box_height = 0.09
origin = [0.31, 0.685]
ax13ii.arrow(origin[0], origin[1]-box_height*0.4, 0,  box_height*0.8, length_includes_head=True, lw=0, width=0.004, head_width=0.011, head_length=0.024, overhang=0, ec='none', fc=clr3, alpha=0.7)
ax13ii.text(origin[0]+0.01, origin[1], r'$\mathcal{P}^{+}$', ha='left', va='center', color=clr3, fontdict=mathfont)
origin = [0.39, 0.685]
ax13ii.arrow(origin[0], origin[1]+box_height*0.4, 0, -box_height*0.8, length_includes_head=True, lw=0, width=0.004, head_width=0.011, head_length=0.024, overhang=0, ec='none', fc=clr4, alpha=0.7)
ax13ii.text(origin[0]+0.01, origin[1], r'$\mathcal{P}^{-}$', ha='left', va='center', color=clr4, fontdict=mathfont)
origin = [0.47, 0.685]
ax13ii.scatter(origin[0], origin[1], marker='o', s=7, color='C7', linewidths=0, alpha=0.9)
ax13ii.text(origin[0]+0.01, origin[1], r'$\mathcal{M}$', ha='left', va='center', fontdict=mathfont)
origin = [0.55, 0.685]
ax13ii.arrow(origin[0], origin[1]+box_height*0.3, 0, -box_height*0.6, length_includes_head=True, lw=0, width=0.004, head_width=0.011, head_length=0.024, overhang=0, ec='none', fc='C7', alpha=0.9)
ax13ii.text(origin[0]+0.01, origin[1], r'$\mathcal{A}$', ha='left', va='center', fontdict=mathfont)

# Subfigure C
# Coordinates
ori1 = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0],
                 [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
                 [3.0, 0.0], [3.0, 1.0], [3.0, 2.0],
                 [4.5, 0.5], [4.5, 1.5], [4.5, 2.5],
                 [6.0, 0.0], [6.0, 1.0], [6.0, 2.0]])
ori2 = np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
                 [2.0, 0.0], [2.0, 1.0], [2.0, 2.0],
                 [3.5, 0.5], [3.5, 1.5], [3.5, 2.5],
                 [5.0, 0.0], [5.0, 1.0], [5.0, 2.0],
                 [6.5, 0.5], [6.5, 1.5], [6.5, 2.5]])
ori1[:,0] = ori1[:,0] * el * 1 + (1 - 6.5 * el)/2
ori2[:,0] = ori2[:,0] * el * 1 + (1 - 6.5 * el)/2
ori1[:,1] = ori1[:,1] * el * c + 0.133
ori2[:,1] = ori2[:,1] * el * c + 0.133

# Step 1
# Atoms
for i in range(len(ori1)):
    ax21i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad1, ls=dsh, lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range(len(ori2)):
    ax21i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad1, ls='-', lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range( 0,  3, 1):
    ax21i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
for i in range( 3, 15, 1):
    ax21i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
# Bonds
for i in range(len(ori1)):
    for j in range(len(ori2)):
        distance = np.sqrt((ori1[i,0] - ori2[j,0])**2 + (ori1[i,1] - ori2[j,1])**2)
        if distance < el * 1.1:
            ax21i.plot([ori1[i,0], ori2[j,0]], [ori1[i,1], ori2[j,1]], ls='-', lw=0.6, c='C7', alpha=0.6, zorder=0)
ax21i.fill_between([ori1[ 0,0]-pad, ori2[ 0,0]], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.10, fc=clr3, zorder=0)
ax21i.fill_between([ori1[ 3,0], ori2[12,0]+pad], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.10, fc=clr4, zorder=0)
# Arrows
al1, al2 = el * 0.35, el * 0.3  # arrow lengths
ax21i.arrow(ori1[3,0]+rad2/2, ori1[3,1]+rad2*c/2,  al1/2,  al1*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
ax21i.arrow(ori1[4,0]+rad2/2, ori1[4,1]+rad2*c/2,  al1/2,  al1*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
ax21i.arrow(ori1[5,0]+rad2/2, ori1[5,1]+rad2*c/2,  al1/2,  al1*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
ax21i.arrow(ori1[6,0]-rad1/2, ori1[6,1]-rad1*c/2, -al2/2, -al2*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
ax21i.arrow(ori1[7,0]-rad1/2, ori1[7,1]-rad1*c/2, -al2/2, -al2*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
ax21i.arrow(ori1[8,0]-rad1/2, ori1[8,1]-rad1*c/2, -al2/2, -al2*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
# P
ax21i.text((ori1[ 0,0]-pad+ori2[ 0,0])/2, ori2[2,1]+pad*1.7, r'$\mathcal{P}^{+}$', ha='center', va='center', color=clr3, fontsize=6.5, fontdict=mathfont)
ax21i.text((ori1[ 3,0]+ori2[12,0]+pad)/2, ori2[2,1]+pad*1.7, r'$\mathcal{P}^{-}$', ha='center', va='center', color=clr4, fontsize=6.5, fontdict=mathfont)
# L
ax21i.plot([ori1[ 3,0], ori1[ 3,0]], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax21i.plot([ori1[ 6,0], ori1[ 6,0]], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax21i.plot([ori1[ 9,0], ori1[ 9,0]], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax21i.text(ori1[ 3,0]-0.005, ori1[0,1]-pad*2.0, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax21i.text(ori1[ 6,0]-0.005, ori1[0,1]-pad*2.0, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax21i.text(ori1[ 9,0]-0.005, ori1[0,1]-pad*2.0, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax21i.text(ori1[ 3,0]+0.015, ori1[0,1]-pad*2.0, r'$_1$',          ha='center', va='center', color=clr9, fontsize=6.5)
ax21i.text(ori1[ 6,0]+0.015, ori1[0,1]-pad*2.0, r'$_2$',          ha='center', va='center', color=clr9, fontsize=6.5)
ax21i.text(ori1[ 9,0]+0.015, ori1[0,1]-pad*2.0, r'$_3$',          ha='center', va='center', color=clr9, fontsize=6.5)

origin = [0.12, ori1[0,1]]
ax21i.add_patch(Circle((origin[0], origin[1]+el*0.0), rad1, ls=dsh, lw=0.6, fc='w',   ec=clr1,   transform=ax21i.transAxes))
ax21i.add_patch(Circle((origin[0], origin[1]+el*1.2), rad2, ls='-', lw=0.0, fc=clr2,  ec='none', transform=ax21i.transAxes))
ax21i.add_patch(Circle((origin[0], origin[1]+el*2.4), rad1, ls='-', lw=0.6, fc='w',   ec=clr1,   transform=ax21i.transAxes))
ax21i.text(origin[0]+0.6*el, origin[1]+el*0.0, r'In$^{\mathrm{dn}}$', ha='left', va='center', fontsize=6.5, transform=ax21i.transAxes)
ax21i.text(origin[0]+0.6*el, origin[1]+el*1.2, r'Se',                 ha='left', va='center', fontsize=6.5, transform=ax21i.transAxes)
ax21i.text(origin[0]+0.6*el, origin[1]+el*2.4, r'In$^{\mathrm{up}}$', ha='left', va='center', fontsize=6.5, transform=ax21i.transAxes)

# Step 2
ori1[:,1] = ori1[:,1] - 0.015
ori2[:,1] = ori2[:,1] - 0.015
# Atoms
for i in range(0, 6, 1):
    ax22i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad1, ls=dsh, lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range(9, 15, 1):
    ax22i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad1, ls=dsh, lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range(len(ori2)):
    ax22i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad1, ls='-', lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range( 0,  3, 1):
    ax22i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
for i in range( 6, 15, 1):
    ax22i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
for i in range(3, 5, 1):
    ax22i.add_patch(Circle((ori1[i,0]+el/4, ori1[i,1]+el*c/4), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
ax22i.add_patch(Circle((ori1[5,0]+el/4, ori1[5,1]+el*c/4-el*c*3), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
for i in range(6, 9, 1):
    ax22i.add_patch(Circle((ori1[i,0]-el/4, ori1[i,1]-el*c/4), rad1, ls=dsh, lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
# Bonds
for i in range(len(ori1)):
    for j in range(len(ori2)):
        distance = np.sqrt((ori1[i,0] - ori2[j,0])**2 + (ori1[i,1] - ori2[j,1])**2)
        if distance < el * 1.1:
            ax22i.plot([ori1[i,0], ori2[j,0]], [ori1[i,1], ori2[j,1]], ls='-', lw=0.6, c='C7', alpha=0.6, zorder=0)
ax22i.fill_between([ori1[ 0,0]-pad, ori2[ 0,0]], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.10, fc=clr3, zorder=0)
ax22i.fill_between([ori1[ 9,0], ori2[12,0]+pad], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.10, fc=clr4, zorder=0)
# Arrows
al1, al2 = el * 0.35, el * 0.3  # arrow lengths
ax22i.arrow(ori1[3,0]+el/4+rad2/2, ori1[3,1]+el*c/4+rad2*c/2,        al1/2, al1*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
ax22i.arrow(ori1[4,0]+el/4+rad2/2, ori1[4,1]+el*c/4+rad2*c/2,        al1/2, al1*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
ax22i.arrow(ori1[5,0]+el/4+rad2/2, ori1[5,1]+el*c/4+rad2*c/2-el*c*3, al1/2, al1*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
ax22i.arrow(ori1[6,0]-el/4+rad1/2, ori1[6,1]-el*c/4+rad1*c/2,        al2/2, al2*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
ax22i.arrow(ori1[7,0]-el/4+rad1/2, ori1[7,1]-el*c/4+rad1*c/2,        al2/2, al2*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
ax22i.arrow(ori1[8,0]-el/4+rad1/2, ori1[8,1]-el*c/4+rad1*c/2,        al2/2, al2*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr7)
ax22i.text((ori1[3,0]+el/4+rad2/2+ori1[6,0]-el/4+rad1/2)/2-0.006-0.0007, ori1[3,1]-0.0032, r'$\mathcal{I}$', ha='center', va='center', fontsize=6.5, color=clr7, fontdict=mathfont)
#ax22i.text((ori1[3,0]+el/4+rad2/2+ori1[6,0]-el/4+rad1/2)/2-0.006-0.0007, ori1[3,1]-0.0032, '1', ha='center', va='center', fontsize=6.5, color=clr7)
#ax22i.add_artist(plt.Circle(((ori1[3,0]+el/4+rad2/2+ori1[6,0]-el/4+rad1/2)/2-0.006, ori1[3,1]), 0.016, lw=0.6, facecolor='none', edgecolor=clr7))
ax22i.arrow(ori1[ 6,0]+rad2/2, ori1[ 6,1]+rad2*c/2,  al1/2,  al1*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr8)
ax22i.arrow(ori1[ 7,0]+rad2/2, ori1[ 7,1]+rad2*c/2,  al1/2,  al1*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr8)
ax22i.arrow(ori1[ 8,0]+rad2/2, ori1[ 8,1]+rad2*c/2,  al1/2,  al1*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr8)
ax22i.arrow(ori1[ 9,0]-rad1/2, ori1[ 9,1]-rad1*c/2, -al2/2, -al2*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr8)
ax22i.arrow(ori1[10,0]-rad1/2, ori1[10,1]-rad1*c/2, -al2/2, -al2*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr8)
ax22i.arrow(ori1[11,0]-rad1/2, ori1[11,1]-rad1*c/2, -al2/2, -al2*c/2, length_includes_head=True, lw=0.8, head_width=0.012, head_length=0.008, overhang=0, color=clr8)
ax22i.text((ori1[ 6,0]+rad2/2+ori1[ 9,0]-rad1/2)/2+0.003, ori1[7,1]-0.0027, r'$\mathcal{J}$', ha='center', va='center', fontsize=6.5, color=clr8, fontdict=mathfont)
#ax22i.text((ori1[ 6,0]+rad2/2+ori1[ 9,0]-rad1/2)/2+0.003, ori1[7,1]-0.0027, '2', ha='center', va='center', fontsize=6.5, color=clr8)
#ax22i.add_artist(plt.Circle(((ori1[ 6,0]+rad2/2+ori1[ 9,0]-rad1/2)/2+0.003, ori1[7,1]), 0.016, lw=0.6, facecolor='none', edgecolor=clr8))
# P
ax22i.text((ori1[ 0,0]-pad+ori2[ 0,0])/2, ori2[2,1]+pad*1.7, r'$\mathcal{P}^{+}$', ha='center', va='center', color=clr3, fontsize=6.5, fontdict=mathfont)
ax22i.text((ori1[ 9,0]+ori2[12,0]+pad)/2, ori2[2,1]+pad*1.7, r'$\mathcal{P}^{-}$', ha='center', va='center', color=clr4, fontsize=6.5, fontdict=mathfont)
# MA
pad1 = rad1*1.2
ax22i.fill_between([ori1[3,0]-pad1, ori2[3,0]+pad1], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.2, fc='C7', zorder=0)
ax22i.fill_between([ori1[6,0]-el/4-pad1, ori2[6,0]+pad1], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.2, fc='C7', zorder=0)
ax22i.text((ori1[3,0]+ori2[3,0])/2,      ori2[2,1]+pad*1.7, r'$\mathcal{M}$', ha='center', va='center', fontsize=6.5, fontdict=mathfont)
ax22i.text((ori1[6,0]+ori2[6,0]-el/4)/2, ori2[2,1]+pad*1.7, r'$\mathcal{A}$', ha='center', va='center', fontsize=6.5, fontdict=mathfont)
# L
ax22i.plot([ori1[3,0]+el/4, ori1[3,0]+el/4], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax22i.plot([ori1[6,0]-el/4, ori1[6,0]-el/4], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax22i.plot([ori1[9,0],      ori1[9,0]     ], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax22i.text(ori1[3,0]-0.005+el/4, ori1[0,1]-pad*2.5, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax22i.text(ori1[6,0]-0.005-el/4, ori1[0,1]-pad*2.5, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax22i.text(ori1[9,0]-0.005,      ori1[0,1]-pad*2.5, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax22i.text(ori1[3,0]+0.015+el/4, ori1[0,1]-pad*2.5, r'$_1$',          ha='center', va='center', color=clr9, fontsize=6.5)
ax22i.text(ori1[6,0]+0.015-el/4, ori1[0,1]-pad*2.5, r'$_2$',          ha='center', va='center', color=clr9, fontsize=6.5)
ax22i.text(ori1[9,0]+0.015,      ori1[0,1]-pad*2.5, r'$_3$',          ha='center', va='center', color=clr9, fontsize=6.5)

# Step 3: Line-by-line
# Atoms
ori1[:,0] = ori1[:,0] - (1 - 6.5 * el)/2 + pad
ori2[:,0] = ori2[:,0] - (1 - 6.5 * el)/2 + pad
ori1[:,1] = ori1[:,1] - 0.05
ori2[:,1] = ori2[:,1] - 0.05
for i in range(len(ori1)):
    ax23i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad1, ls=dsh, lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range(len(ori2)):
    ax23i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad1, ls='-', lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range( 0,  6, 1):
    ax23i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
for i in range( 6, 15, 1):
    ax23i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
# Bonds
for i in range(len(ori1)):
    for j in range(len(ori2)):
        distance = np.sqrt((ori1[i,0] - ori2[j,0])**2 + (ori1[i,1] - ori2[j,1])**2)
        if distance < el * 1.1:
            ax23i.plot([ori1[i,0], ori2[j,0]], [ori1[i,1], ori2[j,1]], ls='-', lw=0.6, c='C7', alpha=0.6, zorder=0)
ax23i.fill_between([ori1[ 0,0]-pad, ori2[ 3,0]], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.10, fc=clr3, zorder=0)
ax23i.fill_between([ori1[ 6,0], ori2[12,0]+pad], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.10, fc=clr4, zorder=0)
# P
ax23i.text((ori1[ 0,0]-pad+ori2[ 3,0])/2, ori2[2,1]+pad*1.7, r'$\mathcal{P}^{+}$', ha='center', va='center', color=clr3, fontsize=6.5, fontdict=mathfont)
ax23i.text((ori1[ 6,0]+ori2[12,0]+pad)/2, ori2[2,1]+pad*1.7, r'$\mathcal{P}^{-}$', ha='center', va='center', color=clr4, fontsize=6.5, fontdict=mathfont)
# L
ax23i.plot([ori2[ 3,0], ori2[ 3,0]], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax23i.plot([ori1[ 6,0], ori1[ 6,0]], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax23i.plot([ori1[ 9,0], ori1[ 9,0]], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax23i.text(ori2[ 3,0]-0.005, ori1[0,1]-pad*2.0, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax23i.text(ori1[ 6,0]-0.005, ori1[0,1]-pad*2.0, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax23i.text(ori1[ 9,0]-0.005, ori1[0,1]-pad*2.0, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax23i.text(ori2[ 3,0]+0.015, ori1[0,1]-pad*2.0, r'$_1$',          ha='center', va='center', color=clr9, fontsize=6.5)
ax23i.text(ori1[ 6,0]+0.015, ori1[0,1]-pad*2.0, r'$_2$',          ha='center', va='center', color=clr9, fontsize=6.5)
ax23i.text(ori1[ 9,0]+0.015, ori1[0,1]-pad*2.0, r'$_3$',          ha='center', va='center', color=clr9, fontsize=6.5)

# Step 3: Stone-skipping
# Atoms
ori1[:,0] = ori1[:,0] + 1 - 2 * pad - 6.5 * el
ori2[:,0] = ori2[:,0] + 1 - 2 * pad - 6.5 * el
for i in range(0, 9, 1):
    ax23i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad1, ls=dsh, lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range(12, 15, 1):
    ax23i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad1, ls=dsh, lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range(len(ori2)):
    ax23i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad1, ls='-', lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
for i in range( 0,  6, 1):
    ax23i.add_patch(Circle((ori2[i,0], ori2[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
for i in range( 9, 15, 1):
    ax23i.add_patch(Circle((ori1[i,0], ori1[i,1]), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
for i in range(6, 9, 1):
    ax23i.add_patch(Circle((ori1[i,0]+el/4, ori1[i,1]+el*c/4), rad2, ls='-', lw=0.0, fc=clr2, ec='none', alpha=1.0, zorder=1))
for i in range(9, 12, 1):
    ax23i.add_patch(Circle((ori1[i,0]-el/4, ori1[i,1]-el*c/4), rad1, ls=dsh, lw=0.6, fc='w',  ec=clr1,   alpha=1.0, zorder=1))
# Bonds
for i in range(len(ori1)):
    for j in range(len(ori2)):
        distance = np.sqrt((ori1[i,0] - ori2[j,0])**2 + (ori1[i,1] - ori2[j,1])**2)
        if distance < el * 1.1:
            ax23i.plot([ori1[i,0], ori2[j,0]], [ori1[i,1], ori2[j,1]], ls='-', lw=0.6, c='C7', alpha=0.6, zorder=0)
ax23i.fill_between([ori1[ 0,0]-pad, ori2[ 3,0]], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.10, fc=clr3, zorder=0)
ax23i.fill_between([ori1[12,0], ori2[12,0]+pad], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.10, fc=clr4, zorder=0)
# P
ax23i.text((ori1[ 0,0]-pad+ori2[ 3,0])/2, ori2[2,1]+pad*1.7, r'$\mathcal{P}^{+}$', ha='center', va='center', color=clr3, fontsize=6.5, fontdict=mathfont)
ax23i.text((ori1[12,0]+ori2[12,0]+pad)/2, ori2[2,1]+pad*1.7, r'$\mathcal{P}^{-}$', ha='center', va='center', color=clr4, fontsize=6.5, fontdict=mathfont)
# MA
pad1 = rad1*1.2 
ax23i.fill_between([ori1[6,0]-pad1, ori2[6,0]+pad1], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.2, fc='C7', zorder=0)
ax23i.fill_between([ori1[9,0]-el/4-pad1, ori2[9,0]+pad1], ori1[0,1]-pad, ori2[2,1]+pad, alpha=0.2, fc='C7', zorder=0)
ax23i.text((ori1[6,0]+ori2[6,0])/2,      ori2[2,1]+pad*1.7, r'$\mathcal{M}$', ha='center', va='center', fontsize=6.5, fontdict=mathfont)
ax23i.text((ori1[9,0]+ori2[9,0]-el/4)/2, ori2[2,1]+pad*1.7, r'$\mathcal{A}$', ha='center', va='center', fontsize=6.5, fontdict=mathfont)
# L
ax23i.plot([ori2[3,0],      ori2[3,0]],      [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax23i.plot([ori1[6,0]+el/4, ori1[6,0]+el/4], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax23i.plot([ori1[9,0]-el/4, ori1[9,0]-el/4], [ori1[0,1]-pad, ori2[2,1]+pad], ls='--', c=clr9, zorder=0)
ax23i.text(ori2[3,0]-0.005,      ori1[0,1]-pad*2.0, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax23i.text(ori1[6,0]-0.005+el/4, ori1[0,1]-pad*2.0, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax23i.text(ori1[9,0]-0.005-el/4, ori1[0,1]-pad*2.0, r'$\mathcal{L}$', ha='center', va='center', color=clr9, fontsize=6.5, fontdict=mathfont)
ax23i.text(ori2[3,0]+0.015,      ori1[0,1]-pad*2.0, r'$_1$',          ha='center', va='center', color=clr9, fontsize=6.5)
ax23i.text(ori1[6,0]+0.015+el/4, ori1[0,1]-pad*2.0, r'$_2$',          ha='center', va='center', color=clr9, fontsize=6.5)
ax23i.text(ori1[9,0]+0.015-el/4, ori1[0,1]-pad*2.0, r'$_3$',          ha='center', va='center', color=clr9, fontsize=6.5)

# Arrows
ax2i = ax2.inset_axes([0, 0.1, 1, 1/(3*hw_ratio)], transform=ax2.transAxes)
ax2i.set(xlim=(0,1), ylim=(0,1))
ax2i.axis('off')
ax2i.arrow(0.50, 0.85, 0, -0.05, length_includes_head=True, lw=0, width=0.015, head_width=0.035, head_length=0.02, overhang=0, color='C7', alpha=0.8)
al, theta, pad2 = 0.06, 45, 0.07
ax2i.arrow(0.37, 0.34, -al*np.cos(np.pi/180*theta), -al*np.sin(np.pi/180*theta), length_includes_head=True, lw=0, width=0.015, head_width=0.035, head_length=0.02, overhang=0, color='C7', alpha=0.8, transform=ax2i.transAxes)
ax2i.arrow(0.63, 0.34,  al*np.cos(np.pi/180*theta), -al*np.sin(np.pi/180*theta), length_includes_head=True, lw=0, width=0.015, head_width=0.035, head_length=0.02, overhang=0, color='C7', alpha=0.8, transform=ax2i.transAxes)
ax2i.text(0.37-pad2, 0.31, 'Line-by-line',   ha='right', va='center', color=clr7, fontsize=6.5)
ax2i.text(0.63+pad2, 0.31, 'Stone-skipping', ha='left',  va='center', color=clr8, fontsize=6.5)
ax2i.text(0.22-0.0007, 0.36-0.0032, r'$\mathcal{I}$', ha='center', va='center', fontsize=6.5, color=clr7, fontdict=mathfont)
#ax2i.text(0.22-0.0007, 0.36-0.0032, '1', ha='center', va='center', fontsize=6.5, color=clr7)
#ax2i.add_artist(plt.Circle((0.22, 0.36), 0.016, lw=0.6, facecolor='none', edgecolor=clr7))
ax2i.text(0.76-0.0007, 0.36-0.0032, r'$\mathcal{I}$', ha='center', va='center', fontsize=6.5, color=clr7, fontdict=mathfont)
#ax2i.text(0.76-0.0007, 0.36-0.0032, '1', ha='center', va='center', fontsize=6.5, color=clr7)
#ax2i.add_artist(plt.Circle((0.76, 0.36), 0.016, lw=0.6, facecolor='none', edgecolor=clr7))
ax2i.text(0.805, 0.36, r'$+$', ha='center', va='center', fontsize=6.5, fontdict=mathfont)
ax2i.text(0.85, 0.36-0.0027, r'$\mathcal{J}$', ha='center', va='center', fontsize=6.5, color=clr8, fontdict=mathfont)
#ax2i.text(0.85, 0.36-0.0027, '2', ha='center', va='center', fontsize=6.5, color=clr8)
#ax2i.add_artist(plt.Circle((0.85, 0.36), 0.016, lw=0.6, facecolor='none', edgecolor=clr8))

# Save
fig.savefig('Fig2.jpg')
fig.savefig('Fig2.pdf')

