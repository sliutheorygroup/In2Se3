import numpy as np
import pandas as pd
import open3d as o3d
import sys
import subprocess
import matplotlib.pyplot as plt


# define variables
n_In = int(4000/5*2)
n_Se = int(4000/5*3)
m_se =int(4000/5*2)
singleLayer =int(4000/5)

filenames = []
aveg_y = []
y_all = []
y_err=[]
y_out = []
interval_dump =40
init_dump = interval_dump
tot_dump = 10000
consider_dump = tot_dump + interval_dump

for m in range(init_dump, consider_dump,interval_dump):
    filename = f"dump_e020_{str(m)}.xsf"  
    filenames.append(filename)

def get_cell_size(filename):
    f = open(filename, 'r')
    # skip first five lines
    f.readline()
    f.readline()

    # get line 3 4 5 data
    line3 = f.readline().split()
    line4 = f.readline().split()
    line5 = f.readline().split()

    size = [float(line3[0]), float(line4[1]), float(line5[2])]
    return size

def find_max_y_within_range(arr, interval):
    x_min = 0
    x_max = np.max(arr[:, 0])  

    max_y_values = []

    for x_start in np.arange(x_min, x_max, interval):
        x_end = x_start + interval

        max_y = float('-inf')
        max_x = None
        max_z = None

        for data in arr:
            x = data[0]
            y = data[1]
            z = data[2]

            if x >= x_start and x < x_end:
                if y >= max_y:
                    max_y = y
                    max_x = x
                    max_z = z

        max_y_values.append((max_y, max_x, max_z))

    return max_y_values


if __name__=="__main__":
     fall = open("all_trajectory.txt", 'w')
     for filename in filenames:
        inputfile = filename
        # get cell_size
        cell_size = get_cell_size(filename)
        # column names: element x y z
        lmp_file = pd.read_csv(filename, sep=" ", skiprows=range(0, 7), names=["element", "x", "y", "z"])
        # filter out In/Se data
        In_file = lmp_file.query('element == "In"').iloc[:, 1:4]
        Se_file = lmp_file.query('element == "Se"').iloc[:, 1:4]

        # transform Pandas dataframe to numpy array
        In_data = np.array(In_file, dtype='float32')
        Se_data = np.array(Se_file, dtype='float32')
        Se_data = Se_data[singleLayer:m_se,:]
        # copy origin Se_data to enlarged data
        In_extrapolated_data = In_data.copy()
        Se_extrapolated_data = Se_data.copy()
        fdw = open("dw_%s" % inputfile, 'w')

        a = np.zeros((3, 3))
        a[0, 0] = cell_size[0]
        a[1, 1] = cell_size[1]
        a[2, 2] = cell_size[2]
        for f in [fdw]:
            print("CRYSTAL", file=f)
            print("PRIMVEC", file=f)
            print(a[0, 0], a[0, 1], a[0, 2], file=f)
            print(a[1, 0], a[1, 1], a[1, 2], file=f)
            print(a[2, 0], a[2, 1], a[2, 2], file=f)
            print("PRIMCOORD", file=f)
            print(n_Se / 3,  file=f)

        # exprapolate neighbour data
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    if i ==0 and j==0 and k==0:
                        continue
                    Se_neighbour_data = Se_data + np.array([i*cell_size[0], j*cell_size[1], k*cell_size[2]])
                    In_neighbour_data = In_data + np.array([i*cell_size[0], j*cell_size[1], k*cell_size[2]])
                    # append
                    Se_extrapolated_data = np.append(Se_extrapolated_data, Se_neighbour_data, axis=0)
                    In_extrapolated_data = np.append(In_extrapolated_data, In_neighbour_data, axis=0)
        In_pcd = o3d.geometry.PointCloud()
        In_pcd.points = o3d.utility.Vector3dVector(In_extrapolated_data)

        Se_pcd = o3d.geometry.PointCloud()
        Se_pcd.points = o3d.utility.Vector3dVector(In_extrapolated_data) # From numpy to Open3D

        # create kd-tree
        pcd_tree = o3d.geometry.KDTreeFlann(In_pcd)

        Se_z = Se_data[:, 2]
        middle_Se_index = np.where(Se_z)
        #print(middle_Se_index[0].shape)
        xdata=[];ydata = [];zdata=[];xresult=[];yresult=[];zresult=[];yplot=[]
        data = []
        for index in middle_Se_index[0]:
            [k, idx, _] = pcd_tree.search_radius_vector_3d(Se_data[index, :], 4.2)  # search Se neighboring In atoms
            In_neighbour = In_extrapolated_data[idx]

            ## check the number of neighbours
            len_neigh = len(In_neighbour[:,0])

            if len_neigh > 4:
                 
                x = np.array(Se_data[index, 0])
                y = np.array(Se_data[index, 1])
                z = np.array(Se_data[index, 2])
                xdata = np.append(xdata, x)
                xresult = np.reshape(xdata, (-1, 1))
                ydata = np.append(ydata, y)
                yresult = np.reshape(ydata, (-1, 1))
                zdata = np.append(zdata, z)
                zresult = np.reshape(zdata, (-1, 1))
                #print(yresult)
                data = np.concatenate((xresult, yresult,zresult), axis=1)
        print("data shape: ", data.shape)
        interval = 4.0

        max_y_values = find_max_y_within_range(data, interval)
        print("lens of max_y_values: ", len(max_y_values))

        for max_y, max_x, max_z in max_y_values:   
            #if max_x is not None:
            print(f"Se  {max_x}  {max_y}  {max_z}",file=fdw)
            if max_y<114 and max_x is not None:
                y0 = max_y + cell_size[1]
                yplot = np.append(yplot, y0)

        print("lens of yplot: ", len(yplot))
        aveg_y = np.average(yplot)
        std_y = np.nanstd(yplot)
        print("aveg_y: ", aveg_y)
        print("cell_size[1]: ", cell_size[1])

        y_all = np.append(y_all, aveg_y- cell_size[1])
        y_true = aveg_y- cell_size[1]
        y_err = np.append(y_err,std_y)
        np.savetxt('position_value.out', y_all)
        np.savetxt('position_err.out', y_err)
