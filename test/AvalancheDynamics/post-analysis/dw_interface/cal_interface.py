import numpy as np
import pandas as pd
import open3d as o3d
import sys
import subprocess



# define variables
n_In = int(4000/5*2)
n_Se = int(4000/5*3)
m_se =int(4000/5*2)

#filename="dump4800_1.xsf"
filename =sys.argv[1]
inputfile  = filename

def get_cell_size(filename):
    f = open(filename, 'r')
    # skip first two lines
    f.readline()
    f.readline()
    # get line 3 4 5 data
    line3 = f.readline().split()
    line4 = f.readline().split()
    line5 = f.readline().split()

    size = [float(line3[0]), float(line4[1]), float(line5[2])]
    return size

if __name__=="__main__":
    # get cell_size
    cell_size = get_cell_size(filename)
    # print "cell size: x: %f, y: %f, z: %f " %(cell_size[0], cell_size[1], cell_size[2])
    # read data from file, save data as panda data frame, skip lines 0-5
    # column names: element x y z
    xsf_file = pd.read_csv(filename, sep=" ", skiprows=range(0, 7), names=["element", "x", "y", "z"])
    # filter out In/Se data
    In_file = xsf_file.query('element == "In"').iloc[:, 1:4]
    Se_file = xsf_file.query('element == "Se"').iloc[:, 1:4]

    # transform Pandas dataframe to numpy array
    In_data = np.array(In_file, dtype='float32')
    Se_data = np.array(Se_file, dtype='float32')
    Se_data = Se_data[800:m_se,:]
    # copy origin Se_data to enlarged data
    In_extrapolated_data = In_data.copy()
    Se_extrapolated_data = Se_data.copy()
    # print Se_extrapolated_data.shape

    fd = open("dxyz_%s"%inputfile,'w')
    fdx = open("dx_%s"%inputfile,'w')
    fdy = open("dy_%s"%inputfile,'w')
    fdz = open("dz_%s"%inputfile,'w')

    a = np.zeros((3, 3))
    a[0, 0] = cell_size[0]
    a[1, 1] = cell_size[1]
    a[2, 2] = cell_size[2]
    for f in [fd, fdz, fdy,fdx]:
        print("CRYSTAL", file=f)
        print("PRIMVEC", file=f)
        print(a[0, 0], a[0, 1], a[0, 2], file=f)
        print(a[1, 0], a[1, 1], a[1, 2], file=f)
        print(a[2, 0], a[2, 1], a[2, 2], file=f)
        print("PRIMCOORD", file=f)
        print(n_Se / 3, 1, file=f)

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
    print (Se_extrapolated_data.shape) # check if it is = number_Se *27 lines
    # numpy data to open3d file
    # initlise poincloud(x, y, z)  variable
    In_pcd = o3d.geometry.PointCloud()
    In_pcd.points = o3d.utility.Vector3dVector(In_extrapolated_data)

    Se_pcd = o3d.geometry.PointCloud()
    Se_pcd.points = o3d.utility.Vector3dVector(In_extrapolated_data) # From numpy to Open3D

    # create kd-tree
    pcd_tree = o3d.geometry.KDTreeFlann(In_pcd)

    Se_z = Se_data[:, 2]
    #Se_z = abs(Se_z - 3)  # Se_z = abs(Se_z - 24)  for beta3_prime
    middle_Se_index = np.where(Se_z)
    #middle_Se_index =np.where(Se_z<1)
    #Se_m = Se_data[101:201, 1]
    #middle_Se_index = np.where(Se_m )
    #print(middle_Se_index[0])
    print(middle_Se_index[0].shape)
    for index in middle_Se_index[0]:
        # [k, idx, _] = pcd_tree.search_radius_vector_3d(Se_pcd.points[index], 4)

        [k, idx, _] = pcd_tree.search_radius_vector_3d(Se_data[index, :], 3.6)  # search Se neighboring In atoms
        In_neighbour = In_extrapolated_data[idx]

   ## check the number of neighbours
        len_neigh = len(In_neighbour[:,0])
        print(len_neigh)
        #print(In_neighbour)
        #print("Se", index,len_neigh)
        if len_neigh > 6:
            print("Se", index,len_neigh)
            print(idx)


    ## calculating the dx dy dz
        upper_In_index = np.where(In_neighbour[:, 2] > Se_data[index, 2])
        lower_In_index = np.where(In_neighbour[:, 2] < Se_data[index, 2])
        upper_mean = np.mean(In_neighbour[upper_In_index, 2])
        lower_mean = np.mean(In_neighbour[lower_In_index, 2])
        upper_lower_mean = (upper_mean+lower_mean) / 2
        diff_z = upper_lower_mean - Se_data[index, 2]

        x_mean = np.mean(In_neighbour[:, 0])
        diff_x = x_mean - Se_data[index, 0]

        y_mean = np.mean(In_neighbour[:, 1])
        diff_y = y_mean -Se_data[index, 1]


        print("Se", Se_data[index, 0], Se_data[index, 1], Se_data[index, 2], diff_x, diff_y, diff_z, file=fd)
        print("Se", Se_data[index, 0], Se_data[index, 1], Se_data[index, 2], diff_x, "0.0", "0.0", file=fdx)
        print("Se", Se_data[index, 0], Se_data[index, 1], Se_data[index, 2], "0.0", diff_y, "0.0", file=fdy)
        print("Se", Se_data[index, 0], Se_data[index, 1], Se_data[index, 2], "0.0", "0.0", diff_z, file=fdz)