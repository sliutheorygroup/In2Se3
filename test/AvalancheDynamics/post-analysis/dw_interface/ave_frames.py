#!/usr/bin/python
import numpy as np
import sys
import subprocess


def readcell(f):
    line = f.readline().split()
    line = [ float(x) for x in line]
    xlo_b = line[0]
    xhi_b = line[1]
    xy = line[2]
    line = f.readline().split()
    line = [ float(x) for x in line]
    ylo_b = line[0]
    yhi_b = line[1]
    xz = line[2]
    line = f.readline().split()
    line = [ float(x) for x in line]
    zlo_b = line[0]
    zhi_b = line[1]
    yz = line[2]
    xlo = xlo_b - min(0.0, xy, xz, xy+xz)
    xhi = xhi_b - max(0.0, xy, xz, xy+xz)
    ylo = ylo_b - min(0.0, yz)
    yhi = yhi_b - max(0.0, yz)
    zlo = zlo_b
    zhi = zhi_b 
    xx = xhi - xlo
    yy = yhi - ylo
    zz = zhi - zlo 
    cell = np.empty((3,3))
    cell[0,:] = [xx, 0, 0]
    cell[1,:] = [xy, yy, 0]
    cell[2,:] = [xz, yz, zz]
    return cell 

def readatoms(f, natoms):
    coord = np.zeros((natoms, 3))
    for i in range(natoms):
        line = f.readline().split()
        tmp = [ float(x) for x in line[2:]]
        coord[i, 0] = tmp[0]
        coord[i, 1] = tmp[1]
        coord[i, 2] = tmp[2]
    return coord

#f="ele_f_1005.lammpstrj"

inputfile  = sys.argv[1]
interval_dump = 1
nstructures = subprocess.check_output(" grep TIMESTEP  %s | wc -l "%inputfile, shell=True)
nstructures = int(nstructures.decode().strip("\n"))
natoms=subprocess.check_output(" grep -A1 \"NUMBER OF ATOMS\" %s | tail -1 "%inputfile, shell=True)
#print (natoms)
natoms = int(natoms.decode().strip("\n"))

f = open(inputfile,'r')
for k in range(nstructures//interval_dump):

    rk = (k+1)*interval_dump
    # fi = open("%s_%d.xsf"%inputfile%k, 'w')
    fi = open(str(inputfile) + "_" + str(rk) + ".xsf", 'w')
    cell_ave = np.zeros((3,3))
    cell_tmp = np.zeros((3,3))
    coord_ave = np.zeros((natoms, 3))
    coord_tmp = np.zeros((natoms, 3))
    coord_ini = np.zeros((natoms, 3))

    a = []
    b = []
    c = []
    for i in range(interval_dump):
        for j in range(5):
            f.readline()
        cell_tmp = readcell(f)
        a.append(cell_tmp[0,0]/20.0)
        b.append(cell_tmp[1,1]/40.0)
        c.append(cell_tmp[2,2]/1.0)

        cell_ave += cell_tmp

        f.readline()
        if i == 0 :
            coord_tmp = readatoms(f, natoms)
            coord_ini = coord_tmp
            coord_ave += coord_tmp
        else:
            coord_tmp = readatoms(f, natoms)
            coord_diff = coord_tmp - coord_ini
            for kk in range(3):
                index1 = np.where(coord_diff[:,kk] > cell_tmp[kk,kk]*0.5)
                coord_tmp[index1, kk] -= cell_tmp[kk,kk]
                index2 = np.where(coord_diff[:,kk] < -cell_tmp[kk,kk]*0.5)
                coord_tmp[index2, kk] += cell_tmp[kk,kk]
            coord_ave += coord_tmp

    cell_ave =  cell_ave/interval_dump
    coord_ave = coord_ave/interval_dump

    print("CRYSTAL", file=fi)
    print("PRIMVEC", file=fi)
    print(cell_ave[0,0],cell_ave[0,1],cell_ave[0,2], file=fi)
    print(cell_ave[1,0],cell_ave[1,1],cell_ave[1,2], file=fi)
    print(cell_ave[2,0],cell_ave[2,1],cell_ave[2,2], file=fi)
    print("PRIMCOORD", file=fi)
    print(natoms, file=fi)
    for i in range(natoms//5*2):
        print("In", coord_ave[i, 0], coord_ave[i, 1],coord_ave[i, 2], file=fi)

    for i in range(natoms//5*3):
        print("Se", coord_ave[i+natoms//5*2, 0], coord_ave[i+natoms//5*2, 1],coord_ave[i+natoms//5*2, 2], file=fi)

    fi.close()
f.close()
