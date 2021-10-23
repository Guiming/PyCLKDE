# Author: Guiming Zhang
# Date: Feb. 13 2016
# Last update: Feb. 13 2017
import numpy as np

def write1DArray2CSV(data, filename):
    f = open(filename, 'w')
    for i in range(len(data)):
        f.write(str(data[i]) + '\n')
    f.close()

def write2DArray2CSV(data, filename):
    f = open(filename, 'w')
    for i in range(len(data)):
        row = str(data[i][0])
        for j in range(1, len(data[i])):
            row += ',' + str(data[i][j])
        f.write(row + '\n')
    f.close()

def read1DArrayfromCSV(filename):
    f = open(filename, 'r')
    data = []
    line = f.readline()
    while len(line) > 0:
        data.append(float(line.split('\n')[0]))
        line = f.readline()
    f.close()
    return np.array(data)

def read2DArrayfromCSV(filename):
    f = open(filename, 'r')
    data = []
    line = f.readline()
    while len(line) > 0:
        row = []
        vals = line.split('\n')[0].split(',')
        for val in vals:
            row.append(float(val))
        data.append(row)
        line = f.readline()
    f.close()
    return np.array(data)
