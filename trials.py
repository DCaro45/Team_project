import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize

dir, file = os.path.split(__file__)

''' the numbers of the sensor and repeat (usually one) writen in format 'sensor no.''repeat no.'
    written in format so comp can read'''

N = 1
R = 1
T = 25

sensor_number = [1, 2, 3, 5, 6, 8, 9, 12, 14, 19, 29]
Temp = [25, 35, 45, 55]

files = [
[
    "no1test23official_1.lvm", "no2test23official_1.lvm", "no3test23official_1.lvm", "no5test25official_2.lvm",
    "no6test26official_1.lvm", "no8test23official_4.lvm", "no9test25official_1.lvm", "no12test27official_1.lvm",
    "no14test23official_1.lvm", "no19test24official_1.lvm", "no29test24official_1.lvm"
],
[
    "no1test34official_1.lvm", "no2test35official_1.lvm", "no3test35official_1.lvm", "no5test35official_1.lvm",
    "no6test35official_1.lvm", "no8test35official_1.lvm", "no9test35official_1.lvm", "no12test35official_1.lvm",
    "no14test35official_1.lvm", "no19test35official_1.lvm", "no29test35official_1.lvm"
],
[
    "no1test45official_1.lvm", "no2test45official_1.lvm", "no3test45official_1.lvm", "no5test45official_1.lvm",
    "no6test45official_1.lvm", "no8test45official_1.lvm", "no9test45official_1.lvm", "no12test45official_1.lvm",
    "no14test45official_1.lvm", "no19test45official_1.lvm", "no29test45official_1.lvm"
],
[
    "no1test55official_1.lvm", "no2test55official_1.lvm", "no3test55official_1.lvm", "no5test61official_1.lvm",
    "no6test55official_1.lvm", "no8test55official_1.lvm", "no9test55official.lvm", "no12test53official_1.lvm",
    "no14test55official_1.lvm", "no19test54official_1.lvm", "no29test53official_1.lvm"

]
]

repeat_number = [
    [1, 1, 1, 2, 1, 4, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

index_ref_S = [
#reference =[1,   2,   3,   5,  6,  8,   9,  12,  14,  19,  29]  # list of the reference number of the sensor
            [210, 326, 47, 26,  35, 120, 20, 20,  20,  12,  16], #25
            [48,   0,   0,   0,  0,  0,  0, 0,  0,  0,  0],      #35
            [41,   0,   0,   0,  0,  0,  0, 0,  0,  0,  0],      #45
            [39,   0,   0,   0,  0,  0,  0, 0,  0,  0,  0]       #55
]
index_ref_E = [
#reference =[1,    2,    3,   5,   6,   8,   9,   12,  14,  19,  29]  # list of the reference number of the sensor
            [1003, 900,  711, 396, 852, 843, 857, 814, 862, 821, 831],  #25
            [753,    0,    0,   0,   0,   0,   0,   0,   0,   0,   0],  #35
            [1080,    0,    0,   0,   0,   0,   0,   0,   0,   0,   0], #45
            [977,    0,    0,   0,   0,   0,   0,   0,   0,   0,   0]  #55
]




"""define functions"""
def find_sensor(files, N, R):
    # finds the number of the sensor from the file name
    name = 'no' + str(N) + 'test'

    index = 0
    file = []
    for i, f in enumerate(files):
        if name in f:
            file.append(f)
    File = []
    for f in file:
        if 'official_' + str(R) in f:
            File = f
    index = files.index(File)

    return files[index], index


def get_T(file):
    # np.array of extracted time of dat file
    data_T = np.loadtxt(file)
    T = data_T[:, 0]
    return T


def get_R(file):
    # np.array of extracted resistance of dat file
    data_R = np.loadtxt(file)
    R = data_R[:, 1]
    return R

def remove(file, size):
    # removes large readings from the data
    cleaned = []
    s = size

    for d in file:
        if 0 < d < s:
            cleaned.append(d)
        if d > s:
            cleaned.append(0)
    return cleaned

def find_value(file, value):
    # if you want to find a specific value in the file
    positions = []
    for i, d in enumerate(file):
        if float('%.2g' % d) == value:
            positions.append(i)
    return positions[0]

"""import data"""
idx = Temp.index(T)
Files = files[idx]
S_ref = index_ref_S[idx]
E_ref = index_ref_E[idx]

sensor, i = find_sensor(Files, N, R)
data = (dir + "\\Data\\" + str(T) + " Temp\\" + sensor)
S, E = S_ref[i], E_ref[i]   # S = start index of resistance data, E = end index of resistance data
#S, E = 1, -1
E_T = E - S + 1             # E_T = end index of time data
#E_T = -1



T = get_T(data)
R = get_R(data)

"""clean data"""
"specifying the start and end indices"

"slices the arrays according to the given indices"
T = np.array(T[1:E_T])
R = R[S:E]
R_inv = 1/(np.array(R))
R_err = 0.00001 * np.ones(len(R))      # error in R check
#plt.plot(T, R)
#plt.show()

"show R v T plot of cleaned data"
a = 0
b = 3
'''
for k, t in enumerate(Temp[a:b+1]):
    k = k + a
    for j, n in enumerate(sensor_number):
        print(k, j)
        T = t
        R = repeat_number[k][j]
        print('Num = ' + str(n) + ', ' + 'Repeat = ' + ', ' + str(R) + ' ,' +  'Temp = ' + str(T) )# + ', ' + 'index = ' + str(j))
        sensor, i = find_sensor(files[k], n, R)
        print(sensor)
        file = (dir + "\\Data\\" + str(T) + " Temp\\" + sensor)
        #S, E = 0, -1
        S, E = index_ref_S[k][i], index_ref_E[k][i]   # S = start index of resistance data, E = end index of resistance data
        T = get_T(file)
        R = get_R(file)
        T = np.array(T[S:E])
        R = np.array(R[S:E])
        if j == 5 and k == 0:
            R = remove(R, 105)
            print('removed')
        plt.plot(T, R)
        plt.show()
a = 0
for j, n in enumerate(sensor_number):
    for k, t in enumerate(Temp[a:]):
        k = k + a
        print(k, j)
        T = t
        R = repeat_number[k][j]
        print('Num = ' + str(n) + ', ' +  'Temp = ' + str(T) + ', ' + 'Repeat = ' + str(R))# + ', ' + 'index = ' + str(j))
        sensor, i = find_sensor(files[k], n, R)
        print(sensor)
        file = (dir + "\\Data\\" + str(T) + " Temp\\" + sensor)
        #S, E = 0, -1
        S, E = index_ref_S[k][i], index_ref_E[k][i]   # S = start index of resistance data, E = end index of resistance data
        T = get_T(file)
        R = get_R(file)
        T = np.array(T[S:E])
        R = np.array(R[S:E])
        plt.plot(T, R, label = str(t) + 'C')
    plt.legend()
    plt.show()
    
'''

"""starting chi_sq optimisation"""


def linear_S(t, *vals):
    F = 1/(vals[0] + vals[1]) + ( 1/vals[0] - 1/(vals[0] + vals[1]) ) * ( 1 - np.exp(- t * (vals[0] * vals[1])/(vals[2] * (vals[0] + vals[1]))) )
    return F


model_funct = linear_S    # choose the model function to fit the data

"initial parameters"
a = (88.0036163282118)

b = (-1207.2183402614955)

c = (2385.010449469219)

initial = np.array([a, b, c])
param = np.zeros([len(Temp), len(sensor_number), 3])
s = 0
e = 0
for k, t in enumerate(Temp[s:e+1]):
    k = k + s
    for j, n in enumerate(sensor_number):
        T = t
        R = repeat_number[k][j]
        sensor, i = find_sensor(files[k], n, R)
        file = (dir + "\\Data\\" + str(T) + " Temp\\" + sensor)
        S, E = index_ref_S[k][i], index_ref_E[k][i]  # S = start index of resistance data, E = end index of resistance data
        T = get_T(file)
        R = get_R(file)
        T = np.array(T[S:E])
        R = np.array(R[S:E])

        o = 5
        T_off = T - T[0] + o
        T = T_off
        R_inv = 1/(np.array(R))
        R_err = 0.00001 * np.ones(len(T))
        
        xval = T
        yval = R_inv
        yerr = R_err

        popt, cov = scipy.optimize.curve_fit(model_funct,  # function to fit
                                             xval,  # x data
                                             yval,  # y data
                                             sigma=yerr,  # set yerr as the array of error bars for the fit
                                             absolute_sigma=True,  # errors bars DO represent 1 std error
                                             p0=initial,  # starting point for fit
                                             check_finite=True)
        # bounds=((0,, ),(10e+10,np.inf, ))) # provides bounts to the optimised parameters, use if 'raise ValueError if NaN' encountered
        param[k][j] = popt
        
        """plots the optimised model function against data"""
        "plot of 1/R"
        plt.figure(figsize=[10, 6])
        plt.errorbar(xval, yval, yerr=yerr, marker='o', ms=1, linestyle='None', label='Data')
        plt.plot(xval,
                    model_funct(xval, *popt),
                    'r', label='Model function')
        plt.xlabel('Time')
        plt.ylabel('1/R')
        plt.ylim(0,R_inv[-1])
        plt.xlim(0,T[-1])
        plt.legend()
        plt.show()

print(param[0])
