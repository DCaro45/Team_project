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
Temp = [25, 35, 45, 55, 65]

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
    "no1test55official_1.lvm", "no2test53official_1.lvm", "no3test55official_1.lvm", "no5test61official_1.lvm",
    "no6test55official_1.lvm", "no8test55official_1.lvm", "no9test55official_1.lvm", "no12test53official_1.lvm",
    "no14test55official_1.lvm", "no19test54official_1.lvm", "no29test53official_1.lvm"

],
[


]

]

repeat_number = [
    [1, 1, 1, 2, 1, 4, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

index_ref_S = [
#reference =[1,   2,   3,   5,   6,   8,   9,   12,  14,  19,  29]  # list of the reference number of the sensor
            [210, 326, 47,  26,  35,  120, 20,  20,  20,  12,  16],   #25
            [48,  25,  21,  31,  21,  27,  27,  24,  21,  54,  33],   #35
            [41,  37,  46,  16,  52,  15,  9,   17,  20,  20,  53],   #45
            [39,  20,  34,  18,  31,  38,  19,  29,  29,  36,  31]    #55
]
index_ref_E = [
#reference =[1,     2,    3,    5,    6,    8,    9,    12,   14,   19,   29]  # list of the reference number of the sensor
            [1003,  900,  711,  836,  852,  843,  857,  814,  862,  821,  831],  #25
            [753,   957,  638,  990,  883,  834,  870,  860,  926,  964,  866],  #35
            [1080,  937,  985,  2129, 1337, 1055, 941,  930,  1213, 924,  920],  #45
            [977,   793,  833,  1019, 954,  1101, 959,  867,  850,  874,  811]   #55
]




"""define functions"""
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

def find_sensor(files, N, R):
    # finds the number of the sensor from the file name
    name = 'no' + str(N) + 'test'

    file = []
    for f in files:
        if name in f:
            file.append(f)
    sensor = []
    for g in file:
        if 'official_' + str(R) in g:
            sensor = g
    index = files.index(sensor)

    return files[index], index


def remove(file, size):
    # removes large readings from the data
    cleaned = []
    s = size
    avg = sum(file)/len(file)
    for d in file:
        if 0 <= d <= s:
            cleaned.append(d)
        if d > s or d < 0:
            cleaned.append(avg)
    return cleaned

def find_largest_section_within_threshold(resistance_array, lower_threshold, upper_threshold):
    current_section_start = None
    largest_section_start = None
    largest_section_length = 0

    for i, val in enumerate(resistance_array):
        if lower_threshold <= val <= upper_threshold:
            if current_section_start is None:
                current_section_start = i
        else:
            if current_section_start is not None:
                current_section_length = i - current_section_start
                if current_section_length > largest_section_length:
                    largest_section_start = current_section_start
                    largest_section_length = current_section_length
                current_section_start = None

    # Check if the last section is the largest
    if current_section_start is not None:
        current_section_length = len(resistance_array) - current_section_start
        if current_section_length > largest_section_length:
            largest_section_start = current_section_start
            largest_section_length = current_section_length

    largest_section_end = largest_section_start + largest_section_length if largest_section_start is not None else None

    return largest_section_start + 5 , largest_section_end - 10 if largest_section_end is not None else None

def linear_S(t, *vals):
    F = 1/(vals[0] + vals[1]) + ( 1/vals[0] - 1/(vals[0] + vals[1]) ) * ( 1 - np.exp(- t * (vals[0] * vals[1])/(vals[2] * (vals[0] + vals[1]))) )
    return F

def Burger(t, *vals):
    F = 1/vals[0] + 1/vals[1] * (1 - np.exp(- (t - vals[1]/vals[2]))) + t / vals[3]
    return F


"""import data and plot for one sensor"""

i = Temp.index(T)
File = files[i]
S_ref = index_ref_S[i]
E_ref = index_ref_E[i]

sensor, i = find_sensor(File, N, R)
data = (dir + "\\Data\\" + str(T) + " Temp\\" + sensor)
T = get_T(data)
R = get_R(data)
S, E = S_ref[i], E_ref[i]   # S = start index of resistance data, E = end index of resistance data
#S, E = 1, -1



"""clean data"""
"specifying the start and end indices"

"slices the arrays according to the given indices"
T = np.array(T[S:E])
R = R[S:E]
R_inv = 1/(np.array(R))
R_err = 0.00001 * np.ones(len(R))      # error in R check
#plt.plot(T, R)
#plt.show()


"""clean data"""
a = 0
b = 3
c = 0
d = len(sensor_number)
indexes = np.zeros([len(Temp), len(sensor_number), 2])
for k, t in enumerate(Temp[a:b+1]):
    k = k + a
    for j, n in enumerate(sensor_number[c:d+1]):
        j = j + c
        R = repeat_number[k][j]
        sensor, i = find_sensor(files[k], n, R)
        file = (dir + "\\Data\\" + str(t) + " Temp\\" + sensor)
        T = np.array(get_T(file))
        R = np.array(get_R(file))
        first, last = find_largest_section_within_threshold(R, 0, 200)
        indexes[k][j] = [first, last]

"""
"show R v T plot of cleaned data"
a = 0
b = 3
c = 0
d = len(sensor_number)
for k, t in enumerate(Temp[a:b+1]):
    k = k + a
    for j, n in enumerate(sensor_number[c:d+1]):
        j = j + c
        R = repeat_number[k][j]
        print('Num = ' + str(n) + ', ' + 'Repeat = ' + ', ' + str(R) + ' ,' +  'Temp = ' + str(t) + ',' + 'index = '
              + str(k) + ' ,' + str(j))
        sensor, i = find_sensor(files[k], n, R)
        file = (dir + "\\Data\\" + str(t) + " Temp\\" + sensor)
        #S, E = 0, -1
        S, E = index_ref_S[k][i], index_ref_E[k][i]   # S = start index of resistance data, E = end index of resistance data
        #S, E = int(indexes[k][j][0]), int(indexes[k][j][1])
        T = get_T(file)
        R = get_R(file)
        T = np.array(T[S:E])
        T_off = T - T[0]
        T = T_off
        R = np.array(R[S:E])
        R = 1/R
        grad = np.gradient(R)
        #mask_g = (grad < 0.0001) & (grad > 0)
        print(grad)
        index = 0
        for i, g in enumerate(grad):
            if  0 < g < 10**(-7):
                index = i
                break
        print(index)
        R = R[index:]
        T = T[index:]
        assert len(T) == len(R)
        if k == 0 and j == 5:
            R = remove(R, 110)
        #plt.plot(T, grad)
        plt.plot(T, R)
        plt.show()
"""
"""
a = 0
b = 3
c = 0
d = len(sensor_number)
for j, n in enumerate(sensor_number[c:d+1]):
    j = j + c
    for k, t in enumerate(Temp[a:b+1]):
        k = k + a
        T = t
        R = repeat_number[k][j]
        sensor, i = find_sensor(files[k], n, R)
        file = (dir + "\\Data\\" + str(T) + " Temp\\" + sensor)
        #S, E = index_ref_S[k][i], index_ref_E[k][i]   # S = start index of resistance data, E = end index of resistance data
        S, E = int(indexes[k][j][0]), int(indexes[k][j][1])
        T = get_T(file)
        R = get_R(file)
        T = np.array(T[S:E])
        R = np.array(R[S:E])
        T_off = T - T[0]
        T = T_off
        plt.title('no' + str(n))
        plt.plot(T, R, label = str(t) + 'C')
    plt.legend()
    plt.show()
"""
model_funct = linear_S   # choose the model function to fit the data

"initial parameters"
A = (88.0036163282118)

B = (11)

C = (2385.010449469219)

initial = np.array([A, B, C])
param = np.zeros([len(Temp), len(sensor_number), 3])
a = 0
b = 1
for k, t in enumerate(Temp[a:b+1]):
    k = k + a
    for j, n in enumerate(sensor_number):
        R = repeat_number[k][j]
        sensor, i = find_sensor(files[k], n, R)
        file = (dir + "\\Data\\" + str(t) + " Temp\\" + sensor)
        S, E = index_ref_S[k][i], index_ref_E[k][i]  # S = start index of resistance data, E = end index of resistance data
        T = get_T(file)
        R = get_R(file)
        T = np.array(T[S:E])
        R = np.array(R[S:E])

        T_off = T - T[0]
        T = T_off
        R = 1/(np.array(R))

        grad = np.gradient(R)
        index = 0
        for i, g in enumerate(grad):
            if  0 < g < 10**(-7):
                index = i
                break
        R = R[index:]
        T = T[index:]
        R_err = 0.00001 * np.ones(len(T))


        xval = T
        yval = R
        yerr = R_err

        popt, cov = scipy.optimize.curve_fit(model_funct,  # function to fit
                                             xval,  # x data
                                             yval,  # y data
                                             sigma=yerr,  # set yerr as the array of error bars for the fit
                                             absolute_sigma=True,  # errors bars DO represent 1 std error
                                             p0=initial,  # starting point for fit
                                             check_finite=True,
                                             bounds=(0, 10000) )   # provides bounts to the optimised parameters, use if 'raise ValueError if NaN' encountered
        '''
        plt.figure(figsize=[10, 6])
        plt.errorbar(xval, yval, yerr=yerr, marker='o', ms=1, linestyle='None', label='Data')
        plt.plot(xval,
                 model_funct(xval, *popt),
                 'r', label='Model function')
        plt.xlabel('Time')
        plt.ylabel('1/R')
        #plt.ylim(0, R_inv[-1])
        #plt.xlim(0, T[-1])
        plt.legend()
        plt.show()
        '''
        param[k][j] = popt

a = 0
b = 3
c = 0
d = len(sensor_number)
for j, n in enumerate(sensor_number[c:d+1]):
    j = j + c
    for k, t in enumerate(Temp[a:b+1]):
        k = k + a
        A = param[k][j][0]
        B = param[k][j][1]
        C = param[k][j][2]
        R = repeat_number[k][j]
        plt.figure(figsize=[10, 6])
        plt.title('no' + str(n))
        plt.scatter(t, A, label = 'ro')
        plt.scatter(t, B, label = 'bo')
        plt.scatter(t, C, label = 'go')
    plt.legend()
    plt.show()