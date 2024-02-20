import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize

dir, file = os.path.split(__file__)
file = (dir + "\\Data\\"
#              "test_23.dat"
        "official_test1_22C-159.5g.dat"
        )

def get_usec(file):
    # np.array of length of microsec signals
    data_microsec = np.loadtxt(file)
    return data_microsec[:, 0]


def get_F(file):
    # np.array of the force on load cell per reading N? per second
    data_F = np.loadtxt(file)
    L = len(data_F)
    print(L)
    return data_F[:, 1]


def get_T(file):
    # np.array of the time in the resistance reading
    data_T = np.loadtxt(file)
    T = data_T[:, 0]
    return T


def get_R(file):
    # np.array of the resistance of the sensor
    data_R = np.loadtxt(file)
    R = data_R[:, 1]
    return R


def clean(file, size):
    # cleaning the data of large readings
    cleaned = []
    s = size

    for d in file:
        if -s < d < s:
            cleaned.append(d)
        if d > s or d < - s:
            cleaned.append(0)
    return cleaned

def find_value(file, value):
    positions = []
    for i, d in enumerate(file):
        if float('%.2g' % d) == value:
            positions.append(i)
    return positions[0]


T = get_T(file)
R = get_R(file)
#R = clean(R, 10 ** 3)

plt.plot(T,R)
plt.show()
