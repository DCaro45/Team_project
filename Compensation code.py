import numpy as np
import os
import matplotlib.pyplot as plt

dir, file = os.path.split(__file__)
file = (dir + "\\Data\\"
              "no1test22official.dat"
        )
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

def compensate(data, A, B):
    F = A*(1-np.exp(-data/B))
    return F


T = get_T(file)
R = get_R(file)
R = clean(R, 10 ** 3)

'''
L = len(T)  # number of data points
S = 515     # skip over the first S readings
E = 1056    # end at this reading

T = T[:E - S]
R = R[S:E]
R_err = 0.0001 * np.ones(len(R))
'''

L = len(T)   # number of data points
S_r = 20     # skip over the first S readings
E_r = 700    # end at this reading
S_t = 1
E_t = E_r - S_r + S_t


T = np.array(T[S_t:E_t])
R = np.array(R[S_r:E_r])
R_inv = 1/(np.array(R))

A = 0.007777826691491801
B = 40.416052730399066


R_inv_model= np.array(compensate(T, A, B))

R_comp = max(R) * R * R_inv_model


plt.plot(T, R, color = 'r', label = 'Raw data')
#plt.plot(T, R_inv, color='r', label = 'Inverse data')
#plt.plot(T, R_comp, color='b', label = 'Compensated data')
plt.plot(T, R_comp, color='g', label='constant data')
plt.legend()
plt.show()


