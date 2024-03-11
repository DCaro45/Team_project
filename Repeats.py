import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams["font.family"]="serif"
plt.rcParams["mathtext.fontset"]="dejavuserif"


dir, file = os.path.split(__file__)

''' the numbers of the sensor and repeat (usually one) writen in format 'sensor no.''repeat no.'
    written in format so comp can read'''

N = 1
Tp = 25
R_threshold = 200
U = R_threshold

sensor_number = [1, 2, 3, 5, 6, 8, 9, 12, 14, 19, 29]
Temp = [25, 35, 45, 55, 65]

files = [
    [
        "no1test23official.lvm", "no2test23official.lvm", "no3test23official.lvm", "no5test25official.lvm",
        "no6test26official.lvm", "no8test23official.lvm", "no9test25official.lvm", "no12test27official.lvm",
        "no14test21official.lvm", "no19test24official.lvm", "no29test24official.lvm"
    ],
    [
        "no1test34official.lvm", "no2test35official.lvm", "no3test35official.lvm", "no5test35official.lvm",
        "no6test35official.lvm", "no8test35official.lvm", "no9test35official.lvm", "no12test35official.lvm",
        "no14test35official.lvm", "no19test35official.lvm", "no29test35official.lvm"
    ],
    [
        "no1test45official.lvm", "no2test45official.lvm", "no3test45official.lvm", "no5test45official.lvm",
        "no6test45official.lvm", "no8test45official.lvm", "no9test45official.lvm", "no12test45official.lvm",
        "no14test45official.lvm", "no19test45official.lvm", "no29test45official.lvm"
    ],
    [
        "no1test55official.lvm", "no2test53official.lvm", "no3test55official.lvm", "no5test61official.lvm",
        "no6test55official.lvm", "no8test55official.lvm", "no9test55official.lvm", "no12test53official.lvm",
        "no14test55official.lvm", "no19test54official.lvm", "no29test53official.lvm"
    ],
    [
        "no1test64official.lvm", "no2test65official.lvm", "no3test64official.lvm", "no5test65official.lvm",
        "no6test64official.lvm", "no8test65official.lvm", "no9test65official.lvm", "no12test66official.lvm",
        "no14test66official.lvm", "no19test66official.lvm", "no29test66official.lvm"
    ]
]

Repeats = [
    [],
    [],
    [],
    [],
    []
]

"""define functions"""


def get_t(file):
    # np.array of extracted time of dat file
    data_t = np.loadtxt(file)
    t = data_t[:, 0]
    return t


def get_R(file):
    # np.array of extracted resistance of dat file
    data_R = np.loadtxt(file)
    R = data_R[:, 1]
    return R


def find_sensor(files, N):
    # finds the number of the sensor from the file name
    name = 'no' + str(N) + 'test'
    file = None
    index = 0
    T = 0

    for i, f in enumerate(files):
        if name in f:
            file = f
            index = i
    if 'test' in str(file):
        i = file.index('test')
        T = file[i + 4:i + 6]


    return file, index, T


def remove(file, size):
    # removes large readings from the data
    cleaned = []
    s = size
    avg = sum(file) / len(file)
    for d in file:
        if 0 <= d <= s:
            cleaned.append(d)
        if d > s or d < 0:
            cleaned.append(avg)
    return cleaned


def find_value(data, value):
    # if you want to find a specific value in the file
    positions = []
    for i, d in enumerate(data):
        if float('%.2g' % d) == value:
            positions.append(i)
            break
    return positions[0]


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

    return largest_section_start, largest_section_end - 20 if largest_section_end is not None else None


def Norm_R(y, y_mdl, yerr):
    N_R = (y - y_mdl) / yerr
    return N_R


def chisq(model_params, model, x_data, y_data, y_err):
    chi_sq = np.sum(((y_data - model(x_data, *model_params)) / y_err) ** 2)
    return chi_sq


def kelvin(t, *vals):
    F = 1 / vals[0] * (1 - np.exp(-vals[0] / vals[1] * t))
    return F


def linear_S(t, *vals):
    F = 1 / (vals[0] + vals[1]) + (1 / vals[0] - 1 / (vals[0] + vals[1])) * (
                1 - np.exp(- t * (vals[0] * vals[1]) / (vals[2]
                                                        * (vals[0] + vals[1]))))
    return F


def linear_S_decay(t, *vals):
    F = linear_S(t, *vals[:3]) - vals[3] * np.exp(- vals[4] * (t - vals[5]))
    return F


def linear_kelvin(t, *vals):
    F = linear_S(t, *vals[:3]) + kelvin(t, *vals[3:])
    return F


def Burger(t, *vals):
    F = 1 / (vals[0]) + (1 / vals[1]) * (1 - np.exp(- (vals[1] * t) / vals[2])) + t / vals[3]
    return F


"""clean data"""
a = 0
b = 0
c = 7
d = 8
indexes = np.zeros([len(Temp), len(sensor_number), 2])
for k, T in enumerate(Temp[a:b + 1]):
    k = k + a
    for j, n in enumerate(sensor_number[c:d + 1]):
        j = j + c
        sensor, i, temp = find_sensor(files[k], n)
        file = (dir + "\\Repeats\\" + sensor)
        print('Num = ' + str(n) + ', ' + 'Temp = ' + str(temp) + ', ' + 'index = '
              + str(k) + ',' + str(j))
        t = np.array(get_t(file))
        R = np.array(get_R(file))
        first, last = find_largest_section_within_threshold(R, 0, U)
        indexes[k][j] = [first, last]
print('cleaning done')
print(indexes)

"""import data and plot for one sensor"""

i = Temp.index(Tp)
j = sensor_number.index(N)
File = files[i]
S, E = indexes[i][j][0], indexes[i][j][1]
# S, E = 1, -1
S, E = int(S), int(E)

sensor, i, temp = find_sensor(File, N)
data = (dir + "\\Data\\" + str(Tp) + " Temp\\" + sensor)
t = get_t(data)
R = get_R(data)
t = np.array(t[S:E])
R = R[S:E]
R_inv = 1 / (np.array(R))
R_err = 0.00001 * np.ones(len(R))  # error in R check

# plt.plot(t, R)
# plt.plot(t, R_inv)
# plt.show()

"show R v t plot of cleaned data"

for k, T in enumerate(Temp[a:b+1]):
    k = k + a
    for j, n in enumerate(sensor_number[c:d+1]):
        j = j + c
        sensor, i, temp = find_sensor(files[k], n)
        file = (dir + "\\Repeats\\" + sensor)
        print('Num = ' + str(n) + ', ' + 'Temp = ' + str(temp) + ', ' + 'index = '
              + str(k) + ',' + str(j))
        #S, E = 0, -1
        S, E = indexes[k][j][0], indexes[k][j][1]
        S, E = int(S), int(E)
        t = get_t(file)
        R = get_R(file)
        t = np.array(t[S:E])
        R = np.array(R[S:E])
        print(S, E)
        print(t, R)
        t_off = t - t[0]
        t = t_off
        if j == 8:
            R = R + 2
        R = 1/R

        plt.plot(t, R, label=str(n))
    #plt.legend()
    plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
    plt.tick_params(bottom=True, top=False, left=True, right=False)
    plt.xlabel('Time (s)')
    plt.ylabel('Resistance (Ω)')
    plt.savefig(dir + '\\Graphs\\Sensor 12 Repeat.png')
    plt.show()


"""
"show R v t over range of temp plot"
a = 0
b = 0
c = 0
d = 0
for j, n in enumerate(sensor_number[c:d+1]):
    j = j + c
    for k, T in enumerate(Temp[a:b+1]):
        k = k + a
        sensor, i, temp = find_sensor(files[k], n)
        print('Num = ' + str(n) + ', ' + 'Temp = ' + str(temp) + ', ' + 'index = '
              + str(k) + ',' + str(j))
        file = (dir + "\\Data\\" + str(T) + " Temp\\" + sensor)
        S, E = indexes[k][j][0], indexes[k][j][1]
        S, E = int(S), int(E)
        t = get_t(file)
        R = get_R(file)
        t = np.array(t[S:E])
        R = np.array(R[S:E])
        t_off = t - t[0]
        t = t_off
        plt.title('no' + str(n))
        plt.plot(t, R, label = str(temp) + 'C')
    plt.legend()
    plt.show()
"""


"show 1/R v t over all temp and sensors plot"
a = 0
b = len(Temp)
c = 0
d = len(sensor_number)
cmap = 'inferno_r'
cmin, cmax = 23, 66
N = sum(len(x) for x in files)
N = 1000

plt.figure(figsize=([12, 6]))
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
plt.tick_params(bottom=True, top=False, left=True, right=False)
cmap = plt.get_cmap(cmap)(np.linspace(0.1, 0.9, N + 1))
norm = plt.Normalize(cmin, cmax)
for j, n in enumerate(sensor_number[c:d+1]):
    j = j + c
    for k, T in enumerate(Temp[a:b+1]):
        k = k + a
        sensor, i, temp = find_sensor(files[k], n)
        print('Num = ' + str(n) + ', ' + 'Temp = ' + temp + ', ' + 'index = '
              + str(k) + ',' + str(j))
        temp = int(temp)
        file = (dir + "\\Data\\" + str(T) + " Temp\\" + sensor)
        S, E = indexes[k][j][0], indexes[k][j][1]
        S, E = int(S), int(E)
        t = get_t(file)
        R = get_R(file)
        t = np.array(t[S:E])
        R = np.array(R[S:E])
        R = 1/R
        t_off = t - t[0]
        t = t_off
        perc = int((temp - cmin)/(cmax - cmin) * N)
        plt.scatter(t, R, s=0.5, c=cmap[perc], label=str(temp) + 'C')
#plt.legend()
plt.colorbar(cm.ScalarMappable(norm=norm, cmap='inferno_r'), label='Temperature (C)', ax=plt.gca())
plt.xlabel('Time (s)')
plt.ylabel('$\dfrac{1}{R}$ $(Ω^{-1})$')
plt.savefig(dir + '\\Graphs\\All Sensors.png')
plt.show()
