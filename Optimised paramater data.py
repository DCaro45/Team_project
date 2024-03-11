import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import matplotlib.cm as cm
import pandas as pd
from matplotlib import rc
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


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
        "no14test23official.lvm", "no19test24official.lvm", "no29test24official.lvm"
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


model_funct = linear_S_decay # choose the model function to fit the data


"""clean data"""
a = 0
b = len(Temp)
c = 0
d = len(sensor_number)
indexes = np.zeros([len(Temp), len(sensor_number), 2])
for k, T in enumerate(Temp[a:b + 1]):
    k = k + a
    for j, n in enumerate(sensor_number[c:d + 1]):
        j = j + c
        sensor, i, temp = find_sensor(files[k], n)
        file = (dir + "\\Data\\" + str(T) + " Temp\\" + sensor)
        t = np.array(get_t(file))
        R = np.array(get_R(file))
        first, last = find_largest_section_within_threshold(R, 0, U)
        indexes[k][j] = [first, last]
print('cleaning done')

"initial parameters"
if model_funct == kelvin:
    A = 65
    B = 43
    initial = np.array([A, B])
    bounds = ([0, 0], [1000, 1000])
if model_funct == linear_S:
    A = 88.0036163282118
    B = 11
    C = 2385.010449469219
    initial = np.array([A, B, C])
    bounds = ([0, 0, 0], [500, 500, 10000])
if model_funct == linear_S_decay:
    A = 54
    B = 26
    C = 3881
    D = 0.0002
    E = 2
    F = 5
    initial = np.array([A, B, C, D, E, F])
    bounds = ([0, 0, 0, 0, 0, 0], [500, 500, 10000, 100, 1000, 1000])
if model_funct == linear_kelvin:
    A = 65
    B = 43
    C = 5300
    D = 400
    E = 2000
    initial = np.array([A, B, C, D, E])
    bounds = ([0, 0, 0, 0, 0], [500, 500, 10000, 10000, 10000])
if model_funct == Burger:
    A = 300
    B = 150
    C = 5000
    D = 900000
    initial = np.array([A, B, C, D])
    bounds = ([0, 0, 0, 0], [1000, 1000, 10000, 1000000])

a = 0
b = 4
c = 0
d = 10
param = np.zeros([len(Temp), len(sensor_number), len(initial)])
param_err = np.zeros([len(Temp), len(sensor_number), len(initial)])
red_chi_sq = np.zeros([len(Temp), len(sensor_number)])
for k, T in enumerate(Temp[a:b + 1]):
    k = k + a
    for j, n in enumerate(sensor_number[c:d + 1]):
        j = j + c
        sensor, i, temp = find_sensor(files[k], n)
        file = (dir + "\\Data\\" + str(T) + " Temp\\" + sensor)
        #print('Num = ' + str(n) + ', ' + 'Temp = ' + str(temp))
        S, E = int(indexes[k][j][0]), int(indexes[k][j][1])
        t = get_t(file)
        R = get_R(file)
        t = np.array(t[S:E])
        R = np.array(R[S:E])
        t_off = t - t[0]
        t = t_off
        R = 1 / R
        R_err = 0.00001 * np.ones(len(t))

        popt, cov = scipy.optimize.curve_fit(model_funct,  # function to fit
                                             t,  # x data
                                             R,  # y data
                                             sigma=R_err,  # set yerr as the array of error bars for the fit
                                             absolute_sigma=True,  # errors bars DO represent 1 std error
                                             p0=initial,  # starting point for fit
                                             check_finite=True,
                                             maxfev=5000,  # maximum number of iterations
                                             bounds=bounds)  # provides bounts to the optimised parameters, use if 'raise ValueError if NaN' encountered

        popt_errs = np.sqrt(np.diag(cov))
        deg_freedom = t.size - initial.size
        chisq_min = chisq(popt, model_funct, t, R, R_err)
        chisq_reduced = chisq_min / deg_freedom

        param[k][j] = popt
        param_err[k][j] = popt_errs
        red_chi_sq[k][j] = chisq_reduced

print('optimisation done')


P_mean = np.zeros([len(Temp), len(initial)])
P_err = np.zeros([len(Temp), len(initial)])

"calc mean optimised parameters"
for i in range(param.shape[2]):
    for j in range(param.shape[1]):
        mean = []
        std = []
        for k in range(param.shape[0]):
            m = np.mean(param[k, :, i])
            s = np.std(param[k, :, i])
            mean.append(m)
            std.append(s)
        P_mean[:, i] = mean
        P_err[:, i] = std
var = P_err / P_mean
print(np.mean(var))

print(P_mean)
print(P_err)
data = np.zeros([len(Temp), len(initial), 2])
data[:, :, 0] = P_mean
data[:, :, 1] = P_err

"plot mean optimised parameters"
plt.figure(figsize=[10, 6])
colours_2 = ['b', 'y', 'g', 'r', 'm', 'k']
if model_funct == kelvin:
    labels = ['k', chr(951)]
if model_funct == linear_S_decay:
    labels = ['k\u2080', 'k\u2081', chr(951), 'A', 'B', 'C']
if model_funct == linear_kelvin:
    labels = ['k\u2080', 'k\u2081',  chr(951) + '\u2080', 'k\u2082', chr(951) + '\u2081']
if model_funct == Burger:
    labels = ['k\u2080', 'k\u2081',  chr(951) + '\u2081', chr(951) + '\u2080']
if model_funct == linear_S:
    labels = ['k\u2080', 'k\u2081', chr(951)]
print(model_funct.__name__)
for i in range(param.shape[2]):
    T_mean = np.zeros([len(Temp)])
    T_err = np.zeros([len(Temp)])
    p_mean = np.zeros([len(Temp)])
    p_err = np.zeros([len(Temp)])
    for k in range(param.shape[0]):
        Ts = []
        for j in range(param.shape[1]):
            temp = find_sensor(files[k], sensor_number[j])[2]
            Ts.append(int(temp))
        T_mean[k] = np.mean(Ts)
        T_err[k] = np.std(Ts)
        if 0 <= T_err[k] < 1:
            T_err[k] = 1
        p_mean[k] = np.mean(param[k, :, i])
        std_1 = np.std(param[k, :, i])
        std_2 = np.mean(param_err[k, :, i])
        #if std_1 > std_2:
        p_err[k] = std_1
        #else:
         #   p_err[k] = std_2
    plt.errorbar(T_mean, p_mean, xerr=T_err, yerr=p_err, fmt='o', color=colours_2[i])
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Parameter value ' + '(' + labels[i] + ')')
    plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
    plt.tick_params(bottom=True, top=False, left=True, right=False)
    plt.savefig(dir + '\\Graphs\\' + str(model_funct.__name__) + "-" + str(labels[i]) + '.png', dpi=1000, bbox_inches='tight')
    plt.show()


"""
columns = []
index = []
for l in labels:
    index.append('Parameter' + str(l))
for T in Temp:
    columns.append(T)
table = pd.DataFrame(np.transpose(P_mean), index=index, columns=columns)
table.to_csv(dir + '\\Graphs\\' + 'Averaged parameters' + str(model_funct.__name__) + '.csv')
print('tabulating done')
"""

"""
"show 1/R v t over range of temp for one sensor using mean op_param"
a = 0
b = len(Temp)
c = 0
d = 10
colour = ['b', 'y', 'g', 'r', 'm']
for j, n in enumerate(sensor_number[c:d+1]):
    j = j + c
    plt.title('no' + str(n))
    for k, T in enumerate(Temp[a:b+1]):
        k = k + a
        sensor, i, temp = find_sensor(files[k], n)

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
        xs = np.linspace(t[0], t[-1], 1000)
        ys = model_funct(xs, *P_mean[k][:])
        plt.scatter(t, R, s=0.5, color=colour[k], label = str(temp) + 'C')
        plt.plot(xs, ys, color=colour[k], label = 'Model function')
    plt.show()
"""



"""
"show comp_R v t over range of temp for one sensor"
a = 0
b = 4
c = 0
d = 10
for j, n in enumerate(sensor_number[c:d+1]):
    j = j + c
    plt.figure(figsize=[10, 6])
    plt.title('no' + str(n))
    for k, T in enumerate(Temp[a:b+1]):
        k = k + a
        sensor, i, temp = find_sensor(files[k], n)
        file = (dir + "\\Data\\" + str(T) + " Temp\\" + sensor)
        S, E = indexes[k][j][0], indexes[k][j][1]
        S, E = int(S), int(E)
        t = get_t(file)
        R = get_R(file)
        t = np.array(t[S:E])
        R = np.array(R[S:E])
        t_off = t - t[0]
        t = t_off
        R_err = 0.00001 * np.ones(len(t))

        xs = np.linspace(t[0], t[-1], len(t))
        ys = model_funct(xs, *P_mean[k][:])
        comp = R * ys
        plt.scatter(xs, comp, s=0.5, label='Compensated Resistance')
        #plt.savefig(dir + '\\Graphs\\Compensated Resistance at' + str(T) + '.png')
    #plt.ylim(0.9,1.1)
    plt.xlabel('Time')
    plt.ylabel('1/R')
    plt.legend()
    plt.show()
"""

"""
"show comp_R v t over one temp and all sensors"
a = 0
b = len(Temp)
c = 0
d = len(sensor_number)
cmap = 'inferno_r'
cmin, cmax = 23, 66
N = sum(len(x) for x in files)
N = 1000
cmap = plt.get_cmap(cmap)(np.linspace(0.1, 0.9, N + 1))
#fig = plt.subplots(figsize=[10, 6])
for k, T in enumerate(Temp[a:b + 1]):
    k = k + a
    Norm = plt.Normalize(cmin, cmax)
    for j, n in enumerate(sensor_number[c:d+1]):
        #2, 5, 8,9 12, 29
        #if j == 0 or j == 1 or j ==
        j = j + c
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
        t_off = t - t[0]
        t = t_off
        xs = np.linspace(t[0], t[-1], len(t))
        ys = model_funct(xs, *P_mean[k][:])
        comp = R * ys
        perc= int((temp - cmin)/(cmax - cmin) * N)
        plt.subplot(5,1,(k+1))
        plt.scatter(xs, comp, s=0.5, color=cmap[perc])
        plt.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False, direction='in')
        plt.tick_params(bottom=True, top=False, left=True, right=False)
        #plt.colorbar(cm.ScalarMappable(norm=Norm, cmap='inferno_r'), ax=plt.gca(), label='Temperature (C)')
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
plt.tick_params(bottom=True, top=False, left=True, right=False)
plt.xlabel('Time (s)')
plt.savefig(dir + '\\Graphs\\Compensated Resistance.png', dpi=1000, bbox_inches='tight')
plt.show()
"""