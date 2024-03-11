import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import norm
plt.rcParams["font.family"]="serif"
plt.rcParams["mathtext.fontset"]="dejavuserif"


dir, file = os.path.split(__file__)

''' the numbers of the sensor and repeat (usually one) writen in format 'sensor no.''repeat no.'
    written in format so comp can read'''

N = 1
Tp = 25
R_threshold = 200
U = R_threshold
#R_err= 25 * 10 ** (-3)


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
        T = file[i+4:i+6]

    return file, index, T


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

    return largest_section_start , largest_section_end - 20 if largest_section_end is not None else None


def Norm_R(y, y_mdl, yerr):
    N_R = (y - y_mdl) / yerr
    return N_R

def min_chi(model_params, model, x_data, y_data, y_err):
    chi_sq = np.sum(((y_data - model(x_data, *model_params)) / y_err) ** 2)
    return chi_sq


def kelvin(t, *vals):
    F = 1/vals[0]*(1-np.exp(-vals[0]/vals[1] * t))
    return F

def linear_S(t, *vals):
    F = 1/(vals[0] + vals[1]) + ( 1/vals[0] - 1/(vals[0] + vals[1]) ) * ( 1 - np.exp(- t * (vals[0] * vals[1])/(vals[2]
        * (vals[0] + vals[1]))) )
    return F

def linear_S_decay(t, *vals):
    F = linear_S(t, *vals[:3]) - vals[3] * np.exp( - vals[4] * (t - vals[5]))
    return F

def linear_kelvin(t, *vals):
    F = linear_S(t, *vals[:3]) + kelvin(t, *vals[3:])
    return F

def Burger(t, *vals):
    F = 1/(vals[0]) + (1/vals[1]) * (1 - np.exp(- (vals[1] * t)/vals[2])) + t/vals[3]
    return F


model_functs = [kelvin, linear_S, linear_S_decay, linear_kelvin, Burger]
model_funct = linear_S_decay # choose the model function to fit the data

"defining sensors + temps to iterate over"
a = 25
b = 65
x = 0 # avoided temps
c = 29
d = 29
y = 0 # avoided sensors

a = Temp.index(a)
b = Temp.index(b) + 1
c = sensor_number.index(c)
d = sensor_number.index(d) + 1
if a >= 0 and b <= len(Temp) and c >= 0 and d <= len(sensor_number):
    print('good')
else:
    print('bad')
    exit()

print(Temp[a:b])
print(sensor_number[c:d])
files = files[a:b]
Temp = Temp[a:b]
sensor_number = sensor_number[c:d]
lim_T = b - a
lim_S = d - c

"""clean data"""
indexes = np.zeros([lim_T, lim_S, 2])
for k, T in enumerate(Temp):
    for j, n in enumerate(sensor_number):
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
    B = 3000
    initial = np.array([A, B])
    bounds = ([0, 0], [1000, 10000])
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
    bounds = ([0, 0, 0, 0, 0, 0], [500, 500, 10000, 10, 100, 100])
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

L = len(initial)
param = np.zeros([lim_T, lim_S,  L])
param_err = np.zeros([lim_T, lim_S, L])
red_chisq = np.zeros([lim_T, lim_S])
for k, T in enumerate(Temp):
    for j, n in enumerate(sensor_number):
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
        R = 1/R
        R_err = 100 * 10 ** (-6) * np.ones(len(t))

        popt, cov = scipy.optimize.curve_fit(model_funct,  # function to fit
                                             t,  # x data
                                             R,  # y data
                                             sigma=R_err,  # set yerr as the array of error bars for the fit
                                             absolute_sigma=True,  # errors bars DO represent 1 std error
                                             p0=initial,  # starting point for fit
                                             check_finite=True,
                                             maxfev=5000,  # maximum number of iterations
                                             bounds=bounds )   # provides bounts to the optimised parameters, use if 'raise ValueError if NaN' encountered

        popt_errs = np.sqrt(np.diag(cov))
        deg_freedom = t.size - initial.size
        chisq_min = min_chi(popt, model_funct, t, R, R_err)
        chisq_reduced = chisq_min / deg_freedom

        param[k][j] = popt
        param_err[k][j] = popt_errs
        red_chisq[k][j] = chisq_reduced

print('optimisation done')

mean_red_chisq = np.zeros([lim_S + 1])
err_red_chisq = np.zeros([lim_S + 1])

"show 1/R v t over range of temp plot"
cmaps = ['Greys', 'Blues', 'Greys', 'viridis_r', 'Greys', 'Reds', 'Purples', 'plasma_r', 'Greys', 'Greys', 'Greens']
for j, n in enumerate(sensor_number):
    print('sensor no.' + str(n))
    fig = plt.figure(j, figsize=[12,6])

    ax1 = fig.add_axes((0.07, 0.295, 0.968, 0.65))
    ax1.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False, direction='in')
    ax1.tick_params(bottom=True, top=False, left=True, right=False)

    ax2 = fig.add_axes((0.07, 0.1, 0.775, 0.17))
    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
    ax2.tick_params(bottom=True, top=True, left=True, right=False)

    ax3 = fig.add_axes((0.857, 0.1, 0.135, 0.17))
    ax3.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False, direction='in')
    ax3.tick_params(bottom=True, top=True, left=True, right=False)

    cmin, cmax = 25, 65
    cmap = plt.get_cmap(cmaps[j+c])(np.linspace(0.25, 0.8, len(Temp)))
    Norm = plt.Normalize(cmin, cmax)
    #for k in range(param.shape[0]):
     #   Ts = []
      #  for j in range(param.shape[1]):
       #     temp = find_sensor(files[k], sensor_number[j])[2]
        #    Ts.append(int(temp))
    for k, T in enumerate(Temp):
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
        R = 1/R
        R_err = 0.0001 * np.ones(len(t))

        xs = np.linspace(t[0], t[-1], 1000)
        ys = model_funct(xs, *param[k][j])
        Res = Norm_R(R, model_funct(t, *param[k][j]), R_err)
        mean = np.mean(Res)
        std = np.std(Res)
        x_hist = np.arange(min(Res), max(Res), 0.001)
        gauss = norm.pdf(x_hist, mean, std)

        ax1.plot(xs, ys, color=cmap[k], linestyle='dashed')
        ax1.scatter(t, R, s=0.5, label=str(T) + 'C', color=cmap[k])
        ax2.plot(t, Res, color=cmap[k])
        ax3.hist(Res, bins=50, orientation='horizontal', density=True, rwidth=0.7, color=cmap[k], alpha=0.7)
        ax3.plot(norm.pdf(x_hist, 0, std), x_hist, color=cmap[k], linestyle='dashed', alpha=0.5)

    ax2.axhline(y=0, linestyle='dashed', color='black', linewidth='1')
    ax2.axhline(y=5, color='darkgrey', linewidth='1')
    ax2.axhline(y=-5, color='darkgrey', linewidth='1')
    ax2.axhspan(-5, 5, color='lightgrey', alpha=0.3)

    ax3.axhline(y=0, linestyle='dashed', color='black', linewidth='1')
    ax3.axhline(y=5, color='darkgrey', linewidth='1')
    ax3.axhline(y=-5, color='darkgrey', linewidth='1')
    ax3.axhspan(5, 5, color='lightgrey', alpha=0.3)

    ax1.set_ylabel(r' $\dfrac{1}{R}$ $(Ω^{-1})$')
    #ax1.legend(loc='upper left', markerscale=5)
    plt.colorbar(cm.ScalarMappable(norm=Norm, cmap=cmaps[j+c]), ax=ax1, orientation='vertical', label='Temperature (°C)')

    ax2.set_ylim([-10, 10])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel('Normalised \n Residuals')

    ax3.set_ylim([-10, 10])
    ax3.set_xlim([0, 2])
    ax3.set_xlabel('Occurences')

    #plt.savefig(dir + '\\Graphs\\' + str(n) + str(model_funct.__name__) + '.png')
    plt.show()
    #plt.figure(figsize=[6, 6])
    #plt.title('Sensor no.' + str(n))
    #plt.scatter(Temp, red_chisq[:, j])
    #plt.show()

    #mean = np.mean(red_chisq[:, j])     # average red_chi_sq across all Temps
    #err = np.std(red_chisq[:, j])       # standard deviation of red_chi_sq across all Temps
    #mean_red_chisq[j] = mean
    #err_red_chisq[j] = err
print('graphing done')


"""
"show comp_R v t over range of temp plot"
a = 0
b = 5
c = 0
d = 10
for j, n in enumerate(sensor_number[c:d+1]):
    j = j + c
    plt.figure(figsize=[10, 6])
    plt.title('no' + str(n))
    for k, T in enumerate(Temp[a:b+1]):
        k = k + a
        sensor, i, temp = find_sensor(files[k], n)
        print('Num = ' + str(n) + ', ' + 'Temp = ' + str(temp))
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
        ys = model_funct(xs, *param[k][j])
        comp = R * ys
        plt.plot(xs, comp, label='Model function')
    plt.ylim(0.9,1.1)
    plt.xlabel('Time')
    plt.ylabel('1/R')
    plt.legend()
    plt.show()
"""

"""
mean_red_chisq[lim_S] = np.mean(mean_red_chisq[:lim_S])   # average red_chi_sq across all sensors and temps
err_red_chisq[lim_S] = np.mean(err_red_chisq[:lim_S])     # average red_chi_sq error all sensors and temps
data = np.zeros([lim_S + 1, 2])
data[:, 0] = mean_red_chisq
data[:, 1] = err_red_chisq


columns = []
index = []
for n in sensor_number:
    columns.append('Sensor no.' + str(n))
    index.append(n)
index.append('Sensor_av')
table = pd.DataFrame(data, index=index, columns=['Mean reduced Chi_sq', 'Error in reduced Chi_sq'])
table.to_csv(dir + '\\Graphs\\' + 'Reduced Chi_sq-' + str(model_funct.__name__) + '@' + str(Temp[-1]) + 'C.csv')
print('tabulating done')
"""