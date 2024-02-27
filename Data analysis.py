"""import all data, clean it and chi-squared fit it"""

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize

dir, file = os.path.split(__file__)

''' the numbers of the sensor and repeat (usually one) writen in format 'sensor no.''repeat no.'
    written in format so comp can read'''

N = 2
R = 1
T = 25

sensor_number = [
            'no1', 'no2', 'no3', 'no5', 'no6', 'no8', 'no9', 'no12', 'no14', 'no19',
            'no29'
]

Temp = [25, 35, 45, 55]

files_25 = [
        "no1test23official.lvm", "no2test23official_1.lvm", "no3test23official_1.dat", "no5test23official_2.dat",
        "no5test25official.dat", "no6_1test26official.dat", "no8_1test23official.dat", "no9_1test25official.dat",
        "no12_1test27official.dat", "no14_1test23official.dat", "no19_1test24official.dat", "no26_1test24official.dat"
]

reference =           [1,   2,   3,  3,  5,  6,  8, 9, 12, 14, 19, 24]  # list of the reference number of the sensor
Start_resistance_RT = [210, 210, 15, 48, 60, 35, 41, 86, 19, 20, 12, 25]              # list of the index to start resistance data
End_resistance_RT =   [1000, 1002, 900, 700, 216, 325, 843, 850, 814, 862, 820, 820]  # list of the index to end resistance data



"""define functions"""
def find_sensor(files, N, R):
    # finds the number of the sensor from the file name
    number = 'no' + str(N) + '_' + str(R)

    index = 0
    for i, f in enumerate(files):
        if number in f:
            index = i
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
        if -s < d < s:
            cleaned.append(d)
        if d > s or d < - s:
            cleaned.append(0)
    return cleaned


def find_value(file, value):
    # if you want to find a specific value in the file
    positions = []
    for i, d in enumerate(file):
        if float('%.2g' % d) == value:
            positions.append(i)
    return positions[0]


def chisq(model_params, model_function, x_data, y_data, y_err):
    # chi-squared function
    model_funct = model_function
    chisqval = 0
    for i in range(len(x_data)):
        chisqval += ((y_data[i] - model_funct(x_data[i], *model_params)) / y_err[i]) ** 2
    return chisqval


def linear(t, *vals):
    F = vals[0] * t + vals[1]
    return F


def exp(t, *vals):
    F = vals[0] * np.exp(-vals[1] * t)
    return F


def kelvin(t, *vals):
    F = vals[0]*(1-np.exp(-t/vals[1]))
    return F


def linear_S(t, *vals):
    F = 1/(vals[0] + vals[1]) + ( 1/vals[0] - 1/(vals[0] + vals[1]) ) * ( 1 - np.exp(- t * (vals[0] * vals[1])/(vals[2] * (vals[0] + vals[1]))) )
    return F

def Burger(t, *vals):
    F = 1/vals[0] + 1/vals[1] * (1 - np.exp(- (t - vals[1]/vals[2]))) + t / vals[3]
    return F


def quant(t, *vals):
    F = 1/(vals[0] * t) ** 2
    return F

"""import data"""
sensor, i = find_sensor(files_RT, N, R)
file = (dir + "\\Data\\" + Temp[i] + " Temp\\" + sensor)
print(file)
Start_resistance = Start_resistance_RT
End_resistance = End_resistance_RT

T = get_T(file)
R = get_R(file)


"""clean data"""
"specifying the start and end indices"
S_R = Start_resistance[i]  # start index of resistance data
E_R = End_resistance[i]    # end index of resistance data

S_T = 1
E_T = E_R - S_R + S_T

"slices the arrays according to the given indices"
T = np.array(T[S_T:E_T])
R = R[S_R:E_R]
R_inv = 1/(np.array(R))
R_err = 0.0001 * np.ones(len(R))      # error in R check
a = 13.42
T_off = T - T[0] + a

values = [0.00063, 0]

"show R v T plot of cleaned data"
plt.plot(T, R)
plt.show()

"""starting chi_sq optimisation"""

model_funct =  linear_S    # choose the model function to fit the data

"initial parameters"

if model_funct == kelvin:           # if the model function is the Kelvin-Voigt model
    a = (
        #0.0077   #1
        #0.010013316627657575 #2
        0.009500850220917489 #3
    )
    b = (
        #40      #1
        #3.1369479144666803    #2
        30.734435398949792    #3
    )

    initial = np.array([a, b])

if model_funct == linear_S:        # if the model function is the linear solid model
    a = (
        #666 #1
        #99.56929353055796 #6
        88.0036163282118 # 26
    )
    b = (
        #667 #1
        #-1207.2183402614955 #6
        - 491297389.5000274 #26
    )
    c = (
        #3000 #1
        #2835.0578267342435 #6
        2385.010449469219 #26

    )

    initial = np.array([a, b, c])

if model_funct == Burger:        # if the model function is the Burger model
    a = (
        0.0008 #1
    )

    b = (
        -5000  #1
    )
    c = (
        3000 #1
    )

    d = (
        10000 #1
    )

    initial = np.array([a, b, c, d])

"defines x,y and yerr values for optimisation"
xval = T
yval = R_inv
yerr = R_err
assert len(yval) == len(xval)
assert len(yerr) == len(yval)

plt.figure(figsize=[10, 6])
plt.errorbar(xval, yval, yerr=yerr, marker='o', ms=1, linestyle='None', label='Data')
plt.plot(xval,
            model_funct(xval, *initial),
            'r', label='Initial')
plt.xlabel('Time')
plt.ylabel('1/R')
#plt.ylim(0,R_inv[-1])
#plt.xlim(0,T[-1])
plt.legend()
plt.show()


"degrees of freedom"
deg_freedom = xval.size - initial.size  # Make sure you understand why!
print('DoF = {}'.format(deg_freedom))

"code for the optimisation of parameters"
popt, cov = scipy.optimize.curve_fit(model_funct,  # function to fit
                                     xval,  # x data
                                     yval,  # y data
                                     sigma=yerr,  # set yerr as the array of error bars for the fit
                                     absolute_sigma=True,  # errors bars DO represent 1 std error
                                     p0=initial,  # starting point for fit
                                     check_finite=True)
# bounds=((0,, ),(10e+10,np.inf, ))) # provides bounts to the optimised parameters, use if 'raise ValueError if NaN' encountered

"prints the optimised parameters and errors"
print('Optimised parameters = ', popt, '\n')
print('Covariance matrix = \n', cov)
print(np.sqrt(np.diag(cov)))
print(cov.shape)

"value of chi^2 min"
chisq_min = chisq(popt, model_funct, xval, yval, yerr)
print('chi^2_min = {}'.format(chisq_min))

"value of chi^2 reduced"
chisq_reduced = chisq_min / deg_freedom
print('reduced chi^2 = {}'.format(chisq_reduced))

"??"
P = scipy.stats.chi2.sf(chisq_min, deg_freedom)
print('$P(chi^2_min, DoF)$ = {}'.format(P))

"errors of the optimised parameters"
popt_errs = np.sqrt(np.diag(cov))
print(popt_errs)
for i in range(len(popt)):
    print('optimised parameter[{}] = {} +/- {}'.format(i, popt[i], popt_errs[i]))

"""plots the optimised model function against data"""
"plot of 1/R"
plt.figure(figsize=[10, 6])
plt.errorbar(xval, yval, yerr=yerr, marker='o', ms=1, linestyle='None', label='Data')
plt.plot(xval,
            model_funct(xval, *popt),
            'r', label='Model function')
plt.xlabel('Time')
plt.ylabel('1/R')
#plt.ylim(0,R_inv[-1])
#plt.xlim(0,T[-1])
plt.legend()
plt.show()

"plot of R"
plt.figure(figsize=[10, 6])
plt.plot(xval, 1/yval)
plt.errorbar(xval, 1/model_funct(xval, *popt), yerr=0.01 * np.ones(len(R)))
plt.show() 

