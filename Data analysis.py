"""import all data, clean it and chi-squared fit it"""

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize

dir, file = os.path.split(__file__)
file = (dir + "\\Data\\"
        "no6test26official.dat"
        )

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

def quant(t, *vals):
    F = 1/(vals[0] * t) ** 2
    return F

"""import data"""
T = get_T(file)
R = get_R(file)

"""clean data"""
L = len(T)    # number of data points (i.e number of time readings)

"specifying the start and end indices"
S_r = (       # take the first resistance reading at this index
#19  #1 (index of resistance to start at in no1test22official.dat file)
#15  #2
#48  #3
#60  #5
35 #6
)
E_r = (       # index of final resistance to end at
#700  #1 (no1test22official.dat file)
#900  #2
#700  #3
#216  #5
325 #6
)

S_t = (                    # ignores the time value starting at 0
1     #1, 2, 3, 5, 6
)
E_t = E_r - S_r + S_t      # makes sure the time and resistance arrays are the same length

"slices the arrays according to the given indices"
T = np.array(T[S_t:E_t])
R = R[S_r:E_r]
R_inv = 1/(np.array(R))
R_err = 0.00001 * np.ones(len(R))      # error in R check
a = 13.42
T_off = T - T[0] + a
T = T_off

values = [0.00063, 0]

"shows plot of cleaned data"
plt.errorbar(T_off, R_inv, yerr=R_err)
plt.plot(T - a, linear(T - a, *values))
plt.ylim(0,R_inv[-1])
plt.xlim(0,T_off[-1])
plt.show()


"""starting chi_sq optimisation"""

model_funct = linear_S     # choose the model function to fit the data

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
        99.56929353055796 #6
    )
    b = (
        #667 #1
        -1207.2183402614955 #6
    )
    c = (
        #3000 #1
        2835.0578267342435 #6

    )

    "trials of initial parameters (loops through the arrays to produce graphs)"
    values = [[666, 667, 3000], [99.56929353055796, -1207.2183402614955, 2835.0578267342435]]
    label = ['change ko', 'change k1', 'change e']
    plt.figure()
    for v in values:
        y = model_funct(T, *v)
        plt.plot(T, y, label=label[values.index(v)])
        #plt.ticklabel_format(axis='y', style='plain')
        plt.show()

    initial = np.array([a, b, c])

"defines x,y and yerr values for optimisation"
xval = T
yval = R_inv
yerr = R_err
assert len(yval) == len(xval)
assert len(yerr) == len(yval)

"plots the initial model function against data"
plt.figure()
plt.errorbar(xval, yval, yerr=yerr, ms=1, marker='+', alpha=1, linewidths=None, label='data')
plt.plot(xval, model_funct(xval, *initial), label='initial')
plt.xlabel('x (units)')
plt.ylabel('y (units)')
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
plt.ylim(0,R_inv[-1])
plt.xlim(0,T[-1])
plt.legend()
plt.show()

"plot of R"
plt.figure(figsize=[10, 6])
plt.plot(xval, 1/yval)
plt.errorbar(xval, 1/model_funct(xval, *popt), yerr=1.5 * np.ones(len(R)))
plt.show()

