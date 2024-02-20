import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize

dir, file = os.path.split(__file__)
file = (dir + "\\Data\\"
#              "test_23.dat"
        "no6test26official.dat"
        )

#### import all data, clean it and chi-squared fit it 


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
R = clean(R, 10 ** 3)

L = len(T)   # number of data points
S_r = (    # skip over the first S readings
#19  #1
#15  #2
#48  #3
#60  #5
35 #6
)
E_r = (   # end at this reading
#700  #1
#900  #2
#700  #3
#216  #5
325 #6
)
S_t = (
1     #1, 2, 3, 5, 6
)
E_t = E_r - S_r + S_t


T = np.array(T[S_t:E_t])
R = R[S_r:E_r]
R_inv = 1/(np.array(R))
R_err = 0.00001 * np.ones(len(R))      # error in R check
#T_off = T - T[0]


plt.errorbar(T, R_inv, yerr=R_err)
plt.ylim(0,R_inv[-1])
plt.xlim(0,T[-1])
plt.show()

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

model_funct = kelvin

# inital parameters
if model_funct == kelvin:
    a = (
        #0.0077   #1
        #0.010013316627657575  #2
        0.009500850220917489 #3
    )
    b = (
        #40      #1
        #3.1369479144666803    #2
        30.734435398949792    #3
    )

    initial = np.array([a, b])

if model_funct == linear_S:
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

    values = [[666, 667, 3000], [99.56929353055796, -1207.2183402614955, 2835.0578267342435]]
    label = ['change ko', 'change k1', 'change e']
    plt.figure()
    for v in values:
        y = model_funct(T, *v)
        plt.plot(T, y, label=label[values.index(v)])
        #plt.ticklabel_format(axis='y', style='plain')
        plt.show()

    initial = np.array([a, b, c])

###plotting intensity of image with initial best fit and errors

# defining x,y,yerr
xval = T
yval = R_inv
yerr = R_err
assert len(yval) == len(xval)
assert len(yerr) == len(yval)

# visualising the error
n_bins = 20
# plt.hist(yerr, bins=n_bins)
# plt.show

# plotting initial model function against data
plt.figure()
plt.errorbar(xval, yval, yerr=yerr, ms=1, marker='+', alpha=1, linewidths=None, label='data')
plt.plot(xval, model_funct(xval, *initial), label='initial')
plt.xlabel('x (units)')
plt.ylabel('y (units)')
plt.legend()
plt.show()

# degrees of freedom of ??
deg_freedom = xval.size - initial.size  # Make sure you understand why!
print('DoF = {}'.format(deg_freedom))

###optimised parameters (hope)

popt, cov = scipy.optimize.curve_fit(model_funct,  # function to fit
                                     xval,  # x data
                                     yval,  # y data
                                     sigma=yerr,  # set yerr as the array of error bars for the fit
                                     absolute_sigma=True,  # errors bars DO represent 1 std error
                                     p0=initial,  # starting point for fit
                                     check_finite=True)
# bounds=((0,, ),(10e+10,np.inf, ))) # raise ValueError if NaN encountered (don't allow errors to pass)

print('Optimised parameters = ', popt, '\n')
print('Covariance matrix = \n', cov)
print(np.sqrt(np.diag(cov)))
print(cov.shape)

# defining chi^2 function

def chisq(model_params, model_function, x_data, y_data, y_err):
    model_funct = model_function
    chisqval = 0
    for i in range(len(x_data)):
        chisqval += ((y_data[i] - model_funct(x_data[i], *model_params)) / y_err[i]) ** 2
    return chisqval


# value of chi^2 min
chisq_min = chisq(popt, model_funct, xval, yval, yerr)
print('chi^2_min = {}'.format(chisq_min))

# value of chi^2 reduced
chisq_reduced = chisq_min / deg_freedom
print('reduced chi^2 = {}'.format(chisq_reduced))

# ??
P = scipy.stats.chi2.sf(chisq_min, deg_freedom)
print('$P(chi^2_min, DoF)$ = {}'.format(P))

###full plot of data with optimised model function
#plot 1/R
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

#plot R
plt.figure(figsize=[10, 6])
plt.plot(xval, 1/yval)
plt.errorbar(xval, 1/model_funct(xval, *popt), yerr=1.5 * np.ones(len(R)))
plt.show()

### Uncertainties
# errors of the optimised parameters
popt_errs = np.sqrt(np.diag(cov))
print(popt_errs)

for i in range(len(popt)):
    print('optimised parameter[{}] = {} +/- {}'.format(i, popt[i], popt_errs[i]))