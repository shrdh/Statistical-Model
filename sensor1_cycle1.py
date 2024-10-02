# Importing Libaries

import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
from scipy.fft import fft, fftfreq
import os
import matplotlib.cm as cm
import pandas as pd
from utils import preprocess_data
from itertools import groupby
from scipy.stats import poisson
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import stats
import scipy.optimize as opt
import math
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from scipy.optimize import curve_fit

tf.get_logger().setLevel('ERROR')
# folder= r'D:\Downloads\Sensor_1_ESR_cycling-20240213T204739Z-001\Sensor_1_ESR_cycling\sensor1_esr_temp_cycle_1'
folder= r'D:\Downloads\sensor1_esr_temp_cycle_1'

# Get a list of all the files in the folder (excluding the PARAMS file)
files = os.listdir(folder)
files = [f for f in files if "PARAMS" not in f]
with open('temperature_file.txt', 'r') as f:
    temperatures = [int(line.strip().replace('°C', '')) for line in f]

# Defining Parameters
s1 = np.array([[0.0,1.0,0.0],
    [1.0,0.0,1.0],
    [0.0,1.0,0.0]])

s2 = np.array([[0.0,-1.0j,0.0],
    [1.0j,0.0,-1.0j],
    [0.0,1.0j,0.0]])

s3 = np.array([[1.0,0.0,0.0],
    [0.0,0.0,0.0],
    [0.0,0.0,-1.0]])

spin1 = (1.0/np.sqrt(2.0))*s1
spin2 = (1.0/np.sqrt(2.0))*s2
spin3=s3


spin1 = tf.constant(spin1, dtype = 'complex128')
spin2 = tf.constant(spin2, dtype = 'complex128')
spin3 = tf.constant(spin3, dtype = 'complex128')

#a= tf.constant(-7.647058823532282e-05,dtype='float64') # Literature Value
# a=tf.constant(-7.86851953723355e-05,dtype='float64')#  cycle 1 parameters
# b= tf.constant(2.870665858002803,dtype='float64') # cycle 1
# a=tf.constant(-7.772773696087131e-05,dtype='float64') # (cycle 2)
# b=tf.constant(2.8706500246147373,dtype='float64') # cycle 2

# b= tf.constant( 2.87068615576284,dtype='float64') # Grad search cycle 2 
# a=tf.constant(-7.723607188481802e-05, dtype='float64') # Grad search cycle 2 


# b= tf.constant(2.870692879582925,dtype='float64') # Grad search cycle 2 
# a=tf.constant( -7.723606797749979e-05, dtype='float64') # Grad search cycle 2 
a=tf.constant(-7.723606797749979e-05,dtype='float64') # (MLE cycle 1 used)
b=tf.constant(2.870745489088582,dtype='float64') # (MLE cycle 1 used)
# a=tf.constant( -7.723607188481802e-05, dtype='float64') # Grad search cycle 2 (used)
# b=tf.constant(2.87068615576284,dtype='float64') # Grad search cycle 2(used)
# a=tf.constant( -7.723606797749979e-05,   dtype='float64')# cycle 1- ( Grad search cycle1)
# b= tf.constant( 2.870745489088582,dtype='float64') # cycle 1- ( Grad search cycle1)
#c=tf.constant( -4.3478260869566193e-07,dtype='float64') # sensor 1
d=tf.constant(0.005185511627906974,dtype='float64')#Literature Value
c=tf.constant( -4.6511627906973704e-07,dtype='float64')#Literature Value
# a=tf.constant( -7.723606710665643e-05,dtype='float64') # (MLE cycle 1)
# b=tf.constant(2.8707465740453,dtype='float64') # (MLE cycle 1)






v = tf.constant(0.00, dtype = 'float64')    # ~N(0,sigma_v)
w = tf.constant(0.00, dtype = 'float64')    # ~N(0,sigma_w)

P_0 = tf.constant(1e-4, dtype = 'float64')
P = tf.constant(0.18, dtype = 'float64')
alpha= tf.constant(14.52e-3, dtype = 'float64')
I = tf.eye(3,dtype = 'complex128')
    
    
    

    
def getD(T, P):
    D = a* T + b + alpha * (P_0 - P_0)
    E = c * T + d + w

    
    return D, E

def H(D, E):
    Ham = tf.complex(D * (tf.math.real(spin3 @ spin3 - 2 / 3 * I)) + E * (tf.math.real(spin1 @ spin1 - spin2 @ spin2)),
                    D * (tf.math.imag(spin3 @ spin3 - 2 / 3 * I)) + E * (tf.math.imag(spin1 @ spin1 - spin2 @ spin2)))
    return Ham


@tf.autograph.experimental.do_not_convert
@tf.function
def getP_k(T, P):
    D, E = getD(T, P)
    Ham = H(D, E)
    eigenvalues = tf.linalg.eigvals(Ham)
    return eigenvalues



@tf.function
def bilorentzian(x, T, P):
    eigenvalues = getP_k(T, P)
    x0 = tf.cast(eigenvalues[1] - eigenvalues[2], tf.float64)
    x01 = tf.cast(eigenvalues[0] - eigenvalues[2], tf.float64)
    x = tf.cast(x, tf.float64)
    # a = tf.cast(9.17793792e+01, tf.float64)  
    # gamma = tf.cast(4.49567820e-03, tf.float64)  
    a=tf.cast(   47.34202029457381, tf.float64)
    gamma = tf.cast(0.004272331099904648, tf.float64)  
 
 
    return a * gamma**2 / ((x - x0)**2 + gamma**2) + a * gamma**2 / ((x - x01)**2 + gamma**2)

def _get_vals(T, P):
    timespace = np.linspace(start_frequency, end_frequency, num=N)
    timespace = tf.cast(timespace, 'float64')
    vals = bilorentzian(timespace, T, P)
    return tf.reshape(vals, [N, 1])



# Initialize lists to store results
all_data = []
all_temperatures = []
all_roots = []
mt_list, mt_orig_list, valt_list, valt_orig_list = [[] for _ in range(4)]

# Reading Data and taking everything that can be changed
delimiter = "\t"
variable_names = ["Frequency", "Intensity1", "Intensity2"]   
Frequency = None 
num_files_per_temp = 20

temperatures = [25, 25, 30, 35, 40, 45, 50, 45, 40, 35, 30, 25, 20, 15, 10, 10]



#Removing first point    
for i in range(num_files_per_temp, len(files), num_files_per_temp):
    files_group = files[i:i+num_files_per_temp]
    temp = temperatures[i//num_files_per_temp]  # Get the corresponding temperature for this group
    T = tf.constant(temp, dtype=tf.float64)
    ratios = np.array([])

   
    for file in files_group:
        data = pd.read_csv(os.path.join(folder, file), delimiter=delimiter, header=None, names=variable_names)

        ratio = np.divide(data['Intensity2'], data['Intensity1'])
        if ratios.size == 0:
            ratios = np.array([ratio])
        else:
            ratios = np.vstack((ratios, [ratio]))  # Add ratio to the numpy array

    avg_intensity = np.mean(ratios, axis=0)
    if Frequency is None:
        Frequency = data['Frequency'].values
        # Assuming Frequency is in Hz
        Frequency_GHz = Frequency / 1e9
        start_frequency = np.min(Frequency)/1e9

    end_frequency = np.max(Frequency)/1e9

    N = Frequency.shape[0]
    dt = np.round((end_frequency - start_frequency) / N, 4)

    timespace = np.linspace(start_frequency, end_frequency, num=N)
    
    sim_val = _get_vals(T, P)
    noise_sample= avg_intensity[np.where(np.abs(timespace)<2.85)[0]] 
    noise_mean = np.mean(noise_sample)
    avg_intensity = avg_intensity - noise_mean
    avg_intensity = np.max(sim_val)*( avg_intensity)/(np.max(avg_intensity))
    noise_sample= avg_intensity[np.where(np.abs(timespace)<2.85)[0]]
    std_noise=np.std(noise_sample)





    # Constants
    pi = tf.cast(tf.constant(np.pi), 'float64')

    I = tf.eye(3, dtype='complex128')
    r = 1 / dt


    m= np.reshape(avg_intensity,[-1,1])

    @tf.autograph.experimental.do_not_convert
    @tf.function
    def _getd_d_loglike(T):
        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch(T)
            val = _get_vals(T, P)
            loglike=tf.reduce_sum( (tf.reshape(m  - dt * r * val, [N, -1])**2/(2*std_noise**2)+math.log(math.sqrt(2 * math.pi) * std_noise)))
        d_loglike = tf.convert_to_tensor(tape3.gradient(loglike,T))
        return (loglike), tf.squeeze(d_loglike)  # , HinvG,
                                        

    def func(T):
        vv, _ = _getd_d_loglike(tf.constant(T, dtype=tf.float64))
        return vv.numpy().astype(np.float64)

    def funcd(T):
        _, vv = _getd_d_loglike(tf.constant(T, dtype=tf.float64))
        return vv.numpy().astype(np.float64)

    root = opt.minimize(fun=func, x0=T.numpy(), jac=funcd, method='Newton-CG', options={'xtol': 1e-5, 'maxiter': 100, 'disp': False, 'return_all': True})

    all_roots.append(root.x[0])
    print(f'Result for temperature {temp}: {root.x[0]}')



    mt, mt_orig=np.zeros([N,1]), np.zeros([N,1])
    valt= _get_vals(tf.constant(root.x[0]),P)
    valt_orig=  _get_vals(T,P)
    for i in range(N):
            mt[i,:] = tf.random.poisson([1],dt*r*(valt[i,:]))
            mt_orig[i,:] = tf.random.poisson([1],dt*r*(valt_orig[i,:]))

    mt_list.append(mt)
    mt_orig_list.append(mt_orig)
    valt_list.append(valt.numpy())
    valt_orig_list.append(valt_orig.numpy())
    all_data.append(avg_intensity)
    all_temperatures.append(temp)

std_roots = np.std(np.array(all_temperatures) - np.array(all_roots))

rmse = root_mean_squared_error(all_temperatures, all_roots) # 1.8 grid search
# 1.3 cycle 2
# 1.26 cycle 1
# Plot the results

# D vs T plot
D_values_estimated = [getD(T, P)[0] for T in all_roots]




plt.figure(figsize=(6, 4))
plt.plot(all_temperatures, D_values_estimated, 'o')
plt.xlabel('Temperatures (℃)', fontsize=12)
plt.ylabel('D (Ghz)', fontsize=12)

slope, intercept = np.polyfit(all_temperatures, D_values_estimated, 1)

#
x_fit = np.linspace(min(all_temperatures), max(all_temperatures), 100)

y_fit = slope * x_fit + intercept
  # 
plt.plot(x_fit, y_fit, 'r-')
plt.annotate(f'Slope: {slope:.10e}', xy=(0.6, 0.5), xycoords='axes fraction', bbox=dict(boxstyle='round', facecolor='none', edgecolor='black'), fontsize=9, color='red')
plt.tick_params(axis='both', which='major', labelsize=12)
# Save the figure to the D drive downloads folder with a DPI of 750
# plt.savefig('D:\\Downloads\\D_T.png', dpi=750)
plt.tight_layout()
# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\D_T_cycle1.png', dpi=300) 

plt.show()


# a) Average Spectrum


# Create a colormap
cmap = cm.get_cmap('viridis')  # 
# Convert temperatures to Kelvin
all_temperatures_kelvin = [temp + 273.15 for temp in all_temperatures]

# Normalize the temperatures to the range [0, 1] for the colormap
norm = plt.Normalize(min(all_temperatures_kelvin), max(all_temperatures_kelvin))

# Create a new figure and axes
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the data for each group with a color from the colormap
for i, (data, temp) in enumerate(zip(all_data, all_temperatures_kelvin)):
    ax.plot(timespace, data, color=cmap(norm(temp)))

# Add a colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # this line is needed for matplotlib versions >= 3.1
fig.colorbar(sm, ax=ax, label='Temperatures (K)')  # Change label to 'Temperatures (K)'

# Set the x-axis label, y-axis label, and title
ax.set_xlabel('Frequencies (GHz)', fontsize=12)
ax.set_ylabel('Intensity (arb. Units)', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)

ax.invert_yaxis()
plt.tight_layout()

# Uncomment the next line if you want to save the figure
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Paper Figs\\cycle2\\Statistical Model\\avgspectrum_cycle1_using_cycle_2_MLE.png', dpi=300) 

# Show the plot
plt.show()

# Create a colormap
cmap = cm.get_cmap('viridis')  # 'viridis' is one of the predefined colormaps, but there are many others

# Normalize the temperatures to the range [0, 1] for the colormap
norm = plt.Normalize(min(all_temperatures), max(all_temperatures))
plt.figure(figsize=(6, 4))
# Plot the data for each group with a color from the colormap
for i, (data, temp) in enumerate(zip(all_data, all_temperatures)):
    plt.plot(timespace, data, color=cmap(norm(temp)))

# Add a colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
plt.colorbar(sm, label='Temperatures (°C)')

# Set the x-axis label, y-axis label, and title
plt.xlabel('Frequencies (GHz)', fontsize=12)
plt.ylabel('Intensity (arb. units)',fontsize=12)
# plt.title('Average Spectrum for Each Group')
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)  # Set the font size of y values

plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\flipped_average_spectrum_sensor_1_cycle_1.png', dpi=300) 
# Show the plot
plt.show()    

# Quadratic fit

coefficients = np.polyfit(all_temperatures, all_roots, 2)
polynomial = np.poly1d(coefficients)
first_coefficient = coefficients[0]

# Generate x values
x_values = np.linspace(min(all_temperatures), max(all_temperatures), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(all_temperatures)
# Your predicted values
predicted =all_roots

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.figure(figsize=(6, 4))
plt.plot(all_temperatures, all_roots, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('Measured Temperatures (℃)', fontsize=12)
plt.ylabel('Predicted Temperatures (℃)', fontsize=12)
# plt.title('True vs Estimated for Sensor I (GaussLorenz)')
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)  # Set the font size of y values
plt.text(min(temperatures), max(predicted) * 0.7, 'R-squared = {:.3f}\nFirst Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly, first_coefficient, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\quadratic_fit_cycle1.png', dpi=300) 
plt.show()


# Plot the residuals (Quadratic Fit)

coefficients = np.polyfit(all_temperatures, all_roots, 2)
polynomial = np.poly1d(coefficients)
first_coefficient = coefficients[0]

# Generate x values
x_values = np.linspace(min(all_temperatures), max(all_temperatures), 100)

# Generate y values
y_values = polynomial(x_values)

true = all_temperatures

# Your predicted values
predicted = all_roots

# Calculate R-squared
r_squared = r2_score(true, predicted)

# Calculate RMSE
rmse = mean_squared_error(true, predicted, squared=False)

# Calculate residuals
residuals =  predicted - polynomial(true)

# Plot residuals
plt.figure(figsize=(6, 4))
plt.plot(true, residuals, 'o')
plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at y=0
plt.xlabel('Measured Temperatures (℃)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)  # Set the font size of y values
plt.tight_layout()
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\residuals_temperature_quadratic_cycle1.png', dpi=300) 
plt.show()


# Plot the residuals (Linear Fit)

coefficients = np.polyfit(all_temperatures, all_roots, 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(all_temperatures), max(all_temperatures), 100)

# Generate y values
y_values = polynomial(x_values)

true = all_temperatures

# Your predicted values
predicted = all_roots

# Calculate R-squared
r_squared = r2_score(true, predicted)

# Calculate RMSE
rmse = mean_squared_error(true, predicted, squared=False)

# Calculate residuals
residuals =  predicted - polynomial(true)

# Plot residuals
plt.figure(figsize=(6, 4))
plt.plot(true, residuals, 'o')
plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at y=0
plt.xlabel('Measured Temperatures (℃)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)  # Set the font size of y values
plt.tight_layout()
# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\residuals_temperature_linear_cycle1.png', dpi=300) 
plt.show()


# Fitting Curve (true-predicted) (Using cycle 2 parameters)
coefficients = np.polyfit(all_temperatures, all_roots, 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(all_temperatures), max(all_temperatures), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(all_temperatures)
# Your predicted values
predicted =all_roots

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.figure(figsize=(6, 4))
plt.plot(all_temperatures, all_roots, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('Measured Temperatures (℃)', fontsize=12)
plt.ylabel('Predicted Temperatures (℃)', fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)  # Set the font size of y values
plt.text(min(temperatures), max(predicted) * 0.5, 'R-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
# plt.tight_layout()
# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\cycle1_true_estimated_using_cycle2.png', dpi=300) 
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()

# Training 1.10 (Linear)

plt.figure(figsize=(6, 4))

# Fit a linear polynomial
coefficients1 = np.polyfit(all_temperatures, all_roots, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(all_temperatures), max(all_temperatures), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(all_temperatures)
predicted =all_roots
r_squared_poly1 = r2_score(true_poly1, predicted)

# Plot the linear fit in red
plt.plot(all_temperatures, all_roots, 'o', color='green')  # Change color to green
plt.plot(x_values, y_values1, '-', color='red')

# Fit a quadratic polynomial
coefficients2 = np.polyfit(all_temperatures, all_roots, 2)
polynomial2 = np.poly1d(coefficients2)
coefficient = coefficients2[0]
y_values2 = polynomial2(x_values)
true_poly2 = polynomial2(all_temperatures)
predicted =all_roots
r_squared_poly2 = r2_score(true_poly2, predicted)

# Plot the quadratic fit in blue with dashed line
plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xlabel('Measured Temperatures (℃)', fontsize=12)
plt.ylabel('Predicted Temperatures (℃)', fontsize=12)

# Add a legend
plt.legend(['Data', 'Linear Fit', 'Quadratic Fit'])

# Add text boxes for the linear and quadratic fits
plt.text(min(all_temperatures), max(predicted) * 0.5, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.text(max(all_temperatures)*0.489, max(predicted)*0.25, 'Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}'.format(r_squared_poly2, coefficient), color='blue', bbox=dict(facecolor='white', alpha=0.7))

# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\combined_fit_using_cycle_2.png', dpi=300) 

# Show the plot
plt.show()

temperature = 50
indices = [i for i, temp in enumerate(all_temperatures) if temp == temperature]

# Create a figure
plt.figure(figsize=(6, 4))

# Plot the estimated spectra
for i in indices:
    plt.plot(timespace, valt_list[i], label='Estimated Spectra')

# Plot the real spectra
for i in indices:
    plt.plot(timespace, valt_orig_list[i], label='Real Spectra')

# Plot the all_data
for i in indices:
    plt.plot(timespace, all_data[i], label='Data')

# Set the title, x-label, and y-label
plt.xlabel('Frequencies (GHz)', fontsize=12)
plt.ylabel('Intensity of Photons (arb. units)', fontsize=12)

# Add a legend
plt.legend(fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)  # Set the font size of y values
plt.tight_layout()
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\cycle1_temp_50.png', dpi=300) 
# Show the plot
plt.show()

# Bilorenzian Fit

# BiLorentzian

def two_lorentzian( x, x0, x01,a, gam):


    return ( (a * gam**2 / ( gam**2 + ( x - x0 )**2)) + (a * gam**2 / ( gam**2 + ( x - x01 )**2)))
eigenvalues = getP_k(T, P)
x0_value = tf.cast(eigenvalues[1] - eigenvalues[2], tf.float64)
x01_value = tf.cast(eigenvalues[0] - eigenvalues[2], tf.float64)

pobilorentzian=[x0_value,x01_value, 75.00, 0.008]

popt_bilorentzian, pcov_bilorentzian = curve_fit(two_lorentzian, Frequency_GHz, avg_intensity, p0=pobilorentzian)
plt.figure(figsize=(6, 4))
plt.plot(Frequency_GHz, avg_intensity, 'b-', label='Data')

# Calculate the fitted values
fitted_values = two_lorentzian(Frequency_GHz, *popt_bilorentzian)


# Plot the fitted data
plt.plot(Frequency_GHz, fitted_values, 'r--', label='Fitted Data')

# Calculate the residuals
residuals = avg_intensity - fitted_values

# Plot the residuals
plt.plot(Frequency_GHz, residuals, 'k-', label='Residuals')

# Set the x-axis label, y-axis label, and title
plt.xlabel('Frequency (GHz)',fontsize=12)
plt.ylabel('Intensity (arb. units)', fontsize=12)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)  # Set the font size of y values
plt.tight_layout()
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\cycle1_data_fit.png', dpi=300) 



# Show the plot
plt.show()






plt.figure(figsize=(6, 4))

# Convert temperatures to Kelvin
all_temperatures_kelvin = [temp + 273.15 for temp in all_temperatures]
predicted = [root + 273.15 for root in all_roots] 

# Fit a linear polynomial
coefficients1 = np.polyfit(all_temperatures_kelvin, predicted, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(all_temperatures_kelvin), max(all_temperatures_kelvin), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(all_temperatures_kelvin)
r_squared_poly1 = r2_score(true_poly1, predicted)

# Plot the linear fit in red
plt.plot(all_temperatures_kelvin, predicted, 'o', color='green')  # Change color to green
plt.plot(x_values, y_values1, '-', color='red')

# Fit a quadratic polynomial
coefficients2 = np.polyfit(all_temperatures_kelvin, predicted, 2)
polynomial2 = np.poly1d(coefficients2)
coefficient = coefficients2[0]
y_values2 = polynomial2(x_values)
true_poly2 = polynomial2(all_temperatures_kelvin)
r_squared_poly2 = r2_score(true_poly2, predicted)

# Plot the quadratic fit in blue with dashed line
plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xlabel('Measured Temperatures (K)', fontsize=12)  # Change label to Kelvin
plt.ylabel('Predicted Temperatures (K)', fontsize=12)  # Change label to Kelvin

# Add a legend
plt.legend(['Data', 'Linear Fit', 'Quadratic Fit'])

# Add text boxes for the linear and quadratic fits
plt.text(min(all_temperatures_kelvin), max(predicted) * 0.9, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.text(min(all_temperatures_kelvin), max(predicted) * 0.8, 'Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}'.format(r_squared_poly2, coefficient), color='blue', bbox=dict(facecolor='white', alpha=0.7))

# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\combined_fit_using_cycle_2.png', dpi=300) 

# Show the plot
plt.show()


plt.figure(figsize=(6, 4))

# Convert temperatures to Kelvin
all_temperatures_kelvin = [temp + 273.15 for temp in all_temperatures]
predicted = [root + 273.15 for root in all_roots] 

# Fit a linear polynomial
coefficients1 = np.polyfit(all_temperatures_kelvin, predicted, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(all_temperatures_kelvin), max(all_temperatures_kelvin), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(all_temperatures_kelvin)
r_squared_poly1 = r2_score(true_poly1, predicted)

# Plot the linear fit in red
plt.plot(all_temperatures_kelvin, predicted, 'o', color='green')  # Change color to green
plt.plot(x_values, y_values1, '-', color='red')

# Fit a quadratic polynomial
coefficients2 = np.polyfit(all_temperatures_kelvin, predicted, 2)
polynomial2 = np.poly1d(coefficients2)
coefficient = coefficients2[0]
y_values2 = polynomial2(x_values)
true_poly2 = polynomial2(all_temperatures_kelvin)
r_squared_poly2 = r2_score(true_poly2, predicted)

# Plot the quadratic fit in blue with dashed line
plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xlabel('Measured Temperatures (K)', fontsize=12)  # Change label to Kelvin
plt.ylabel('Predicted Temperatures (K)', fontsize=12)  # Change label to Kelvin

# Add a legend
plt.legend(['Data', 'Linear Fit', 'Quadratic Fit'], loc='best')

# Add text boxes for the linear and quadratic fits
# plt.text(min(all_temperatures_kelvin) , max(predicted) * 0.8 , 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
# plt.text(min(all_temperatures_kelvin) , max(predicted) * 0.93, 'Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}'.format(r_squared_poly2, coefficient), color='blue', bbox=dict(facecolor='white', alpha=0.7))
# Add text boxes for the linear and quadratic fits
plt.annotate('Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), 
             xy=(0.02, 0.6), xycoords='axes fraction', color='red', 
             bbox=dict(facecolor='white', alpha=0.7))

plt.annotate('Quadratic Fit:\nR-squared = {:.3f}\n1st Coefficient = {:.2e}'.format(r_squared_poly2, coefficient), 
             xy=(0.6, 0.05), xycoords='axes fraction', color='blue', 
             bbox=dict(facecolor='white', alpha=0.7))
# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
#plt.savefig('D:\\Downloads\\ODMR Paper Figs\\combined_fit_using_cycle_2_kelvin.png', dpi=300) 

# Show the plot
plt.show()


from sklearn.metrics import mean_squared_error

# Calculate RMSE for the linear fit
rmse_poly1 = mean_squared_error(true_poly1, predicted, squared=False)

# Calculate RMSE for the quadratic fit
rmse_poly2 = mean_squared_error(true_poly2, predicted, squared=False)

print(f'RMSE for Linear Fit: {rmse_poly1}')
print(f'RMSE for Quadratic Fit: {rmse_poly2}')


from sklearn.metrics import mean_squared_error

# Calculate RMSE for the linear fit
rmse_poly1 = mean_squared_error(true_poly1, predicted, squared=False)

# Calculate RMSE for the quadratic fit
rmse_poly2 = mean_squared_error(true_poly2, predicted, squared=False)

# RMSE for Linear Fit: 1.0987155468780232
# RMSE for Quadratic Fit: 1.0261098765219119


# Convert all_roots from Celsius to Kelvin
all_roots_kelvin = [temp + 273.15 for temp in all_roots]
all_temperatures_kelvin = [temp + 273.15 for temp in all_temperatures]
# Residuals Combined Plot

# Fit a linear polynomial
coefficients1 = np.polyfit(all_temperatures_kelvin, all_roots_kelvin, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
residuals1 = all_roots_kelvin - polynomial1(all_temperatures_kelvin)

# Fit a quadratic polynomial
coefficients2 = np.polyfit(all_temperatures_kelvin, all_roots_kelvin, 2)
polynomial2 = np.poly1d(coefficients2)
first_coefficient = coefficients2[0]
residuals2 = all_roots_kelvin - polynomial2(all_temperatures_kelvin)

# Plot residuals
plt.figure(figsize=(6, 4))
plt.plot(all_temperatures_kelvin, residuals1, 'o', color='red')  # Plot residuals of linear fit in red
plt.plot(all_temperatures_kelvin, residuals2, 'o', color='blue')  # Plot residuals of quadratic fit in blue
plt.axhline(0, color='black', linestyle='--')  # Add a horizontal line at y=0
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)  # Change label to Kelvin
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)  # Set the font size of y values
plt.legend(['Linear Fit', 'Quadratic Fit'])  # Add a legend
plt.tight_layout()
#plt.savefig('D:\\Downloads\\ODMR Paper Figs\\residuals_temperature_kelvin_combined_cycle1.png', dpi=300) 
plt.show()


# Convert all_temperatures from Celsius to Kelvin
all_temperatures_kelvin = [temp + 273.15 for temp in all_temperatures]

# D vs T plot
D_values_cycle2 = [getD(T, P)[0] for T in all_roots]
plt.figure(figsize=(6, 4))  # 

plt.plot(all_temperatures_kelvin, D_values_cycle2, 'o', color='green')
plt.xlabel('Temperatures (K)', fontsize=12)  # Change label to Kelvin
plt.ylabel('D (Ghz)', fontsize=12)

slope,intercept = np.polyfit(all_temperatures_kelvin, D_values_cycle2, 1)

x_fit = np.linspace(min(all_temperatures_kelvin), max(all_temperatures_kelvin), 100)

y_fit = slope * x_fit + intercept

plt.plot(x_fit, y_fit, 'r-')
plt.annotate(f'Slope: {slope:.10e}', xy=(0.6, 0.5), xycoords='axes fraction', bbox=dict(boxstyle='round', facecolor='none', edgecolor='black'), fontsize=9, color='red')
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)  # Set the font size of y values

plt.tight_layout()
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\D_T_cycle1_kelvin', dpi=300) 
plt.show()