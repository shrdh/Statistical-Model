# Importing Libaries

import numpy as np
import tensorflow as tf
from scipy.fft import fft, fftfreq
import os
import matplotlib.cm as cm
import pandas as pd
from utils import preprocess_data

from scipy.stats import poisson
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import stats
import scipy.optimize as opt
import math
from scipy.optimize import minimize


tf.get_logger().setLevel('ERROR')

folder= r'D:\Downloads\sensor_1_ESR_cycle_2'
# Specify the path to the temperature file
temperature_file_path = r"D:\Downloads\temp_history_cycle_sensor_2.txt"

# Read the temperature file into a DataFrame
df = pd.read_csv(temperature_file_path, header=None)

# Get the temperatures from the second column (index 1)
temperatures = df[1].tolist()

# Get a list of all the files in the folder (excluding the PARAMS file)
files = os.listdir(folder)
files = [f for f in files if "PARAMS" not in f]

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

d=tf.constant(0.005185511627906974,dtype='float64')#Literature Value

a=tf.constant(-7.647058823532282e-05,dtype='float64')#Literature
b=tf.constant( 2.8681826470588225,dtype='float64') # Literature
c=tf.constant( -4.6511627906973704e-07,dtype='float64')#Literature Value



v = tf.constant(0.00, dtype = 'float64')    # ~N(0,sigma_v)
w = tf.constant(0.00, dtype = 'float64')    # ~N(0,sigma_w)

P_0 = tf.constant(1e-4, dtype = 'float64')
P = tf.constant(0.5, dtype = 'float64')
alpha= tf.constant(14.8e-3, dtype = 'float64')
I = tf.eye(3,dtype = 'complex128')
    
def getD(a,b,temp):
    D = a * temp+ b + alpha * (P_0 - P_0)
    E = c *temp+ d + w
    return D, E

def H(D, E):
    Ham = tf.complex(D * (tf.math.real(spin3 @ spin3 - 2 / 3 * I)) + E * (tf.math.real(spin1 @ spin1 - spin2 @ spin2)),
                    D * (tf.math.imag(spin3 @ spin3 - 2 / 3 * I)) + E * (tf.math.imag(spin1 @ spin1 - spin2 @ spin2)))
    return Ham


@tf.autograph.experimental.do_not_convert
@tf.function
def getP_k(a, b,temp):
    D, E = getD(a, b,temp)
    Ham = H(D, E)
    eigenvalues = tf.linalg.eigvals(Ham)
    return eigenvalues


@tf.function
def bilorentzian(x, a, b,temp):
    eigenvalues = getP_k(a, b,temp)
    x0 = tf.cast(eigenvalues[1] - eigenvalues[2], tf.float64)
    x01 = tf.cast(eigenvalues[0] - eigenvalues[2], tf.float64)
    x = tf.cast(x, tf.float64)
    amp=tf.cast(    47.95674298146268,tf.float64) # avg
    gamma = tf.cast( 0.004293972996039488, tf.float64) # avg
 
 
    return amp * gamma**2 / ((x - x0)**2 + gamma**2) + amp * gamma**2 / ((x - x01)**2 + gamma**2)
def _get_vals(a, b, temp, start_frequency, end_frequency, N):
    timespace = np.linspace(start_frequency, end_frequency, num=N)
    timespace = tf.cast(timespace, 'float64')
    vals = bilorentzian(timespace, a, b, temp)
    return tf.reshape(vals, [N, 1])


all_data = []
all_temperatures = []
all_roots = []
mt_list, mt_orig_list, valt_list, valt_orig_list = [[] for _ in range(4)]

# # Reading Data and taking everything that can be changed
delimiter = "\t"
variable_names = ["Frequency", "Intensity1", "Intensity2"]   
Frequency = None 




num_files_per_temp = 2 

# Constants
pi = tf.cast(tf.constant(np.pi), 'float64')

I = tf.eye(3, dtype='complex128')
#     m= np.reshape(avg_intensity,[-1,1])
def total_loglike(params):
    Frequency = None 
    # global Frequency 
    a, b = params
    total = 0
    for i in range(0, len(files) - num_files_per_temp + 1, num_files_per_temp):
        ratios = []
        for j in range(num_files_per_temp):
            file = files[i+j]
            temp = temperatures[i+j]  # Get the corresponding temperature for this file
            T = tf.constant(temp, dtype=tf.float64)
            data = pd.read_csv(os.path.join(folder, file), delimiter=delimiter, header=None, names=variable_names)
            ratio = np.divide(data['Intensity2'], data['Intensity1']) # Average of ratios
            ratios.append(ratio) # ratios for both files in the pair

        avg_intensity = np.mean(ratios, axis=0)  # Calculate the average ratios for each pair and store them in avg_intensity

        if Frequency is None:
            Frequency = data['Frequency'].values
            # Assuming Frequency is in Hz
        Frequency_GHz = Frequency / 1e9
        start_frequency = np.min(Frequency)/1e9

        end_frequency = np.max(Frequency)/1e9

        N = Frequency.shape[0]
        dt = np.round((end_frequency - start_frequency) / N, 4)
        r = 1 / dt


        timespace = np.linspace(start_frequency, end_frequency, num=N)
        sim_val = _get_vals(a, b, temp, start_frequency, end_frequency, N)
        noise_sample= avg_intensity[np.where(np.abs(timespace)<2.85)[0]] 
        noise_mean = np.mean(noise_sample)
        avg_intensity = avg_intensity - noise_mean
        avg_intensity = np.max(sim_val)*( avg_intensity)/(np.max(avg_intensity))
        noise_sample= avg_intensity[np.where(np.abs(timespace)<2.85)[0]]
        std_noise=np.std(noise_sample)

        # Calculate m for each temperature
        m = np.reshape(avg_intensity, [-1, 1])


        @tf.autograph.experimental.do_not_convert
        @tf.function
        def _getd_d_loglike(a, b, temp, m):
            val = _get_vals(a, b,temp, start_frequency, end_frequency, N)
            loglike=tf.reduce_sum(1* (tf.reshape(m  - dt * r * val, [N, -1])**2 ))/(2*std_noise**2)
            return (loglike)

        loglike = _getd_d_loglike(a, b, temp, m).numpy()
        total += loglike
    return total
if __name__ == "__main__":
    #initial_guess = [-7.772773696087131e-05, 2.8706500246147373] # cycle 2 parameters
    initial_guess=[-7.647058823532282e-05, 2.8681826470588225] #Literature
    # Bounds for a and b
    bounds = [(-7.8e-05, -7.6e-05), (2.869, 2.872)]

    # Run the optimizer
    result = minimize(total_loglike, initial_guess, method='Powell', bounds=bounds
       )

    # The best a and b values
    best_a, best_b = result.x
    print(f"Best a: {best_a}, Best b: {best_b}")

    # a:-7.723607188481802e-05, Best b: 2.87068615576284
     #-7.723607188481802e-05, Best b: 2.87068615576284
     #Best a: -7.723606797749979e-05, Best b: 2.870692879582925
    
# a=tf.constant(-7.772773696087131e-05,dtype='float64') # (cycle 2)
# b=tf.constant(2.8706500246147373,dtype='float64') # cycle 2

