import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from matplotlib.ticker import MultipleLocator

def read_1D_info(filename, col_x, col_y):
    x_array = []
    y_array = []

    infile = open(filename, "r")

    for val in infile:
        read = not any(char == "#" for char in val)
        if read:
            x_array.append(float(val.split()[col_x]))
            y_array.append(float(val.split()[col_y]))

    infile.close()

    x_array = np.array(x_array)
    y_array = np.array(y_array)

    return x_array, y_array

def get_fourier_transform(data, time_step):
    n = len(data)

    # Calculate the frequencies corresponding to the Fourier transform
    freq = np.fft.fftfreq(10*n, d=time_step)

    # Compute the Fourier transform of the data
    ft_data = np.fft.fft(data, n=10*n)

    # Return the frequencies and the Fourier-transformed data
    return freq, ft_data

########################################

filename = "01_1mol/output_detector_0000001/point.dat"

time, ExL_t = read_1D_info(filename, 0, 1)

filename = "01_1mol/output_detector_0000002/point.dat"

time, HyL_t = read_1D_info(filename, 0, 1)


dt = time[2]-time[1]

freq, ExL_w = get_fourier_transform(ExL_t, dt)
freq, HyL_w = get_fourier_transform(HyL_t, dt)


filename = "00_vacumm/output_detector_0000001/point.dat"

time, E0xL_t = read_1D_info(filename, 0, 1)

filename = "00_vacumm/output_detector_0000002/point.dat"

time, H0yL_t = read_1D_info(filename, 0, 1)

dt = time[2]-time[1]

freq, E0xL_w = get_fourier_transform(E0xL_t, dt)
freq, H0yL_w = get_fourier_transform(H0yL_t, dt)


#Pw = np.conj(ExR_w) * HyR_w  - np.conj(ExL_w) * HyL_w

Pw = - np.conj(ExL_w-E0xL_w) * (HyL_w-H0yL_w)

freq = freq*2*np.pi*27.2114

fig, ax1 = plt.subplots(figsize=(12,8))

N = len(freq)
spec = np.real(Pw[:N//2])
ax1.plot(freq[:N//2], spec)

plt.tight_layout()
plt.show()

file_out = open("spec.dat", "w")

for ii in range(N//2):
   file_out.write(str(freq[ii])+"    "+str(spec[ii])+"\n")

file_out.close()

