import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Function to apply a bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

# Load the .mat files
fp = '../../dataset/Aygun2024/physio/S0116/' 
eeg_2_mat_file_path     = fp + 'EEG_S0116_2.mat'
nirs_2_mat_file_path    = fp + 'NIRS_S0116_2.mat'
pupil_2_mat_file_path   = fp + 'PupilData_S0116_2.mat'
sim_2_mat_file_path     = fp + 'SIMlsl_S0116_2.mat'
physio_2_mat_file_path  = fp + 'Physio_S0116_2.mat'

# Loading the .mat contents
eeg_2_mat_contents = scipy.io.loadmat(eeg_2_mat_file_path)
nirs_2_mat_contents = scipy.io.loadmat(nirs_2_mat_file_path)
pupil_2_mat_contents = scipy.io.loadmat(pupil_2_mat_file_path)
sim_2_mat_contents = scipy.io.loadmat(sim_2_mat_file_path)
physio_2_mat_contents = scipy.io.loadmat(physio_2_mat_file_path)

# Extracting the relevant data from each file
eeg_2_data = eeg_2_mat_contents['rawEEG']
nirs_2_data = nirs_2_mat_contents['NIRSdata']
pupil_2d_2_data = pupil_2_mat_contents['Pupil2D']
sim_lsl_2_data = sim_2_mat_contents['SIM_lsl']
physio_2_data = physio_2_mat_contents['PhysioData']

# Transpose the data for easier processing
eeg_2_data_transposed = eeg_2_data.T
nirs_2_data_transposed = nirs_2_data.T
pupil_2d_2_data_transposed = pupil_2d_2_data.T
sim_lsl_2_data_transposed = sim_lsl_2_data.T
physio_2_data_transposed = physio_2_data.T

# Extract time and representative signal for visualization from each dataset
eeg_2_time = eeg_2_data_transposed[:, 0]
eeg_2_signal = eeg_2_data_transposed[:, 1]

nirs_2_time = nirs_2_data_transposed[:, 0]
nirs_2_signal = nirs_2_data_transposed[:, 2]

pupil_2_time = pupil_2d_2_data_transposed[:, 0]
pupil_2_signal = pupil_2d_2_data_transposed[:, 1]

sim_lsl_2_time = sim_lsl_2_data_transposed[:, 0]  # LSL time
sim_lsl_2_signal = sim_lsl_2_data_transposed[:, -1]  # Using the 'Velocity_m_per_s' equivalent

physio_2_time = physio_2_data_transposed[:, 0]
physio_2_signal = physio_2_data_transposed[:, 1]

# Sampling frequency (assuming 500 Hz)
fs = 500

# Applying the bandpass filter for alpha waves (8-13 Hz) on EEG Channel FC1
alpha_wave = bandpass_filter(eeg_2_signal, 8, 13, fs)

# Plotting the representative signals from each dataset
plt.figure(figsize=(10, 25))

# rawEEG Alpha Wave data plot
plt.subplot(5, 1, 1)
plt.plot(eeg_2_time, alpha_wave, label='Alpha Wave (8-13 Hz) - Channel FC1', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (nV)')
plt.title('Alpha Wave - rawEEG Data (Channel FC1)')
plt.grid(True)

# NIRSdata plot
plt.subplot(5, 1, 2)
plt.plot(nirs_2_time, nirs_2_signal, label='fNIRS Channel 2 (NIRS)', color='m')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('NIRS Data - S0116_2')
plt.grid(True)

# Pupil2D data plot
plt.subplot(5, 1, 3)
plt.plot(pupil_2_time, pupil_2_signal, label='Left Eye Pupil Diameter (Pupil2D)', color='c')
plt.xlabel('Time (s)')
plt.ylabel('Pupil Diameter')
plt.title('Pupil2D Data - S0116_2')
plt.grid(True)

# SIM_lsl data plot
plt.subplot(5, 1, 4)
plt.plot(sim_lsl_2_time, sim_lsl_2_signal, label='Velocity (SIM_lsl)', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('SIM_lsl Data - S0116_2')
plt.grid(True)

# PhysioData plot
plt.subplot(5, 1, 5)
plt.plot(physio_2_time, physio_2_signal, label='Blood Pressure (Physio)', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Blood Pressure (mmHg)')
plt.title('PhysioData - S0116_2')
plt.grid(True)

plt.tight_layout()
plt.show()
