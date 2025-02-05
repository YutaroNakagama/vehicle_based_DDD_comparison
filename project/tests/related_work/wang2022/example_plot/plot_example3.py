import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

# File paths for MAT and CSV data
event_times_path = 'EventTimes_S0116_1.mat'
simlsl_path = 'SIMlsl_S0116_1.mat'
speed_data_file = 'cleaned_speed_data.csv'
lon_accel_data_file = 'final_filtered_lon_accel.csv'
lat_accel_data_file = 'uniform_lat_accel_data.csv'
lane_offset_data_file = 'filtered_lane_offset_data.csv'
steering_wheel_rate_data_file = 'filtered_steering_wheel_rate_data.csv'

# Load MAT data
event_times_data = loadmat(event_times_path)
simlsl_data = loadmat(simlsl_path)

# Load CSV data
speed_data = pd.read_csv(speed_data_file)
lon_accel_data = pd.read_csv(lon_accel_data_file)
lat_accel_data = pd.read_csv(lat_accel_data_file)
lane_offset_data = pd.read_csv(lane_offset_data_file)
steering_wheel_rate_data = pd.read_csv(steering_wheel_rate_data_file)

# Extract DRT event times
drt_signal_on = event_times_data.get('DRT_SignalOn', [])[0]
drt_response = event_times_data.get('DRT_response', [])[0]
drt_start_time = drt_signal_on[0]
drt_end_time = drt_response[0]
plot_start_time = drt_start_time - 3
plot_end_time = drt_end_time + 1.5

# Extract relevant data from SIMlsl
lsl_time = simlsl_data['SIM_lsl'][0, :]
position_data = simlsl_data['SIM_lsl'][30, :]
lateral_acc = simlsl_data['SIM_lsl'][19, :]
longitudinal_acc = simlsl_data['SIM_lsl'][18, :]
lane_offset_sim = simlsl_data['SIM_lsl'][28, :]
steering_position = simlsl_data['SIM_lsl'][29, :]
steering_rate_sim = np.diff(steering_position) / np.diff(lsl_time)

# Apply smoothing
def smooth_data(data, sigma=2):
    return gaussian_filter1d(data, sigma=sigma)

smoothed_velocity_sim = smooth_data(np.diff(position_data) / np.diff(lsl_time))
smoothed_lateral_acc_sim = smooth_data(lateral_acc)
smoothed_longitudinal_acc_sim = smooth_data(longitudinal_acc)
smoothed_lane_offset_sim = smooth_data(lane_offset_sim)
smoothed_steering_rate_sim = smooth_data(steering_rate_sim)

# Filter MAT data for the specific time window
mask_sim = (lsl_time >= plot_start_time) & (lsl_time <= plot_end_time)
adjusted_mask_sim = mask_sim[:-1]  # Adjust mask for np.diff results

# Extract and smooth CSV data
time_speed = speed_data.iloc[:, 0]
speed = speed_data.iloc[:, 1]
speed_smooth = smooth_data(speed)

time_lon_accel = lon_accel_data['Time']
lon_accel = lon_accel_data['Lon_Accel']
lon_accel_smooth = smooth_data(lon_accel)

time_lat_accel = lat_accel_data['Time']
lat_accel = lat_accel_data['Lat_Accel']
lat_accel_smooth = smooth_data(lat_accel)

time_lane_offset = lane_offset_data['Time']
lane_offset = lane_offset_data['Offset']
lane_offset_smooth = smooth_data(lane_offset)

time_steering_rate = steering_wheel_rate_data['Time']
steering_rate = steering_wheel_rate_data['Steering_Rate']
steering_rate_smooth = smooth_data(steering_rate)

# Create the figure
fig, axs = plt.subplots(2, 2, figsize=(20, 10))
font_size = 15

# Left plots (CSV data)
# Top left
ax5 = axs[0, 0]
ax6 = ax5.twinx()
ax5.plot(time_speed, speed_smooth, label='Speed (m/s)', color='blue')
ax6.plot(time_lon_accel, lon_accel_smooth, label='Longitudinal Acceleration (m/s²)', color='red')
ax6.plot(time_lat_accel, lat_accel_smooth, label='Lateral Acceleration (m/s²)', color='green', linestyle='--')
ax5.set_title('CSV: Speed and Acceleration', fontsize=font_size)

# Bottom left
ax7 = axs[1, 0]
ax8 = ax7.twinx()
ax7.plot(time_lane_offset, lane_offset_smooth, label='Lane Offset (cm)', color='cyan')
ax8.plot(time_steering_rate, steering_rate_smooth, label='Steering Rate (degrees/s)', color='magenta')
ax7.set_title('CSV: Lane Offset and Steering Rate', fontsize=font_size)

# Right plots (MAT data)
# Top right
ax1 = axs[0, 1]
ax2 = ax1.twinx()
ax1.plot(lsl_time[:-1][adjusted_mask_sim], smoothed_velocity_sim[adjusted_mask_sim], label='Speed (m/s)', color='blue')
ax2.plot(lsl_time[mask_sim], smoothed_lateral_acc_sim[mask_sim], label='Lateral Acceleration (m/s²)', color='orange', linestyle='--')
ax2.plot(lsl_time[mask_sim], smoothed_longitudinal_acc_sim[mask_sim], label='Longitudinal Acceleration (m/s²)', color='green', linestyle=':')
ax1.axvline(x=drt_start_time, color='black', linestyle='--', label='DRT Start')
ax1.axvline(x=drt_end_time, color='black', linestyle='--', label='DRT End')
ax1.set_title('MAT: Speed and Acceleration (Filtered)', fontsize=font_size)

# Bottom right
ax3 = axs[1, 1]
ax4 = ax3.twinx()
ax3.plot(lsl_time[mask_sim], smoothed_lane_offset_sim[mask_sim], label='Lane Offset (cm)', color='red')
ax4.plot(lsl_time[:-1][adjusted_mask_sim], smoothed_steering_rate_sim[adjusted_mask_sim], label='Steering Rate (deg/s)', color='purple', linestyle='--')
ax3.axvline(x=drt_start_time, color='black', linestyle='--', label='DRT Start')
ax3.axvline(x=drt_end_time, color='black', linestyle='--', label='DRT End')
ax3.set_title('MAT: Lane Offset and Steering Rate (Filtered)', fontsize=font_size)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

