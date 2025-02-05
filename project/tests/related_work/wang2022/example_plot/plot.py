import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
speed_data = pd.read_csv('cleaned_speed_data.csv')
lon_accel_data = pd.read_csv('final_filtered_lon_accel.csv')
lat_accel_data = pd.read_csv('uniform_lat_accel_data.csv')
lane_offset_data = pd.read_csv('filtered_lane_offset_data.csv')
steering_wheel_rate_data = pd.read_csv('filtered_steering_wheel_rate_data.csv')

# Extract relevant columns
# Adjust column names based on dataset structure
time_speed = speed_data.iloc[:, 0]
speed = speed_data.iloc[:, 1]

time_lon_accel = lon_accel_data['Time']
lon_accel = lon_accel_data['Lon_Accel']

time_lat_accel = lat_accel_data['Time']
lat_accel = lat_accel_data['Lat_Accel']

time_lane_offset = lane_offset_data['Time']
lane_offset = lane_offset_data['Offset']

time_steering_rate = steering_wheel_rate_data['Time']
steering_rate = steering_wheel_rate_data['Steering_Rate']

# Create the plot
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Top subplot for speed and acceleration
ax1 = axs[0]
ax2 = ax1.twinx()

ax1.plot(time_speed, speed, label='Speed', color='blue')
ax2.plot(time_lon_accel, lon_accel, label='Longitudinal Acceleration', color='red', linestyle='--')
ax2.plot(time_lat_accel, lat_accel, label='Lateral Acceleration', color='green', linestyle=':')

ax1.set_ylabel('Speed (m/s)', color='blue')
ax2.set_ylabel('Acceleration (m/s²)', color='black')
ax1.set_xlabel('Time (s)')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.set_title('Speed and Acceleration')

# Bottom subplot for lane offset and steering wheel rate (transposed)
ax3 = axs[1]
ax4 = ax3.twinx()

ax3.plot(time_lane_offset, lane_offset, label='Lane Offset', color='cyan')
ax4.plot(time_steering_rate, steering_rate, label='Steering Wheel Rate', color='magenta')

ax3.set_xlabel('Time (s)', color='black')
ax3.set_ylabel('Lane Offset (cm)', color='black')
ax4.set_ylabel('Steering Wheel Rate (degrees/s)', color='black')

ax3.legend(loc='upper left')
ax4.legend(loc='upper right')
ax3.set_title('Lane Offset and Steering Wheel Rate (Transposed)')

plt.tight_layout()
plt.show()
