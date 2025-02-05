# Load the necessary libraries and files
import pandas as pd
import matplotlib.pyplot as plt

# File paths
speed_data_file = 'cleaned_speed_data.csv'
lon_accel_data_file = 'final_filtered_lon_accel.csv'
lat_accel_data_file = 'uniform_lat_accel_data.csv'
lane_offset_data_file = 'filtered_lane_offset_data.csv'
steering_wheel_rate_data_file = 'filtered_steering_wheel_rate_data.csv'

# Load datasets
speed_data = pd.read_csv(speed_data_file)
lon_accel_data = pd.read_csv(lon_accel_data_file)
lat_accel_data = pd.read_csv(lat_accel_data_file)
lane_offset_data = pd.read_csv(lane_offset_data_file)
steering_wheel_rate_data = pd.read_csv(steering_wheel_rate_data_file)

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


# Add additional text annotations for "Event_i" and "Baseline_i2"
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Top subplot for speed and acceleration
ax1 = axs[0]
ax2 = ax1.twinx()

ax1.plot(time_speed, speed, label='Speed (m/s)', color='blue')
ax2.plot(time_lon_accel, lon_accel, label='Longitudinal Acceleration (m/s²)', color='red')  # Solid line
ax2.plot(time_lat_accel, lat_accel, label='Lateral Acceleration (m/s²)', color='green', linestyle='--')

ax1.set_xlim(0, 9.5)
ax2.set_xlim(0, 9.5)

# Apply axis labels
ax1.set_ylabel('Speed (m/s)', fontsize=10, fontname='Times New Roman')
ax2.set_ylabel('Acceleration (m/s²)', fontsize=10, fontname='Times New Roman')

# Add vertical dotted lines at t=4 (t_1) and t=7.5 (t_2)
ax1.axvline(x=4, color='black', linestyle='--', linewidth=1)
ax2.axvline(x=4, color='black', linestyle='--', linewidth=1)
ax1.axvline(x=7.5, color='black', linestyle='--', linewidth=1)
ax2.axvline(x=7.5, color='black', linestyle='--', linewidth=1)

# Combine legends for the top subplot
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10, frameon=False, prop={'family': 'Times New Roman'})

# Add text annotations for "Baseline_i1", "Event_i", and "Baseline_i2"
ax1.text(2, 14.5, 'Baseline $i_1$', fontsize=10, fontname='Times New Roman', ha='center', va='center')
ax1.text(5.75, 14.5, 'Event $i$', fontsize=10, fontname='Times New Roman', ha='center', va='center')
ax1.text(8.5, 14.5, 'Baseline $i_2$', fontsize=10, fontname='Times New Roman', ha='center', va='center')

# Bottom subplot for lane offset and steering wheel rate (transposed)
ax3 = axs[1]
ax4 = ax3.twinx()

ax3.plot(time_lane_offset, lane_offset, label='Lane Offset (cm)', color='cyan')
ax4.plot(time_steering_rate, steering_rate, label='Steering Wheel Rate (degrees/s)', color='magenta')

ax3.set_xlim(0, 9.5)
ax4.set_xlim(0, 9.5)

# Apply axis labels
ax3.set_ylabel('Lane Offset (cm)', fontsize=10, fontname='Times New Roman')
ax4.set_ylabel('Steering Wheel Rate (degrees/s)', fontsize=10, fontname='Times New Roman')

# Add vertical dotted lines at t=4 (t_1) and t=7.5 (t_2)
ax3.axvline(x=4, color='black', linestyle='--', linewidth=1)
ax4.axvline(x=4, color='black', linestyle='--', linewidth=1)
ax3.axvline(x=7.5, color='black', linestyle='--', linewidth=1)
ax4.axvline(x=7.5, color='black', linestyle='--', linewidth=1)

# Combine legends for the bottom subplot
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right', fontsize=10, frameon=False, prop={'family': 'Times New Roman'})

# Add text annotations for "Baseline_i1", "Event_i", and "Baseline_i2" on the bottom subplot
ax3.text(2, 2200, 'Baseline $i_1$', fontsize=10, fontname='Times New Roman', ha='center', va='center')
ax3.text(5.75, 2200, 'Event $i$', fontsize=10, fontname='Times New Roman', ha='center', va='center')
ax3.text(8.5, 2200, 'Baseline $i_2$', fontsize=10, fontname='Times New Roman', ha='center', va='center')

# Customize x-axis ticks and labels
ticks = [0, 4, 7.5, 9.5]
labels = ['t_0', 't_1', 't_2', 't_3']

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=10, fontname='Times New Roman')

# Set black border for both subplots
for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)
for spine in ax2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)
for spine in ax3.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)
for spine in ax4.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.show()

