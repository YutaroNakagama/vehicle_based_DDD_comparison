# Load the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

font_size = 15

# File paths (adjusting based on uploaded filenames)
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

# Apply smoothing to all data series
speed_smooth = gaussian_filter1d(speed, sigma=2)
lon_accel_smooth = gaussian_filter1d(lon_accel, sigma=2)
lat_accel_smooth = gaussian_filter1d(lat_accel, sigma=2)
lane_offset_smooth = gaussian_filter1d(lane_offset, sigma=2)
steering_rate_smooth = gaussian_filter1d(steering_rate, sigma=2)

# Create the plot with smoothed data
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Top subplot for smoothed speed and acceleration
ax1 = axs[0]
ax2 = ax1.twinx()

ax1.plot(time_speed, speed_smooth, label='Speed (m/s)', color='blue')
ax2.plot(time_lon_accel, lon_accel_smooth, label='Longitudinal Acceleration (m/s²)', color='red')
ax2.plot(time_lat_accel, lat_accel_smooth, label='Lateral Acceleration (m/s²)', color='green', linestyle='--')

ax1.set_xlim(0, 9.5)
ax2.set_xlim(0, 9.5)

# Apply axis labels
ax1.set_ylabel('Speed (m/s)', fontsize=font_size)
ax2.set_ylabel('Acceleration (m/s²)', fontsize=font_size)

# Add vertical dotted lines at specific time points
ax1.axvline(x=4, color='black', linestyle='--', linewidth=1, zorder=2)
ax2.axvline(x=4, color='black', linestyle='--', linewidth=1, zorder=2)
ax1.axvline(x=7.5, color='black', linestyle='--', linewidth=1, zorder=2)
ax2.axvline(x=7.5, color='black', linestyle='--', linewidth=1, zorder=2)

# Add text annotations for time segments
ax1.text(2, 14, 'Baseline $i_1$', fontsize=font_size, ha='center', va='bottom')
ax1.text(5.75, 14, 'Event $i$', fontsize=font_size, ha='center', va='bottom')
ax1.text(8.5, 14, 'Baseline $i_2$', fontsize=font_size, ha='center', va='bottom')

# Combine legends for the top subplot
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=font_size)
legend1 = ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=font_size, fancybox=False)
legend1.get_frame().set_edgecolor('black')  # 枠線を黒にする
legend1.get_frame().set_facecolor('white')  # 背景を白にする
legend1.set_zorder(1)                     # 凡例を最前面に設定

# Bottom subplot for smoothed lane offset and steering wheel rate
ax3 = axs[1]
ax4 = ax3.twinx()

ax3.plot(time_lane_offset, lane_offset_smooth, label='Lane Offset (cm)', color='cyan')
ax4.plot(time_steering_rate, steering_rate_smooth, label='Steering Wheel Rate (degrees/s)', color='magenta')

ax3.set_xlim(0, 9.5)
ax4.set_xlim(0, 9.5)

# Apply axis labels
ax3.set_ylabel('Lane Offset (cm)', fontsize=font_size)
ax4.set_ylabel('Steering Rate (degrees/s)', fontsize=font_size)
ax4.set_xlabel('Time (s)', fontsize=font_size)

# Add vertical dotted lines at specific time points
ax3.axvline(x=4, color='black', linestyle='--', linewidth=1)
ax4.axvline(x=4, color='black', linestyle='--', linewidth=1)
ax3.axvline(x=7.5, color='black', linestyle='--', linewidth=1)
ax4.axvline(x=7.5, color='black', linestyle='--', linewidth=1)

# Add text annotations for time segments on the bottom subplot
ax3.text(2, 2000, 'Baseline $i_1$', fontsize=font_size, ha='center', va='bottom')
ax3.text(5.75, 2000, 'Event $i$', fontsize=font_size, ha='center', va='bottom')
ax3.text(8.5, 2000, 'Baseline $i_2$', fontsize=font_size, ha='center', va='bottom')

# Combine legends for the bottom subplot
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right', fontsize=font_size)
legend2 = ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right', fontsize=font_size, fancybox=False)
legend2.get_frame().set_facecolor('white')  # 背景を白にする
legend2.get_frame().set_edgecolor('black')  # 枠線を黒にする
legend2.set_zorder(10)                     # 凡例を最前面に設定

# Customize x-axis ticks and labels
ticks = [0, 4, 7.5, 9.5]
labels = ['$t_0$', '$t_1$', '$t_2$', '$t_3$']

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=font_size, fontname='Times New Roman')
    ax.tick_params(axis='both', direction='in')  # 目盛りを内向きに設定
    ax.tick_params(axis='y', labelsize=font_size)  # 15を希望するフォントサイズに変更

# Top subplot for smoothed speed and acceleration
ax1.set_ylim(0, 14)  # Speed y-axis range
ax2.set_ylim(-0.25, 0.2)  # Acceleration y-axis range

# Bottom subplot for smoothed lane offset and steering wheel rate
ax3.set_ylim(-2500, 2000)  # Lane Offset y-axis range
ax4.set_ylim(-20, 160)  # Steering Wheel Rate y-axis range

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

# Add a shared X-axis title
fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=font_size)

# Adjust layout to prevent overlapping of "Time (s)" title
plt.tight_layout()
fig.subplots_adjust(bottom=0.1)  # Adjust the bottom margin

plt.show()

