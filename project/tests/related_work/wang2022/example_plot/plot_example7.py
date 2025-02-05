import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
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

# 位置データの外れ値除去と補完
def remove_outliers_and_interpolate(data, threshold=2):
    # 外れ値を検出するための閾値設定
    mean = np.mean(data)
    std_dev = np.std(data)

    # 外れ値をNaNに設定（mean ± threshold * std_dev を超える値を外れ値とする）
    outlier_mask = np.abs(data - mean) > threshold * std_dev
    data[outlier_mask] = np.nan  # 外れ値をNaNに置き換え

    # NaNを線形補完で補完
    valid_data_indices = ~np.isnan(data)  # 有効なデータのインデックス
    interpolator = interp1d(np.where(valid_data_indices)[0], data[valid_data_indices], kind='linear', fill_value="extrapolate")
    data_filled = interpolator(np.arange(len(data)))  # 補完されたデータ

    return data_filled

# Apply smoothing
def smooth_data(data, sigma=3):
    return gaussian_filter1d(data, sigma=sigma)

# 位置データに対して外れ値除去と補完を適用
position_data_cleaned = remove_outliers_and_interpolate(position_data)
velocity_sim = np.diff(position_data_cleaned) / np.diff(lsl_time)
velocity_sim = remove_outliers_and_interpolate(velocity_sim)
# 速度の平滑化
smoothed_velocity_sim = smooth_data(velocity_sim)

#smoothed_velocity_sim = smooth_data(np.diff(position_data) / np.diff(lsl_time))
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
fig, axs = plt.subplots(2, 1, figsize=(8, 5))
font_size = 12

# Left plots (CSV data)
# Top left
#ax5 = axs[0, 0]
ax5 = axs[0]
ax6 = ax5.twinx()
#ax5.plot(time_speed, speed_smooth, label='Speed (m/s)', color='blue')
ax5.plot(time_lane_offset, lane_offset_smooth, label='Lane Offset', color='cyan')
ax6.plot(time_lon_accel, lon_accel_smooth, label='Longitudinal Acceleration', color='red')
ax6.plot(time_lat_accel, lat_accel_smooth, label='Lateral Acceleration', color='green', linestyle='--')
ax5.axvline(x=4, color='black', linestyle='--')
ax5.axvline(x=7.5, color='black', linestyle='--')
#ax5.set_xlabel('Time (s)', fontsize=font_size)
#ax5.set_ylabel('Speed (m/s)', fontsize=font_size, color='black')
ax5.set_ylabel('Lane Offset (cm)', fontsize=font_size, color='black')
ax6.set_ylabel('Acceleration (m/s²)', fontsize=font_size, color='black')
#ax5.legend(loc='lower right', fontsize=font_size)
#ax5.grid(True)
lines5, labels5 = ax5.get_legend_handles_labels()
lines6, labels6 = ax6.get_legend_handles_labels()
legend1 = ax5.legend(lines5 + lines6, labels5 + labels6, loc='lower right', fontsize=font_size-3, fancybox=False, framealpha=1, edgecolor='black')
legend1.set_zorder(10)
#ax5.grid(True)

# Add text annotations
#ax5.set_ylim(0, 14)
ax6.set_ylim(-0.25, 0.2)
annotation_y = 0.2
ax6.text(2,     annotation_y, 'Baseline $i_1$', fontsize=font_size, ha='center', va='bottom')
ax6.text(5.75,  annotation_y, 'Event $i$', fontsize=font_size, ha='center', va='bottom')
ax6.text(8.5,   annotation_y, 'Baseline $i_2$', fontsize=font_size, ha='center', va='bottom')

## Bottom left
#ax7 = axs[1, 0]
#ax8 = ax7.twinx()
#ax7.plot(time_lane_offset, lane_offset_smooth, label='Lane Offset (cm)', color='cyan')
#ax8.plot(time_steering_rate, steering_rate_smooth, label='Steering Rate (degrees/s)', color='magenta')
#ax7.axvline(x=4, color='black', linestyle='--')
#ax7.axvline(x=7.5, color='black', linestyle='--')
#ax7.set_xlabel('Time (s)', fontsize=font_size)
#ax7.set_ylabel('Lane Offset (cm)', fontsize=font_size, color='black')
##ax8.set_ylabel('Steering Rate (degrees/s)', fontsize=font_size, color='black')
##ax7.legend(loc='upper right', fontsize=font_size)
##ax7.grid(True)
#lines7, labels7 = ax7.get_legend_handles_labels()
#lines8, labels8 = ax8.get_legend_handles_labels()
#ax7.legend(lines7 + lines8, labels7 + labels8, loc='upper right', fontsize=font_size, fancybox=False, framealpha=1, edgecolor='black')
##ax7.grid(True)
#
## Add text annotations
#ax7.set_ylim(-2500,2000)
#ax8.set_ylim(-20,160)
#ax7.text(2, 2000, 'Baseline $i_1$', fontsize=font_size, ha='center', va='bottom')
#ax7.text(5.75, 2000, 'Event $i$', fontsize=font_size, ha='center', va='bottom')
#ax7.text(8.5, 2000, 'Baseline $i_2$', fontsize=font_size, ha='center', va='bottom')

# Right plots (MAT data)
# Top right
#ax1 = axs[0, 1]
ax1 = axs[1]
ax2 = ax1.twinx()
#ax1.plot(lsl_time[:-1][adjusted_mask_sim], smoothed_velocity_sim[adjusted_mask_sim], label='Speed (m/s)', color='blue')
ax1.plot(lsl_time[mask_sim], smoothed_lane_offset_sim[mask_sim], label='Lane Offset', color='cyan', zorder=5)
ax2.plot(lsl_time[mask_sim], smoothed_longitudinal_acc_sim[mask_sim], label='Longitudinal Acceleration', color='red', zorder=5)
ax2.plot(lsl_time[mask_sim], smoothed_lateral_acc_sim[mask_sim], label='Lateral Acceleration', color='green', linestyle='--', zorder=5)
ax1.axvline(x=drt_start_time, color='black', linestyle='--')
ax1.axvline(x=drt_end_time, color='black', linestyle='--')
ax1.set_xlabel('Time (s)', fontsize=font_size)
#ax1.set_ylabel('Speed (m/s)', fontsize=font_size, color='black')
ax1.set_ylabel('Lane Offset (cm)', fontsize=font_size, color='black')
ax2.set_ylabel('Acceleration (m/s²)', fontsize=font_size, color='black')
#ax1.legend(loc='lower right', fontsize=font_size)
#ax1.grid(True)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend3 = ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=font_size-3, fancybox=False, framealpha=1, edgecolor='black')
legend3.set_zorder(10)
#ax1.grid(True)

# Add text annotations
ax1.set_ylim([4.5,5.5])
ax2.set_ylim([-0.002,0.002])
ax2.set_yticks([-0.002, -0.001, 0, 0.001, 0.002])
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # 2 decimal places
y_pos = 0.002 #max(smoothed_velocity_sim[adjusted_mask_sim]) * 1.1 
ax2.text(drt_start_time-1.5,                y_pos, 'Baseline $i_1$', fontsize=font_size, ha='center', va='bottom')
ax2.text((drt_start_time+drt_end_time)/2,   y_pos, 'Event $i$', fontsize=font_size, ha='center', va='bottom')
ax2.text(drt_end_time+0.7,                  y_pos, 'Baseline $i_2$', fontsize=font_size, ha='center', va='bottom')

## Bottom right
#ax3 = axs[1, 1]
#ax4 = ax3.twinx()
#ax3.plot(lsl_time[mask_sim], smoothed_lane_offset_sim[mask_sim], label='Lane Offset (cm)', color='cyan')
#ax4.plot(lsl_time[:-1][adjusted_mask_sim], smoothed_steering_rate_sim[adjusted_mask_sim], label='Steering Rate (deg/s)', color='magenta')
#ax3.axvline(x=drt_start_time, color='black', linestyle='--')
#ax3.axvline(x=drt_end_time, color='black', linestyle='--')
#ax3.set_xlabel('Time (s)', fontsize=font_size)
#ax3.set_ylabel('Lane Offset (cm)', fontsize=font_size, color='black')
##ax4.set_ylabel('Steering Rate (deg/s)', fontsize=font_size, color='black')
##ax3.legend(loc='upper right', fontsize=font_size)
##ax3.grid(True)
#lines3, labels3 = ax3.get_legend_handles_labels()
#lines4, labels4 = ax4.get_legend_handles_labels()
#ax3.legend(lines3 + lines4, labels3 + labels4, loc='lower left', fontsize=font_size, fancybox=False, framealpha=1, edgecolor='black')
##ax3.grid(True)
#
## Add text annotations
##ax3.set_ylim([0,2000])
#y_pos = 0.01 + max(smoothed_lane_offset_sim[mask_sim]) 
#ax3.text(drt_start_time-1.5, y_pos, 'Baseline $i_1$', fontsize=font_size, ha='center', va='bottom')
#ax3.text((drt_start_time+drt_end_time)/2, y_pos, 'Event $i$', fontsize=font_size, ha='center', va='bottom')
#ax3.text(drt_end_time+0.7, y_pos, 'Baseline $i_2$', fontsize=font_size, ha='center', va='bottom')

# Customize x-axis ticks and labels
ticks = [0, 4, 7.5, 9.5]
labels = ['$t_0$', '$t_1$', '$t_2$', '$t_3$']

#for ax in [ax5, ax6, ax7, ax8]:
for ax in [ax5, ax6]:
    ax.set_xlim(0,9.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=font_size, fontname='Times New Roman')
    ax.tick_params(axis='both', direction='in')  # 目盛りを内向きに設定
    ax.tick_params(axis='y', labelsize=font_size)  # 15を希望するフォントサイズに変更

# Customize x-axis ticks and labels
ticks = [drt_start_time - 3, drt_start_time, drt_end_time, drt_end_time+1.5]
labels = ['$t_0$', '$t_1$', '$t_2$', '$t_3$']

for ax in [ax1, ax2]:
    ax.set_xlim(drt_start_time-3,drt_end_time+1.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=font_size, fontname='Times New Roman')
    ax.tick_params(axis='both', direction='in')  # 目盛りを内向きに設定
    ax.tick_params(axis='y', labelsize=font_size)  # 15を希望するフォントサイズに変更

# Adjust layout
#plt.tight_layout()
fig.subplots_adjust(bottom=0.1)
plt.show()

