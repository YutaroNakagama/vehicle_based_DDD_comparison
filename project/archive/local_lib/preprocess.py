
import datetime
import numpy as np
import pandas as pd

#from tqdm import tqdm
from typing import List, Optional, Tuple
from numpy.typing import NDArray
from scipy import signal
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def calc_fft(yawrate_i_fft, yawrate_a_fft, N_curve, dt):
    F_i,F_a  = np.fft.fft(yawrate_i_fft),  np.fft.fft(yawrate_a_fft)
    freq = np.fft.fftfreq(N_curve, d=dt)
    F_i,F_a = F_i/(N_curve/2), F_a/(N_curve/2)
    Amp_i, Amp_a  = np.abs(F_i), np.abs(F_a)
    return Amp_i, Amp_a, freq

# LPF 
def butter_lowpass(lowcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    return b, a

def butter_lowpass_filter(x, lowcut, fs, order=4):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, x)
    return y

def curve_check(
                    yaw_rate_i_before, 
                    yaw_rate_i_after, 
                    yaw_rate_a_curve, 
                    in_curve_flg_local,
                    dt
                    ):
    before_curve_flg = max(abs(yaw_rate_i_before))     < 0.01 
    after_curve_flg  = abs(sum(yaw_rate_i_after * dt)) > 1.5 
    in_curve_act_yaw = max(abs(yaw_rate_a_curve))      > 0.5 
    return (before_curve_flg and after_curve_flg and in_curve_act_yaw and not in_curve_flg_local)

def load_dataset():
    csv_path_veh = ["" for x in range(8)]
    csv_path_eeg = ["" for x in range(8)]
    
    #dataset_fp = "../../../../git_backup/ynakagama/analysis/DeepLearning/dataset/"
    dataset_fp = "../../dataset/bz2/"
    dataset_format = "bz2" # "csv"
    
    csv_path_veh[0] = dataset_fp + "dataset_pandas_2023-09-10_1404_20230930."   + dataset_format
    csv_path_veh[1] = dataset_fp + "dataset_pandas_2023-11-14_0025_2023-11-14." + dataset_format
    csv_path_veh[2] = dataset_fp + "dataset_pandas_2023-11-15_0115_2023-11-15." + dataset_format
    csv_path_veh[3] = dataset_fp + "dataset_pandas_2023-11-19_2306_2023-11-25." + dataset_format
    csv_path_veh[4] = dataset_fp + "dataset_pandas_2023-11-21_0004_2023-11-26." + dataset_format
    csv_path_veh[5] = dataset_fp + "dataset_pandas_2023-11-26_0959_2023-11-26." + dataset_format
    csv_path_veh[6] = dataset_fp + "dataset_pandas_2023-11-26_1346_2023-11-26." + dataset_format
    csv_path_veh[7] = dataset_fp + "dataset_pandas_2023-12-02_1003_2023-12-02." + dataset_format
    
    csv_path_eeg[0] = dataset_fp + "dataset_EEG_2023-09-10_1404_20230909."   + dataset_format
    csv_path_eeg[1] = dataset_fp + "dataset_EEG_2023-11-14_0022_20231104."   + dataset_format
    csv_path_eeg[2] = dataset_fp + "dataset_EEG_2023-11-15_0113_2023-11-15." + dataset_format
    csv_path_eeg[3] = dataset_fp + "dataset_EEG_2023-11-19_2303_2023-11-25." + dataset_format
    csv_path_eeg[4] = dataset_fp + "dataset_EEG_2023-11-21_0001_2023-11-26." + dataset_format
    csv_path_eeg[5] = dataset_fp + "dataset_EEG_2023-11-26_0959_2023-11-26." + dataset_format
    csv_path_eeg[6] = dataset_fp + "dataset_EEG_2023-11-26_1347_2023-11-26." + dataset_format
    csv_path_eeg[7] = dataset_fp + "dataset_EEG_2023-12-02_1003_2023-12-02." + dataset_format

    time_str             = []
    yawrate_i, yawrate_a = [],[] 
    angular_velocity     = []
    curvature            = []
    steering_angle       = []
    dev_center           = []
    time_eeg             = [] 
    alpha_af7,alpha_af8  = [],[] 
    beta_af7, beta_af8   = [],[] 
    
    #for i in range(len(csv_path_veh)):
    for i in range(0,1): #
        #csv_df_veh = pd.read_csv(csv_path_veh[i]).query('-5<yaw_rate_a<5').query('-5<yaw_rate_i<5')
        #csv_df_eeg = pd.read_csv(csv_path_eeg[i])
        csv_df_veh = pd.read_pickle(csv_path_veh[i]).query('-5<yaw_rate_a<5').query('-5<yaw_rate_i<5')
        csv_df_eeg = pd.read_pickle(csv_path_eeg[i])
        #csv_df_veh = pickle5.load(csv_path_veh[i]).query('-5<yaw_rate_a<5').query('-5<yaw_rate_i<5')
        #csv_df_eeg = pickle5.load(csv_path_eeg[i])
        
        time_str         = np.concatenate((time_str        ,csv_df_veh['time'].values      ), axis = 0) 
        yawrate_i        = np.concatenate((yawrate_i       ,csv_df_veh['yaw_rate_i'].values), axis = 0) 
        yawrate_a        = np.concatenate((yawrate_a       ,csv_df_veh['yaw_rate_a'].values), axis = 0) 
        angular_velocity = np.concatenate((angular_velocity,csv_df_veh['angular_vel']      ), axis = 0) 
        curvature        = np.concatenate((curvature       ,csv_df_veh['curvature']        ), axis = 0) 
        steering_angle   = np.concatenate((steering_angle  ,csv_df_veh['steering_angle']   ), axis = 0) 
        dev_center       = np.concatenate((dev_center      ,csv_df_veh['dev_center']       ), axis = 0) 
    
        time_eeg  = np.concatenate((time_eeg  ,csv_df_eeg['time'].values     ), axis = 0) 
        alpha_af7 = np.concatenate((alpha_af7 ,csv_df_eeg['alpha_af7'].values), axis = 0) 
        alpha_af8 = np.concatenate((alpha_af8 ,csv_df_eeg['alpha_af8'].values), axis = 0) 
        beta_af7  = np.concatenate((beta_af7  ,csv_df_eeg['beta_af7'].values ), axis = 0) 
        beta_af8  = np.concatenate((beta_af8  ,csv_df_eeg['beta_af8'].values ), axis = 0) 
    
        print(len(time_str))
    
    alpha_local = (alpha_af7 + alpha_af8) / 2
    beta_local  = (beta_af7  + beta_af8)  / 2
    beta_alpha_local = beta_local / alpha_local

    return time_str, yawrate_i, yawrate_a,\
           angular_velocity, curvature, steering_angle, dev_center,\
           time_eeg, alpha_local, beta_local, beta_alpha_local

def make_train_dataset(NUM_DATA, NUM_RNN, yawrate_i_filt, yawrate_a_filt):
    x0, x1, y, x0_vali, x1_vali, y_vali = [],[],[],[],[],[]
    for j in (range(NUM_DATA)):
        x0.append(yawrate_i_filt[j  :j+NUM_RNN  ])      
        x1.append(yawrate_a_filt[j  :j+NUM_RNN  ])      
        y.append(yawrate_a_filt[ j+1:j+NUM_RNN+1])  
    else:
        x0_train = np.array(x0).reshape(NUM_DATA, NUM_RNN, 1) 
        x1_train = np.array(x1).reshape(NUM_DATA, NUM_RNN, 1) 
        x_train  = np.concatenate([x0_train, x1_train],2)
        y_train  = np.array(y).reshape(NUM_DATA, NUM_RNN, 1)
    return x0_train,x1_train,x_train,y_train

# linear regression prediction 
def predict(
    model: LinearRegression,
    pf: PolynomialFeatures,
    xs: NDArray[np.float64],
) -> NDArray[np.float64]:
    xs = xs[:, np.newaxis]
    xs = pf.fit_transform(xs)
    ys = model.predict(xs)
    return ys  # type:ignore

def calculate_mse(
    random_xs: NDArray[np.float64],
    random_ys: NDArray[np.float64],
    model: LinearRegression,
    pf: PolynomialFeatures,
) -> np.float64:
    pred_ys = predict(model, pf, random_xs)
    rmse = mean_squared_error(random_ys, pred_ys)
    return rmse  # type:ignore

def calc_rmse(NUM_DATA, y_test, yawrate_a_filt):
    rmse_local,rmse_local_cnt = 0,0
    for j in (range(NUM_DATA)):
        if y_test[j] > 0:
            rmse_local     = rmse_local + abs(y_test[j] - yawrate_a_filt[j])
            rmse_local_cnt += 1
    return rmse_local,rmse_local_cnt

def execute_regression(
    xs: NDArray[np.float64], ys: NDArray[np.float64], degree: int
) -> Tuple[LinearRegression, PolynomialFeatures]:
    xs = xs[:, np.newaxis]
    ys = ys[:, np.newaxis]

    polynomial_features = PolynomialFeatures(degree=degree)
    xs_poly = polynomial_features.fit_transform(xs)
    model = LinearRegression()
    model.fit(xs_poly, ys)
    return model, polynomial_features

def calculate_aic_and_bic(
    mse: np.float64, degree: int, data_num: int
) -> Tuple[np.float64, np.float64]:
    a = data_num * np.log(mse)
    return (a + 2 * degree, a + degree * np.log(data_num))

def save_aic_and_bic(
    curve_cnt:int, degrees: List[int], aics: List[np.float64], bics: List[np.float64], data_type: str
) -> None:
    csv_df = pd.DataFrame(
                    data={'train_start': pd.Series(curve_cnt), 
                          'degree[2]': pd.Series((aics[0]-aics[2])/abs(aics[2])), 
                          'degree[3]': pd.Series((aics[1]-aics[2])/abs(aics[2])), 
                          'degree[4]': pd.Series((aics[2]-aics[2])/abs(aics[2])), 
                          'degree[5]': pd.Series((aics[3]-aics[2])/abs(aics[2])), 
                          'degree[6]': pd.Series((aics[4]-aics[2])/abs(aics[2])), 
                          'degree[7]': pd.Series((aics[5]-aics[2])/abs(aics[2])), 
                          'degree[8]': pd.Series((aics[6]-aics[2])/abs(aics[2])), 
                          'degree[9]': pd.Series((aics[7]-aics[2])/abs(aics[2])), 
                          })     
    csv_path = "../result/csv/aic/aic_norm_aic3_manual_"+data_type+"_"+str(datetime.datetime.utcnow().date())+".csv"
    print(csv_path)
    header_flg = True if curve_cnt==1 else False
    csv_df.to_csv(csv_path,mode='a', header=header_flg) 

def make_aic_and_bic(
    curve_cnt: int,
    random_xs: NDArray[np.float64],
    random_ys: NDArray[np.float64],
    N: int,
    max_degree: int,
    data_type: str,
) -> None:
    aics,bics = [],[]
    degrees = list(range(2, max_degree))
    for degree in degrees:
        model, polynomial_features = execute_regression(
            random_xs, random_ys, degree
        )

        mse = calculate_mse(random_xs, random_ys, model, polynomial_features)

        (aic, bic) = calculate_aic_and_bic(mse, degree, N)
        aics.append(aic)
        bics.append(bic)

    save_aic_and_bic(curve_cnt, degrees, aics, bics, data_type)

def curve_detection(
                    time_str,
                    dt,
                    N_curve,
                    yawrate_i,
                    yawrate_a,
                    fs,
                    alpha_local,
                    beta_local,
                    beta_alpha_local,
                    eeg_high_th,
                    eeg_low_th,
                    num_train_dateset
                    ):
    in_curve_flg, in_curve_cnt, curve_cnt, curve_train_cnt = 0,0,0,0
    curve_focus_train, curve_focus_test, curve_drow = [],[],[]
    print("curve detection")
    #for i in tqdm(range(len(time_str)+int(3/dt)-N_curve)):
    for i in range(len(time_str)+int(3/dt)-N_curve):
        if in_curve_flg == True: in_curve_cnt += 1
        if in_curve_cnt > (3//dt): in_curve_flg,in_curve_cnt=0,0
        if curve_check(
                           yawrate_i[i          :i+int(1/dt)],
                           yawrate_i[i+int(1/dt):i+int(4/dt)],
                           yawrate_a[i          :i+int(fs*5)],
                           in_curve_flg,
                           dt
                           ):
            in_curve_flg = True
            idx_start_eeg = int(255.5*(1.0+time_str[i-int(3/dt)        ]))
            idx_end_eeg   = int(255.5*(1.0+time_str[i-int(3/dt)+N_curve]))
            alpha_ave      = np.nanmean(alpha_local[     idx_start_eeg:idx_end_eeg])
            beta_ave       = np.nanmean(beta_local[      idx_start_eeg:idx_end_eeg])
            beta_alpha_ave = np.nanmean(beta_alpha_local[idx_start_eeg:idx_end_eeg])

            if beta_ave>eeg_high_th: 
                if curve_train_cnt<num_train_dateset:
                    curve_focus_train = np.append(curve_focus_train,int(i))
                    curve_train_cnt += 1
                elif curve_train_cnt<num_train_dateset+30:
                    curve_focus_test = np.append(curve_focus_test,int(i))
                    curve_train_cnt += 1
            elif beta_ave<eeg_low_th:
                curve_drow=np.append(curve_drow,int(i))

    print("# of training dataset    :",len(curve_focus_train))
    print("# of test dataset (focus):",len(curve_focus_test))
    print("# of test dataset (drow) :",len(curve_drow))

    return curve_focus_train, curve_focus_test, curve_drow, in_curve_flg, in_curve_cnt, curve_cnt


def draw_fig_fft(time_str, yawrate_i, yawrate_a, dt, i, N_curve, Amp_i, Amp_a):
    axes[0].plot(time_str[i-int(3/dt):i-int(3/dt)+N_curve], 
                 yawrate_i[i-int(3/dt):i-int(3/dt)+N_curve], 
                 label="ideal")
    axes[0].plot(time_str[i-int(3/dt):i-int(3/dt)+N_curve], 
                 yawrate_a[i-int(3/dt):i-int(3/dt)+N_curve], 
                 label="actual(max=%.4f)"%max(abs(yawrate_a[i:i+int(fs*10)])))
    axes[0].set_xlabel("time") 
    axes[0].set_ylabel("yaw rate")
    axes[1].plot(freq[:N_curve//2], Amp_i[:N_curve//2], label="ideal")
    axes[1].plot(freq[:N_curve//2], Amp_a[:N_curve//2], label="actual")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlim(0,1/dt/2)

def filtering_process(filt_type, yawrate_a, yawrate_i, idx_start, idx_end, freq_th, fs):
    if filt_type == "LPF":
        yawrate_a_filt_signed = butter_lowpass_filter(yawrate_a[idx_start:idx_end], freq_th, fs, order=4)
        yawrate_i_filt_signed = butter_lowpass_filter(yawrate_i[idx_start:idx_end], freq_th, fs, order=4)
    elif filt_type == "MA":
        n_conv = 2; b = np.ones(n_conv)/n_conv # >> [1/n_size, 1/n_size, ..., 1/n_size]
        yawrate_a_filt_signed = np.convolve(yawrate_a[idx_start-int(3/dt):idx_end-int(3/dt)], b, mode="same")
    return yawrate_a_filt_signed, yawrate_i_filt_signed

def draw_fig_fft_filt(time_str, time_eeg, yawrate_a_filt, yawrate_i_filt,\
                      idx_start, idx_end, freq, Amp_i_filt, Amp_a_filt, dt, i, N_curve):
    axes[0].plot(time_str[idx_start:idx_end], yawrate_a_filt, label="actual_filt")
    axes[0].plot(time_str[idx_start:idx_end], yawrate_i_filt, label="ideal_filt")
    axes[2].plot(freq[:N_curve//2], Amp_i_filt[:N_curve//2], label="ideal_filt") 
    axes[2].plot(freq[:N_curve//2], Amp_a_filt[:N_curve//2], label="actual_filt")
    
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlim(0,1/dt/2)
    
    idx_start_eeg = int(255.5*(1.0+time_str[i-int(3/dt)]))
    idx_end_eeg   = int(255.5*(1.0+time_str[i-int(3/dt)+N_curve]))
    
    axes[3].plot(time_eeg[   idx_start_eeg:idx_end_eeg], 
                 alpha_local[idx_start_eeg:idx_end_eeg], 
                 label="alpha") 
    axes[3].plot(time_eeg[  idx_start_eeg:idx_end_eeg], 
                 beta_local[idx_start_eeg:idx_end_eeg], 
                 label="beta") 
    axes[3].plot(time_eeg[        idx_start_eeg:idx_end_eeg], 
                 beta_alpha_local[idx_start_eeg:idx_end_eeg], 
                 label="beta/alpha") 
    
    axes[3].set_xlabel("time")
    axes[3].set_ylabel("Amplitude")
    
    axes[0].legend(loc='upper right')
    for x in range(1,4): axes[x].legend() 
    begin_time = str(int(1000*time_str[i-int(3/dt)]))
    plt.savefig("fig/fft_test_focus_"+begin_time+".png", format="png", dpi=1600)
    for x in range(4): axes[x].cla() 

def draw_pzmap(
               t_train_pzmap,
               r_train_pzmap,
               y_train_pzmap,
               yH_pzmap,
               beta_alpha_ave_pzmap,
               a1_pzmap,
               a2_pzmap,
               a3_pzmap,
               a4_pzmap,
               b1_pzmap,
               b2_pzmap,
               b3_pzmap,
               b4_pzmap
               ):
    axes[0].plot(t_train_pzmap, r_train_pzmap, label="ideal_filt")
    axes[0].plot(t_train_pzmap, y_train_pzmap, label="actual_filt")
    axes[0].plot(t_train_pzmap, yH_pzmap, label="predicted output")
    axes[0].text(
                 min(t_train_pzmap),
                 0,
                 str(abs(sum(yH_pzmap-y_train_pzmap)))+"beta_al"+str(beta_alpha_ave_pzmap)
                 )
    axes[1].plot(t_train[j],a1_pzmap)
    axes[1].plot(t_train[j],a2_pzmap)
    axes[1].plot(t_train[j],a3_pzmap)
    axes[1].plot(t_train[j],a4_pzmap)
    axes[1].plot(t_train[j],b1_pzmap)
    axes[1].plot(t_train[j],b2_pzmap)
    axes[1].plot(t_train[j],b3_pzmap)
    axes[1].plot(t_train[j],b4_pzmap)
    axes[0].legend()
    #axes[1].legend()
    plt.savefig(
                "./fig/rls_fc_\
                _train.png",\
                format="png",\
                dpi=1600
                )
    plt.show()
    axes[0].cla()
    axes[1].cla()
    #plt.close()

def save_dataset_csv(time_curve_local, r_input_local, yawrate_a_filt_local, yH_local, curve_cnt): 
    csv_df = pd.DataFrame(
                    data={'time': pd.Series(time_curve_local),
                          'ideal_filt': pd.Series(r_input_local),
                          'actual_filt': pd.Series(yawrate_a_filt_local),
                          'actual_filt_pred': pd.Series(yH_local)
                          })     
                                       
    csv_path = "../result/csv/rls/dataset_"+str(datetime.datetime.utcnow().date())+".csv"
    header_flg = True if curve_cnt == 1 else False
    csv_df.to_csv(csv_path,mode='a', header=header_flg)#False)#True) 

    dataset_show = False
    if dataset_show == True:
        plt.plot(r_input_local)
        plt.plot(yawrate_a_filt_local)
        plt.plot(yH_local)
        plt.show()

def save_rmse_csv(idx_start,y_train,yH,freq_th,eeg_high_th_str,curve_cnt,time_str,drowsiness_flg,input_omega,step):
    csv_df = pd.DataFrame(
                    data={'train_start': pd.Series(int(1000*time_str[idx_start])), 
                          #'rmse': pd.Series(np.nanmean(np.abs(y_train-yH))),
                          'rmse': pd.Series(max(np.abs(y_train-yH))),
                          'drowsiness': pd.Series(drowsiness_flg)
                          })     
                                       
    csv_path = "../result/csv/rls/rmse_" + str(freq_th) + "_EegTh_" + eeg_high_th_str + "_input_"\
               + input_omega + "_step_" + str(step) + "_"\
               + str(datetime.datetime.utcnow().date()) + ".csv"
    
    header_flg = True if curve_cnt == 1 else False
    csv_df.to_csv(csv_path,mode='a', header=header_flg)#False)#True) 
    print(csv_path," has been created.")

def trim_mean(alpha_local, beta_local, beta_alpha_local, time_str, idx_start, idx_end):
    alpha_ave      = np.nanmean(alpha_local[int(255.5*(1.0+time_str[idx_start])):int(255.5*(1.0+time_str[idx_end]))])
    beta_ave       = np.nanmean(beta_local[int(255.5*(1.0+time_str[idx_start])):int(255.5*(1.0+time_str[idx_end]))])
    beta_alpha_ave = np.nanmean(beta_alpha_local[int(255.5*(1.0+time_str[idx_start])):int(255.5*(1.0+time_str[idx_end]))])
    return alpha_ave,beta_ave,beta_alpha_ave

def append_mean_trim(alpha_ave,beta_ave,beta_alpha_ave,
                     alpha_local,beta_local,beta_alpha_local,
                     time_str,idx_start,idx_end):              
    alpha_ave      = np.append(alpha_ave,np.nanmean(alpha_local[int(255.5*(1.0+time_str[idx_start])):\
                                                                int(255.5*(1.0+time_str[idx_end]))]))
    beta_ave       = np.append(beta_ave,np.nanmean(beta_local[int(255.5*(1.0+time_str[idx_start])):\
                                                              int(255.5*(1.0+time_str[idx_end]))]))
    beta_alpha_ave = np.append(beta_alpha_ave,np.nanmean(beta_alpha_local[int(255.5*(1.0+time_str[idx_start])):\
                                                                          int(255.5*(1.0+time_str[idx_end]))]))
    return alpha_ave,beta_ave,beta_alpha_ave

def trim_idx(time_str,yawrate_i,yawrate_a,idx_start,idx_end):
    time_curve      =  time_str[idx_start:idx_end]
    yawrate_i_curve = yawrate_i[idx_start:idx_end]
    yawrate_a_curve = yawrate_a[idx_start:idx_end]
    return time_curve, yawrate_i_curve, yawrate_a_curve

