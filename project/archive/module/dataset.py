import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import antropy as ant
import neurokit2 as nk
import collections
import librosa

from tqdm import tqdm
from darts import TimeSeries
from sklearn import preprocessing
from scipy import stats
from scipy.fftpack import fft, ifft, fftfreq

from module.preprocess import Preprocess

class Datasets(): 
    def __init__(self, data_id=0):
        # load dataset path
        self.dataset_dir  = "../../dataset/yn/"
        type = "full" # full or test
        #with open(self.dataset_dir + 'dataset_veh_'+type+'.txt', 'r') as f_veh:
        #    self.dataset_veh = f_veh.readlines()
        #with open(self.dataset_dir + 'dataset_eeg_'+type+'.txt', 'r') as f_eeg:
        #    self.dataset_eeg = f_eeg.readlines()
        self.ret_veh     = pd.DataFrame()
        self.ret_veh_eeg = pd.DataFrame()
        self.ret_beta    = pd.DataFrame()
        self.drow_flg    = pd.DataFrame()
        self.data_idx    = int(data_id) 

    def load_veh_yn(self, output_type="full", filt=False):
        ''' filt flg is only applicable for "curve" type
        '''
        data_column = ["yaw_rate_i","yaw_rate_a"]
        curv_time = []
        curv_cnt = 0
        #for j in range(len(self.dataset_veh)):
        for j in range(self.data_idx,self.data_idx+1):
            print("reading",self.dataset_dir + self.dataset_veh[j].rstrip('\n'))
            df = pd.read_pickle(self.dataset_dir + self.dataset_veh[j].rstrip('\n'))
            df = df[["time","yaw_rate_i","yaw_rate_a"]]
            df = Preprocess().outlier_min_max(df,target=data_column).drop(index=0)
            if output_type == "curve":
                curv_pt = Preprocess().get_curve_point(df)
                for i in curv_pt:
                    df_curv = df.iloc[i - 25 * 5 : i + 25 * 10].ffill()
                    if abs(df_curv["yaw_rate_i"].sum()) > 80:
                        if filt: 
                            df_curv = Preprocess().butter_lowpass_filter(
                                df_curv,
                                target=data_column
                                )
                        df_curv["curv_cnt"] = curv_cnt
                        curv_cnt += 1
                        self.ret_veh = pd.concat([self.ret_veh, df_curv])
                        curv_time.append(df["time"].iloc[i])
            elif output_type == "full":
                self.ret_veh = pd.concat([self.ret_veh, df.copy()])
                curv_time = None
        return self.ret_veh.fillna(0),curv_time 

    def load_eeg_yn(self, data_type="full", curv_time=None):
        ''' curv_time is value of df["time"] at detected point
        '''
        eeg_select = []
        with open('./module/eeg_signals.txt','r') as f:
            signals = f.readlines()
        signals = [s.rstrip('\n') for s in signals]
        signals = [s for s in signals if not s.startswith('#')]
        df_eeg = pd.DataFrame()
        #for j in range(len(self.dataset_veh)):
        for j in range(self.data_idx,self.data_idx+1):
            print("reading",self.dataset_dir + self.dataset_eeg[j].rstrip('\n'))
            df = pd.read_pickle(self.dataset_dir + self.dataset_eeg[j].rstrip('\n'))
            df = df[signals]  
            df_eeg = df_eeg.append(df)
        self.ret_eeg = self.ret_veh.fillna(0).copy()
        if data_type == "curve":
            for s in signals:
                curv_cnt = 0
                for t in tqdm(curv_time): 
                    # curve area is from "curv_time-5" to "curv_time+10" 
                    df_curv = df[(df["time"] > (t - 5)) & (df["time"] < (t + 10))]
                    self.ret_eeg.loc[self.ret_veh["curv_cnt"] == curv_cnt, s] = df_curv[s].mean()
                    curv_cnt += 1
        elif data_type == "full":
            self.ret_eeg = df_eeg[["time","beta_af7","beta_af8"]].copy()
        return self.ret_eeg

    def load_veh_eeg_yn(self, data_type="full", filt=False):
        ''' filt flg is only applicable for "curve" type
        '''
        data_column = ["yaw_rate_i","yaw_rate_a"]
        curv_time,eeg_select = [],[]
        curv_cnt = 0
        with open('./module/eeg_signals.txt','r') as f:
            signals = f.readlines()
        signals = [s.rstrip('\n') for s in signals]
        signals = [s for s in signals if not s.startswith('#')]
        df_veh, df_eeg, df_ret = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for j in range(len(self.dataset_veh)):
            df = pd.read_pickle(self.dataset_dir + self.dataset_veh[j].rstrip('\n'))
            df = df[["time","steering_angle"]]
            df = Preprocess().outlier_min_max(df,target=data_column).drop(index=0)
            df_veh = df.copy()
            df = pd.read_pickle(self.dataset_dir + self.dataset_eeg[j].rstrip('\n'))
            df = df[signals]  
            df_eeg = df.copy()
            df_eeg["beta_ave"] = (df_eeg["beta_af7"] + df_eeg["beta_af8"]) / 2 
            df_veh_eeg = pd.merge(df_veh, df_eeg, on="time", how="outer")
            df_veh_eeg = df_veh_eeg.sort_values('time').drop_duplicates(keep='first').\
                         reset_index(drop=True).drop(columns=["beta_af7","beta_af8"])
            df_veh_eeg['yaw_rate_a'].interpolate(method="cubic", inplace = True)
            df_veh_eeg['yaw_rate_i'].interpolate(method="cubic", inplace = True)
            df_veh_eeg['beta_ave'].interpolate(method="cubic", inplace = True)
            df_veh_eeg = df_veh_eeg.dropna().drop_duplicates(subset=["time"])
            df_veh_eeg = df_veh_eeg.rolling(10).mean().dropna().loc[::3].reset_index(drop=True)
            df_veh_eeg["time"] = df_veh_eeg["time"].round(2)
            df_veh_eeg['t'] = pd.DataFrame(np.linspace(df_veh_eeg["time"].iloc[0],\
                                           df_veh_eeg["time"].iloc[-1],\
                                           df_veh_eeg.shape[0]))
            df_veh_eeg["dt"] = df_veh_eeg["t"].diff()
            df_veh_eeg = df_veh_eeg.dropna()
            df_veh_eeg['yaw_rate_a'] = df_veh_eeg['yaw_rate_a']/3
            df_veh_eeg['yaw_rate_i'] = df_veh_eeg['yaw_rate_i']/3
            df_ret = df_ret.append(df_veh_eeg)
        self.ret_veh_eeg  = df_ret[["yaw_rate_i","yaw_rate_a","beta_ave"]].values.astype('float32') 
        return self.ret_veh_eeg

    def load_veh_eeg_ppb(self, data_type="full", filt=False):
        df_ret = pd.DataFrame()
        alpha_columns = []
        for i in range(1,33):
            alpha_columns += [''.join('NEBAND_alpha-sensor-' + str(i))]
        beta_columns = []
        for i in range(1,33):
            beta_columns += [''.join('NEBAND_beta-sensor-' + str(i))]
        person_class = []
        for i in range(2,3):#(2,42):
            person_class += [''.join('P' + str(i).zfill(2))]
        category_class = ["AD"]#["AD","SAD","DD","FD","HD","ND","SD"]
        for person in tqdm(person_class):
            for category in category_class:
                fp_ppb = "../../dataset/PPB-Emo/" 
                fp_veh = fp_ppb + "Driving_behavioural_data/" + person +\
                         "/PPB_Emo_dataset@DBD-30s-" + person + "-" +\
                         category + ".csv"
                fp_eeg = fp_ppb + "Physiological_data/" + person +\
                         "/PPB_Emo_dataset@EEG-30s-"+ person+ "-"+\
                         category+ ".csv"
                is_file = os.path.isfile(fp_veh)
                if is_file:
                    df_veh = pd.read_csv(fp_veh) # 6sample/1s 
                    df_eeg = pd.read_csv(fp_eeg) #  
                    df_veh_str = df_veh[["Steering wheel position"]]
                    df_veh_str_sel = df_veh_str[::6].reset_index(drop=True)
                    df_veh_str_sel = (df_veh_str_sel - df_veh_str_sel.min()) /\
                                     (df_veh_str_sel.max() - df_veh_str_sel.min()) 
                    df_eeg_beta = df_eeg[beta_columns]
                    df_eeg_beta["beta_ave"] = df_eeg_beta.mean(axis=1)
                    df_eeg_beta_int = df_eeg_beta["beta_ave"].dropna().reset_index(drop=True)
                    df_eeg_alpha = df_eeg[alpha_columns]
                    df_eeg_alpha["alpha_ave"] = df_eeg_alpha.mean(axis=1)
                    df_eeg_alpha_int = df_eeg_alpha["alpha_ave"].dropna().reset_index(drop=True)
                    df_ret = df_ret.append(pd.concat([df_veh_str_sel, 
                                                      df_eeg_beta_int/df_eeg_alpha_int
                                                      ], axis=1).dropna())
        self.ret_veh_eeg = df_ret.values.astype('float32') 
        return self.ret_veh_eeg, df_ret

    def load_veh_eeg_yn_feat_Aref(self, data_type="full", filt=False):
        ''' filt flg is only applicable for "curve" type
        '''
        data_column = ["yaw_rate_i","yaw_rate_a"]
        curv_time,eeg_select = [],[]
        curv_cnt = 0
        with open('./module/eeg_signals.txt','r') as f:
            signals = f.readlines()
        signals = [s.rstrip('\n') for s in signals]
        signals = [s for s in signals if not s.startswith('#')]
        df_veh, df_eeg, df_ret = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for j in range(len(self.dataset_veh)):
            df_feat = pd.DataFrame()
            # ------------------ vehicle information --------------------- #
            df = pd.read_pickle(self.dataset_dir + self.dataset_veh[j].rstrip('\n'))
            df = df[["time","steering_angle"]]
            window_t = 3 # second
            freq = 30 # Hz
            window_len = int(window_t * freq) # window time * freq
            df["road_curve"] = df["steering_angle"].rolling(window_len).mean().shift(-window_len//2) 
            df["str_no_road"] = df["steering_angle"] - df["road_curve"]
            df["str_vel"] = df["steering_angle"].diff() / df["time"].diff()
            df["time_veh"] = df["time"].rolling(int(window_len)).mean()
            df_feat["time_veh"] = df["time_veh"].iloc[::window_len//2].reset_index(drop=True)
            # energy of steering angle 
            df["power_str_no_road"] = df["str_no_road"].abs() ** 2
            df["engy_str_no_road"] = df["power_str_no_road"].rolling(int(window_len)).mean().shift(-window_len//2) 
            df_feat["engy_str_no_road"] =\
                preprocessing.minmax_scale(df["engy_str_no_road"].iloc[::window_len//2].reset_index(drop=True))
            # ZCR of steering velocity
            df["str_vel_dir"] = df["str_vel"].diff() / df["str_vel"].diff().abs() # calc dir of str vel 
            df["str_vel_dir"] = df["str_vel_dir"].fillna(0) # replace nan to 0
            df["str_vel_dir_change"] = df["str_vel_dir"].diff().abs()/2
            df["str_vel_dir_change"] = df["str_vel_dir_change"].round()
            df["str_vel_dir_zcr"] = df["str_vel_dir_change"].rolling(int(window_len)).sum().shift(-window_len//2)/window_t
            df_feat["str_vel_dir_zcr"] =\
                    preprocessing.minmax_scale(df["str_vel_dir_zcr"].iloc[::window_len//2].reset_index(drop=True))
            # Kurtosis of steering velocity  
            df_feat["kurtosis_str_vel"] = df_feat["str_vel_dir_zcr"].copy()
            x = 0
            #for i in tqdm(range(2500,len(df["str_vel"])-window_len,window_len//2)):
            for i in tqdm(range(0,len(df["str_vel"])-window_len,window_len//2)):
                df_feat["kurtosis_str_vel"].iloc[x] = stats.kurtosis(df["str_vel"].iloc[i:i+window_len-1])
                df.iloc[i:i+window_len-1,:].plot(x="time", y=["str_vel"])
                df["str_vel"].iloc[i:i+window_len-1].hist(bins=15)
                x += 1
            # sample entropy
            df_feat["samen_str_no_road"] = df_feat["str_vel_dir_zcr"].copy()
            df_feat["samen_str_vel"]     = df_feat["str_vel_dir_zcr"].copy()
            x = 0
            for i in tqdm(range(0,min(len(df["str_no_road"]),len(df["str_vel"]))-window_len,window_len//2)):
                df_feat["samen_str_no_road"].iloc[x] = ant.sample_entropy(df["str_no_road"].iloc[i:i+window_len-1])
                df_feat["samen_str_vel"].iloc[x] = ant.sample_entropy(df["str_vel"].iloc[i:i+window_len-1])
                x += 1
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
            df_feat = df_feat.dropna(how='any') 
            df_feat["kurtosis_str_vel"] = preprocessing.minmax_scale(df_feat["kurtosis_str_vel"]) 
            df_feat["samen_str_no_road"] = preprocessing.minmax_scale(df_feat["samen_str_no_road"]) 
            df_feat["samen_str_vel"] = preprocessing.minmax_scale(df_feat["samen_str_vel"]) 
            # ------------------ EEG information --------------------- #
            df = pd.read_pickle(self.dataset_dir + self.dataset_eeg[j].rstrip('\n'))
            df = df[signals]  
            df_eeg = df.shift(-140*3).copy()
            df_eeg["beta_ave"] = (df_eeg["beta_af7"] + df_eeg["beta_af8"]) / 2 
            df_eeg["alpha_ave"] = (df_eeg["alpha_af7"] + df_eeg["alpha_af8"]) / 2 
            df_eeg["time_eeg"] = df_eeg["time"].rolling(281).mean()
            df_feat["time_eeg"] = df_eeg["time_eeg"].iloc[::281//2].reset_index(drop=True)
            df_eeg["beta_ave_win"] = df_eeg["beta_ave"].rolling(281).mean().shift(-281//2).interpolate()
            df_feat["beta_ave_win"] = df_eeg["beta_ave_win"].iloc[::281//2].reset_index(drop=True)
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
            df_feat = df_feat.dropna(how='any') 
            df_ret = df_ret.append(df_feat)

        thld_low = 0.12
        thld_high = 0.7
        df_ret = df_ret[(df_ret["beta_ave_win"] < thld_low) | (df_ret["beta_ave_win"]> thld_high)]
        df_ret["drowsy"] = df_ret["beta_ave_win"] < thld_low 
        self.ret_veh_eeg  = df_ret.reset_index(drop=True)
        return self.ret_veh_eeg

    def load_Aygun2024(self):
        ''' Dataset details: Aygun2024 doi:10.7910/DVN/HMZ5RG
        '''
        col_audio = [
                "subj_num",
                "trial",
                "audio_id",
                "audio_txt",
                "audio_start",
                "audio_end",
                "SOA",
                "FTO",
                "audio_dur"
                ]

        col_physio = [
            "time stamps",      # [s]
            "Blood pressure",   # [mmHg] 
            "skin conductance", # [muS] 
            "respiration",      # in z scores (x –xmean)/std x
            "O2"                # [%] 
            ]
        
        col_simlsl = [
            "row LSL time",         # 1
            "Sim time lsl",         # 2 from the lsl data file 
            "Sim time Sim",         # 3 from the Sim data file-not the lsl file 
            "Brake_Angle",          # 4 [deg]
            "Brake_Event_Time",     # 5 [s]
            "Audio_File_Num",       # 6 (ND)
            "Audio_Time",           # 7 [s]
            "JS_time",              # 8 [s]
            "DRT_Onset",            # 9 [1=light on]
            "DRT_Resp",             # 10 [1=button press]
            "DRT_Button_Val",       # 11 [v] 
            "SOA",                  # 12 (ND)
            "DRT_Onset_Time",       # 13 [s]
            "DRT_Resp_Time",        # 14 [s]
            "DRT_Num",              # 15 [ND]
            "Brake_Lights",         # 16 [1=on]
            "Eval",                 # 17 [1=keep]
            "Route_Position",       # 18 [m]
            "Long_Accel",           # 19 [m/s/s]
            "Lat_Accel",            # 20 [m/s/s]
            "Throttle_Pos",         # 21 [deg]
            "Brake_Force",          # 22 (ND)
            "Gear",                 # 23 (ND)
            "Heading_Error",        # 24 [deg]
            "Headway_Dist",         # 25 [m]
            "Headway_Time",         # 26 [s]
            "Lane_Num",             # 27 (ND)
            "Lane_Offset",          # 28 [m]
            "Road_Offset",          # 29 [m]
            "Steering_Wheel_Pos",   # 30 [rad]
            "Velocity"              # 31 [m/s]
            ]
        
        col_eeg = [
            "LSL time stamps",  # [s]
            "FC5",              # [nV]
            "FC1",              # [nV]
            "CP5",              # [nV]
            "CP1",              # [nV]
            "CP2",              # [nV]
            "CP6",              # [nV]
            "FC2",              # [nV]
            "FC6"               # [nV]
            ]

        col_event = ["event_start","event_end"]

        column = ["time","delta","theta","alpha","beta","gamma"]
        df_temp = pd.DataFrame(columns=column)

        sens_pos = ['FC1','FC2','FC5','FC6','CP1','CP2','CP5','CP6']
        subj_num_all = []

        for i in range(113,210): # 113 - 210 
            subj_num_all += [''.join('S{0:04}'.format(i))]

        col_FC1 = ["time_FC1","delta_FC1","theta_FC1","alpha_FC1","beta_FC1","gamma_FC1"]
        col_FC2 = ["time_FC2","delta_FC2","theta_FC2","alpha_FC2","beta_FC2","gamma_FC2"]
        col_FC5 = ["time_FC5","delta_FC5","theta_FC5","alpha_FC5","beta_FC5","gamma_FC5"]
        col_FC6 = ["time_FC6","delta_FC6","theta_FC6","alpha_FC6","beta_FC6","gamma_FC6"]
        col_CP1 = ["time_CP1","delta_CP1","theta_CP1","alpha_CP1","beta_CP1","gamma_CP1"]
        col_CP2 = ["time_CP2","delta_CP2","theta_CP2","alpha_CP2","beta_CP2","gamma_CP2"]
        col_CP5 = ["time_CP5","delta_CP5","theta_CP5","alpha_CP5","beta_CP5","gamma_CP5"]
        col_CP6 = ["time_CP6","delta_CP6","theta_CP6","alpha_CP6","beta_CP6","gamma_CP6"]

        eeg_psd_FC1 = pd.DataFrame(columns=col_FC1)
        eeg_psd_FC2 = pd.DataFrame(columns=col_FC2)
        eeg_psd_FC5 = pd.DataFrame(columns=col_FC5)
        eeg_psd_FC6 = pd.DataFrame(columns=col_FC6)
        eeg_psd_CP1 = pd.DataFrame(columns=col_CP1)
        eeg_psd_CP2 = pd.DataFrame(columns=col_CP2)
        eeg_psd_CP5 = pd.DataFrame(columns=col_CP5)
        eeg_psd_CP6 = pd.DataFrame(columns=col_CP6)

        for subj in subj_num_all:
            Aygun_fp = "../../../data/external/Aygun2024/physio/"
            subj_num = subj

            for n in range(2):
                print(subj+"_"+str(n+1))
                fp_aud = Aygun_fp + subj_num + "/AudioEvents_" + subj_num+ "_"+str(n+1)+".mat"
                fp_veh = Aygun_fp + subj_num + "/SIMlsl_" + subj_num+ "_"+str(n+1)+".mat"
                fp_eeg = Aygun_fp + subj_num + "/EEG_" + subj_num+ "_"+str(n+1)+".mat"
                fp_evt = Aygun_fp + subj_num + "/EventTimes_" + subj_num+ "_"+str(n+1)+".mat"
    
                is_file_aud = os.path.isfile(fp_aud)
                is_file_veh = os.path.isfile(fp_veh)
                is_file_eeg = os.path.isfile(fp_eeg)
                is_file_evt = os.path.isfile(fp_evt)
    
                if is_file_aud and is_file_veh and is_file_eeg and is_file_evt:
    
                    # reading from .mat file. details can be found in Data description.txt
                    mat_audio  = scipy.io.loadmat(fp_aud)
                    mat_simlsl = scipy.io.loadmat(fp_veh)
                    mat_eeg    = scipy.io.loadmat(fp_eeg)
                    mat_event  = scipy.io.loadmat(fp_evt)
            
                    # make dataframe from .mat file
                    #df_audio  = pd.DataFrame(mat_audio['AudioEvents'], columns=col_audio)
                    df_simlsl = pd.DataFrame(mat_simlsl['SIM_lsl'], index=col_simlsl).T
                    #df_eeg    = pd.DataFrame(mat_eeg['rawEEG'], index=col_eeg).T
                    df_event  = pd.DataFrame(mat_event['EventTime'], columns=col_event)
                    #df_sigon  = pd.DataFrame(mat_event['DRT_SignalOn'], index=["drt_start"]).T
                    #df_resp   = pd.DataFrame(mat_event['DRT_response'], index=["drt_end"]).T
                    #df_drt    = pd.concat([df_sigon,df_resp],axis=1)
    
                    # ---------------------------------------------------------- #
                    #                     audio analysis                         #
                    # ---------------------------------------------------------- #
                    audio_flg = False
                    if audio_flg:
                        df_audio["zeros"] = 0 
                        df_audio["ones"]  = 1 
                        df_audio['evns'] = list(range(0,len(df_audio)*4,4)) 
                        df_audio['odds'] = list(range(1,len(df_audio)*4,4)) 
                        df_audio['evne'] = list(range(2,len(df_audio)*4,4)) 
                        df_audio['odde'] = list(range(3,len(df_audio)*4,4)) 
                        
                        df_audio_start = pd.concat([
                            df_audio[["audio_start","zeros",'evns']].rename({'zeros':'audio_event','evns':'i'}, axis='columns'), 
                            df_audio[["audio_start","ones",'odds']].rename({'ones':'audio_event','odds':'i'}, axis='columns') 
                            ]).rename({'audio_start':'audio_t'}, axis='columns')
        
                        df_audio_end = pd.concat([
                            df_audio[["audio_end","ones",'evne']].rename({'ones':'audio_event','evne':'i'}, axis='columns'),   
                            df_audio[["audio_end","zeros",'odde']].rename({'zeros':'audio_event','odde':'i'}, axis='columns')
                            ]).rename({'audio_end':'audio_t'}, axis='columns')
        
                        df_audio_all = pd.concat([df_audio_start, df_audio_end]).sort_values('i').reset_index(drop=True)
        
                        df_audio_bl = pd.DataFrame() 
                        df_audio_bl["time_audio"] = range(int(df_simlsl['row LSL time'].max()))
        
                        audio_list = []
                        flg,idx = 0,0
                        for i in range(int(df_simlsl['row LSL time'].max())):
                            if df_audio_all.audio_t.iloc[idx] <= float(i) and df_audio_all.audio_t.max() >= float(i)-1 :
                                flg = (flg + 1)%2
                                idx = idx + 2 if idx+2 < len(df_audio_all.audio_t) else idx + 1 
                            audio_list.append(flg)
        
                        df_audio_bl["audio_flg"] = audio_list
    
                    # ---------------------------------------------------------- #
                    #                     calclate eeg psd                       #
                    # ---------------------------------------------------------- #
                    eeg_flg = False 
                    if eeg_flg:
                        time_window_eeg = 1 # [sec]
                        sampling_freq_eeg = 500 # Hz
                        for i in tqdm(range(0,len(df_eeg),sampling_freq_eeg*time_window_eeg)):
                            for x in sens_pos: 
                                (f, S) = scipy.signal.periodogram(
                                        df_eeg[x].iloc[i:i+sampling_freq_eeg*time_window_eeg], 
                                        sampling_freq_eeg, 
                                        scaling='density')
                                exec('eeg_psd_{}.loc[len(eeg_psd_{})+1] = [\
                                        len(eeg_psd_{}),\
                                        np.sum(S[ 1:  5]),\
                                        np.sum(S[ 4:  9]),\
                                        np.sum(S[ 8: 14]),\
                                        np.sum(S[13: 31]),\
                                        np.sum(S[30:101])]'.format(x,x,x))
    
                    # ---------------------------------------------------------- #
                    #                       Event time                           #
                    # ---------------------------------------------------------- #
    
                    event_flg = False
                    if event_flg:
                        df_event["zeros"] = 0 
                        df_event["ones"]  = 1 
                        df_event['evns'] = list(range(0,len(df_event)*4,4)) 
                        df_event['odds'] = list(range(1,len(df_event)*4,4)) 
                        df_event['evne'] = list(range(2,len(df_event)*4,4)) 
                        df_event['odde'] = list(range(3,len(df_event)*4,4)) 
                        
                        df_event_start = pd.concat([
                            df_event[["event_start","zeros",'evns']].rename({'zeros':'event_event','evns':'i'}, axis='columns'), 
                            df_event[["event_start","ones",'odds']].rename({'ones':'event_event','odds':'i'}, axis='columns') 
                            ]).rename({'event_start':'event_t'}, axis='columns')
                        df_event_end   = pd.concat([
                            df_event[["event_end","ones",'evne']].rename({'ones':'event_event','evne':'i'}, axis='columns'),   
                            df_event[["event_end","zeros",'odde']].rename({'zeros':'event_event','odde':'i'}, axis='columns')
                            ]).rename({'event_end':'event_t'}, axis='columns')
                        df_event_all = pd.concat([df_event_start, df_event_end]).sort_values('i').reset_index(drop=True)
                        #df_event_all.astype('float32').plot(x="event_t",y="event_event",marker=".")
        
                        df_event_bl = pd.DataFrame() 
                        df_event_bl["time_event"] = range(int(df_simlsl['row LSL time'].max()))
        
                        event_list = []
                        flg,idx = 0,0
                        for i in range(int(df_simlsl['row LSL time'].max())):
                            #print(df_event_all.event_t.iloc[40] <= float(i),df_event_all.event_t.max() >= float(i)-1 )
                            if df_event_all.event_t.iloc[idx] <= float(i) and df_event_all.event_t.max() >= float(i)-1 :
                                flg = (flg + 1)%2
                                idx = idx + 2 if idx+2 < len(df_event_all.event_t) else idx + 1 
                            event_list.append(flg)
        
                        df_event_bl["event_flg"] = event_list
        
                        # ---------------------------------------------------------- #
        
                        df_drt["zeros"] = 0 
                        df_drt["ones"]  = 1 
                        df_drt['evns'] = list(range(0,len(df_drt)*4,4)) 
                        df_drt['odds'] = list(range(1,len(df_drt)*4,4)) 
                        df_drt['evne'] = list(range(2,len(df_drt)*4,4)) 
                        df_drt['odde'] = list(range(3,len(df_drt)*4,4)) 
                        
                        df_drt_start = pd.concat([
                            df_drt[["drt_start","zeros",'evns']].rename({'zeros':'drt_act','evns':'i'}, axis='columns'), 
                            df_drt[["drt_start","ones",'odds']].rename({'ones':'drt_act','odds':'i'}, axis='columns') 
                            ]).rename({'drt_start':'drt_t'}, axis='columns')
                        df_drt_end = pd.concat([
                            df_drt[["drt_end","ones",'evne']].rename({'ones':'drt_act','evne':'i'}, axis='columns'),   
                            df_drt[["drt_end","zeros",'odde']].rename({'zeros':'drt_act','odde':'i'}, axis='columns')
                            ]).rename({'drt_end':'drt_t'}, axis='columns')
                        df_drt_all = pd.concat([df_drt_start, df_drt_end]).sort_values('i').reset_index(drop=True)
        
                        df_drt_bl = pd.DataFrame() 
                        df_drt_bl["time_drt"] = range(int(df_simlsl['row LSL time'].max()))
        
                        drt_list = []
                        flg,idx = 0,0
                        for i in range(int(df_simlsl['row LSL time'].max())):
                            if df_drt_all.drt_t.iloc[idx] <= float(i) and df_drt_all.drt_t.max() >= float(i)-1 :
                                flg = (flg + 1)%2
                                idx = idx + 2 if idx+2 < len(df_drt_all.drt_t) else idx + 1 
                            drt_list.append(flg)
        
                        df_drt_bl["drt_flg"] = drt_list
    
                    # ---------------------------------------------------------- #
                    #                  combine all dataframes                    #
                    # ---------------------------------------------------------- #
    
                    sampling_freq_simlsl = 60  # Hz
                    df_veh = pd.DataFrame()
                    for i in tqdm(range(0, len(df_simlsl), sampling_freq_simlsl)):
                        df_veh = pd.concat([df_veh,df_simlsl.iloc[i].to_frame().T])
                    df_veh      = df_veh.reset_index(drop=True)

                    if eeg_flg:
                        eeg_psd_FC1 = eeg_psd_FC1.reset_index(drop=True)
                        eeg_psd_FC2 = eeg_psd_FC2.reset_index(drop=True)
                        eeg_psd_FC5 = eeg_psd_FC5.reset_index(drop=True)
                        eeg_psd_FC6 = eeg_psd_FC6.reset_index(drop=True)
                        eeg_psd_CP1 = eeg_psd_CP1.reset_index(drop=True)
                        eeg_psd_CP2 = eeg_psd_CP2.reset_index(drop=True)
                        eeg_psd_CP5 = eeg_psd_CP5.reset_index(drop=True)
                        eeg_psd_CP6 = eeg_psd_CP6.reset_index(drop=True)
                        #self.ret_veh_eeg = pd.concat([df_veh,eeg_psd_FC1],axis=1)
                        ret_veh_eeg_temp = pd.concat([
                            df_veh,
                            eeg_psd_FC1,
                            eeg_psd_FC2,
                            eeg_psd_FC5,
                            eeg_psd_FC6,
                            eeg_psd_CP1,
                            eeg_psd_CP2,
                            eeg_psd_CP5,
                            eeg_psd_CP6,
                            ],axis=1).astype('float64').dropna()
        
                        # ---------------------------------------------------------- #
                        #                 eeg average for ground truth               #
                        # ---------------------------------------------------------- #
    
                        rows_FC1 = ["delta_FC1", "theta_FC1", "alpha_FC1", "beta_FC1", "gamma_FC1"]
                        rows_FC2 = ["delta_FC2", "theta_FC2", "alpha_FC2", "beta_FC2", "gamma_FC2"]
                        rows_FC5 = ["delta_FC5", "theta_FC5", "alpha_FC5", "beta_FC5", "gamma_FC5"]
                        rows_FC6 = ["delta_FC6", "theta_FC6", "alpha_FC6", "beta_FC6", "gamma_FC6"]
                        rows_CP1 = ["delta_CP1", "theta_CP1", "alpha_CP1", "beta_CP1", "gamma_CP1"]
                        rows_CP2 = ["delta_CP2", "theta_CP2", "alpha_CP2", "beta_CP2", "gamma_CP2"]
                        rows_CP5 = ["delta_CP5", "theta_CP5", "alpha_CP5", "beta_CP5", "gamma_CP5"]
                        rows_CP6 = ["delta_CP6", "theta_CP6", "alpha_CP6", "beta_CP6", "gamma_CP6"]
                        rows_all = [rows_FC1, rows_FC2, rows_FC5, rows_FC6,rows_CP1, rows_CP2, rows_CP5, rows_CP6] 
                    
                        ret_veh_eeg_temp["delta"] = ret_veh_eeg_temp[[
                            "delta_FC1","delta_FC2","delta_FC5","delta_FC6",
                            "delta_CP1","delta_CP2","delta_CP5","delta_CP6"
                            ]].mean(axis='columns') 
                    
                        ret_veh_eeg_temp["theta"] = ret_veh_eeg_temp[[
                            "theta_FC1","theta_FC2","theta_FC5","theta_FC6",
                            "theta_CP1","theta_CP2","theta_CP5","theta_CP6"
                            ]].mean(axis='columns') 
                    
                        ret_veh_eeg_temp["alpha"] = ret_veh_eeg_temp[[
                            "alpha_FC1","alpha_FC2","alpha_FC5","alpha_FC6",
                            "alpha_CP1","alpha_CP2","alpha_CP5","alpha_CP6"
                            ]].mean(axis='columns') 
                    
                        ret_veh_eeg_temp["beta"] = ret_veh_eeg_temp[[
                            "beta_FC1","beta_FC2","beta_FC5","beta_FC6",
                            "beta_CP1","beta_CP2","beta_CP5","beta_CP6"
                            ]].mean(axis='columns') 
                    
                        ret_veh_eeg_temp["gamma"] = ret_veh_eeg_temp[[
                            "gamma_FC1","gamma_FC2","gamma_FC5","gamma_FC6",
                            "gamma_CP1","gamma_CP2","gamma_CP5","gamma_CP6"
                            ]].mean(axis='columns') 
                    
                        ret_veh_eeg_temp["theta_p_beta"] = \
                                ret_veh_eeg_temp["theta"] / ret_veh_eeg_temp["beta"] 
                        ret_veh_eeg_temp["theta_p_alpha_beta"] = \
                                ret_veh_eeg_temp["theta"] / (ret_veh_eeg_temp["alpha"] + ret_veh_eeg_temp["beta"]) 
                        ret_veh_eeg_temp["theta_alpha_p_beta"] = \
                                (ret_veh_eeg_temp["theta"] + ret_veh_eeg_temp["alpha"]) / ret_veh_eeg_temp["beta"]
                        ret_veh_eeg_temp["alpha_p_beta"] = \
                                ret_veh_eeg_temp["alpha"] / ret_veh_eeg_temp["beta"] 
    
                    # ---------------------------------------------------------- #
                    #                          concate                           #
                    # ---------------------------------------------------------- #
    
                    df_simlsl['subj'] = subj+"_"+str(n+1)
                    #ret_veh_eeg_temp['subj'] = subj+"_"+str(n+1)
                    self.ret_veh     = pd.concat([df_simlsl,        self.ret_veh    ],axis=0)
                    #self.ret_veh_eeg = pd.concat([ret_veh_eeg_temp, self.ret_veh_eeg],axis=0)

        return self.ret_veh_eeg, self.ret_veh


    def make_dataset(self, threshold):
        self.ret_awake = pd.DataFrame()
        self.ret_drow  = pd.DataFrame()
        for i in self.ret_beta[self.ret_beta["beta_ave"] >= threshold].index.tolist(): 
            df_curv = self.ret_veh[self.ret_veh["curv_cnt"]==i]
            self.ret_awake = pd.concat([self.ret_awake, df_curv])
        for i in self.ret_beta[self.ret_beta["beta_ave"] < threshold].index.tolist(): 
            df_curv = self.ret_veh[self.ret_veh["curv_cnt"]==i]
            self.ret_drow = pd.concat([self.ret_drow, df_curv])
        return self.ret_awake,self.ret_drow

    def convert_ts(self,df):
        ''' function for darts library
        '''
        df = df.reset_index(drop=True)
        step=int(df["time"].diff().mean()*1000)
        df["time (ms)"] = pd.DataFrame(np.arange(df.iloc[0]["time"]*1000,
                                       df.iloc[-1]["time"]*1000,  
                                       step=step,
                                       dtype="int"))
        ts = TimeSeries.from_dataframe(df.drop(columns='time'), 
                                       time_col="time (ms)")
        return ts

