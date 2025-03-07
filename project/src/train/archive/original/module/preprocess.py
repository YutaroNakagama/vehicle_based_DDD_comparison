
import pandas as pd
from scipy.signal import butter,filtfilt

class Preprocess():
    def __init__(self):
        # yawrate min/max threshold for outlier removal 
        self.min = -5
        self.max =  5

        # low pass filter config
        self.cutoff = 2         # desired cutoff freq of filt [Hz]
        self.fs = 30            # sample reate [Hz]
        self.order = 8
        self.nyq = self.fs / 2

        # yawrate threshold for curve detection
        self.curve_th = 0.1

    def outlier_min_max(self, df, target):
        ''' this func will make data to None in case the data is out of range
        '''
        for i in target:
            col = df[i]
            col[col < self.min] = None
            col[col > self.max] = None
        return df

    def butter_lowpass_filter(self, df, target):
        for i in target:
            normal_cutoff = 0.20 #self.cutoff / self.nyq
            # Get the filter coefficients 
            b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
            df[i] = pd.DataFrame(filtfilt(b, a, df[i])).set_index(df[i].index)
        return df

    def get_curve_point(self, df):
        ''' this func detect curve point and returns the index by checking:
        1. the absolute value of 1 step ahead is larger than curve_th.
        2. the current value equals 0. 
        3. the in-curve flag is not True.
        '''
        ret = []
        cnt = 0
        flg = False
        for i in range(len(df)-1):
            # if the conditions of 1-3 are satisfied
            if abs(df["yaw_rate_i"].iloc[i+1]) > self.curve_th and \
               df["yaw_rate_i"].iloc[i] == 0 and not flg:
                ret.append(i)
                flg = True 
            # if the in-curve flag is True
            if flg:
                if cnt < 150: # cnt is incremented til 150 
                    cnt += 1 
                else: # once cnt become 150, cnt initialized & make flg False 
                    cnt = 0 
                    flg = False
        return ret
