# https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html

import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tfest
import control as ct
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from matplotlib.mlab import psd, csd

from tqdm import tqdm_notebook as tqdm

# class in other files
from module.preprocess import Preprocess
from module.dataset import Datasets
from module.train import PredictSimpleFormulaNet, Train 

def calculate_aic_and_bic(mse: np.float64, degree: int, data_num: int):
    a = data_num * np.log(mse)
    return a + 2 * degree, a + degree * np.log(data_num)

if __name__ == '__main__':
    start = time.time()

    # load dataset
    dataset = Datasets()
    df_filt, curv_time = dataset.load_veh(output_type="curve", filt=True)
    #df_eeg, curv_time_drow = dataset.load_eeg(curv_time = curv_time)

    print("cnt max = ",df_filt["curv_cnt"].max())

    df_aic,df_bic = pd.DataFrame(),pd.DataFrame()
    for cnt in range(df_filt["curv_cnt"].max()):
        nz_max = 10

        u = df_filt[df_filt["curv_cnt"]==cnt]["yaw_rate_i"].tolist()
        y = df_filt[df_filt["curv_cnt"]==cnt]["yaw_rate_a"].tolist()
        t = df_filt[df_filt["curv_cnt"]==cnt]["time"].values
        d = df_filt[df_filt["curv_cnt"]==cnt]["time"].diff().mean()
    
        aic_all,bic_all = [],[]
        te = tfest.tfest(u, y)
        for method in ["h1","h2","fft"]:
            for nz in range(nz_max):
                te.estimate(nz+1, nz, method=method, time=t[-1]-t[0])
                sys = te.get_transfer_function()
                G = ct.tf(sys.poles, sys.zeros, d)
                T, yout = ct.forced_response(G, T=np.linspace(t[0],t[-1],len(u)), U=u)
                mse = mean_squared_error(y, yout)
                aic, bic = calculate_aic_and_bic(mse, 2*nz+1, len(u))
                aic_all.append(aic)
                bic_all.append(bic)

        df_aic = pd.concat([df_aic,pd.DataFrame(aic_all)], axis=1)
        df_bic = pd.concat([df_bic,pd.DataFrame(bic_all)], axis=1)

    df_aic = df_aic.T.set_axis(["aic h1(0)", "aic h1(1)", "aic h1(2)", "aic h1(3)", "aic h1(4)", 
                                "aic h1(5)", "aic h1(6)", "aic h1(7)", "aic h1(8)", "aic h1(9)",
                                "aic h2(0)", "aic h2(1)", "aic h2(2)", "aic h2(3)", "aic h2(4)",
                                "aic h2(5)", "aic h2(6)", "aic h2(7)", "aic h2(8)", "aic h2(9)",
                                "aic fft(0)","aic fft(1)","aic fft(2)","aic fft(3)","aic fft(4)",
                                "aic fft(5)","aic fft(6)","aic fft(7)","aic fft(8)","aic fft(9)",
                                ], axis=1)
    df_bic = df_bic.T.set_axis(["bic h1(0)", "bic h1(1)", "bic h1(2)", "bic h1(3)", "bic h1(4)",
                                "bic h1(5)", "bic h1(6)", "bic h1(7)", "bic h1(8)", "bic h1(9)",
                                "bic h2(0)", "bic h2(1)", "bic h2(2)", "bic h2(3)", "bic h2(4)",
                                "bic h2(5)", "bic h2(6)", "bic h2(7)", "bic h2(8)", "bic h2(9)",
                                "bic fft(0)","bic fft(1)","bic fft(2)","bic fft(3)","bic fft(4)",
                                "bic fft(5)","bic fft(6)","bic fft(7)","bic fft(8)","bic fft(9)",
                                ], axis=1)

    df_aic.to_csv("../result/csv/aic/aic.csv")
    df_bic.to_csv("../result/csv/aic/bic.csv")

    df_aic.filter(like='h1', axis=1).mean().plot(label="aic h1")
    df_aic.filter(like='h2', axis=1).mean().plot(label="aic h2")
    df_aic.filter(like='fft',axis=1).mean().plot(label="aic fft")
    df_bic.filter(like='h1', axis=1).mean().plot(label="bic h1")
    df_bic.filter(like='h2', axis=1).mean().plot(label="bic h2")
    df_bic.filter(like='fft',axis=1).mean().plot(label="bic fft")

    plt.legend()
    plt.savefig("../result/visualisation/aic/aic_bic.pdf")
    plt.show()

    print("elapse time", time.time()-start)

