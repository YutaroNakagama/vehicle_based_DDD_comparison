# https://pypi.org/project/tfest/
# https://jckantor.github.io/CBE30338/05.03-Creating-Bode-Plots.html

import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import control as ct 
import tfest

from tqdm import tqdm_notebook as tqdm

# class in other files
from module.preprocess import Preprocess
from module.dataset import Datasets
from module.train import PredictSimpleFormulaNet, Train 

if __name__ == '__main__':
    start = time.time()

    # load dataset
    dataset = Datasets()
    df_filt, curv_time = dataset.load_veh(output_type="curve", filt=True)
    #df_eeg, curv_time_drow = dataset.load_eeg(curv_time = curv_time)

    u = df_filt[df_filt["curv_cnt"]==0]["yaw_rate_i"].values
    y = df_filt[df_filt["curv_cnt"]==0]["yaw_rate_a"].values
    t = df_filt[df_filt["curv_cnt"]==0]["time"].values
    d = df_filt[df_filt["curv_cnt"]==0]["time"].diff().mean()

    plt.plot(t,u)
    plt.plot(t,y)
    plt.show()

    # u: input
    # y: output
    te = tfest.tfest(u, y)

    # n_zeros, n_poles
    te.estimate(3, 4, method="h1", time=t.max()-t.min())
    sys = te.get_transfer_function()
    G_h1 = ct.tf(sys.poles,sys.zeros,d)

    te = tfest.tfest(u, y)
    te.estimate(3, 4, method="h2", time=t.max()-t.min())
    sys = te.get_transfer_function()
    G_h2 = ct.tf(sys.poles,sys.zeros,d)

    te = tfest.tfest(u, y)
    te.estimate(3, 4, method="fft", time=t.max()-t.min())
    sys = te.get_transfer_function()
    G_fft = ct.tf(sys.poles,sys.zeros,d)

    out = ct.bode_plot(G_h1, label="h1")
    out = ct.bode_plot(G_h2, label="h2")
    out = ct.bode_plot(G_fft, label="fft")

    plt.legend()
    plt.savefig("../result/visualisation/bode/bode.pdf")
    plt.show()

    ct.pzmap(G_h1, plot=True,grid=True, title="h1")
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.savefig("../result/visualisation/bode/pzmap_h1.pdf")
    plt.show()

    ct.pzmap(G_h2, plot=True, grid=True, title="h2")
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.savefig("../result/visualisation/bode/pzmap_h2.pdf")
    plt.show()

    ct.pzmap(G_fft, plot=True, grid=True, title="fft")
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.savefig("../result/visualisation/bode/pzmap_fft.pdf")
    plt.show()

    print("elapse time", time.time()-start)
