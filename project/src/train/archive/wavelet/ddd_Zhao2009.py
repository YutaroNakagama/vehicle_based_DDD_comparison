# ************************************************************************* #
#                                                                           #
#        This is for the analysis of DDD system by Zhao2009.          #
#        Dataset to be used for the analysis is Aygun2024.                  #
#                                                                           #
#        DOI                                                                #
#        Aygun2024      :0.7910/DVN/HMZ5RG                                  #
#        Zhao2009       :0.3390/s19040943                                   #
#                                                                           #
# ************************************************************************* #

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import matplotlib.pyplot as plt
import pandas as pd

# class in other files
from module.dataset import Datasets
from module.feat.feat_extract import feat_Zhao2009
from module.classify_model import get_result_svm

if __name__ == '__main__':
    start = time.time()

    # load dataset
    dataset = Datasets()
    data_raw, veh_raw = dataset.load_Aygun2024()

    data_raw.to_csv("./data_raw.csv")
    veh_raw.to_csv("./veh_raw.csv")
    sys.exit()

    X_db6, X_ghm, Y = feat_Zhao2009(data_raw, veh_raw)

    # SVM
    accuracy,precision,recall,AUC = get_result_svm(X_db6,Y)
    print('\naccuracy, precision, recall,AUC, ', accuracy,precision,recall,AUC)
    accuracy,precision,recall,AUC = get_result_svm(X_ghm,Y)
    print('\naccuracy,precision,recall,AUC',accuracy,precision,recall,AUC)
    
    print("elapse time", time.time()-start)
    sys.exit()


