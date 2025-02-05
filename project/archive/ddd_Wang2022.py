# ************************************************************************* #
#                                                                           #
#        This is for the analysis of DDD system by Wang2022                 #
#        Dataset to be used for the analysis is Aygun2024.                  #
#                                                                           #
#        DOI                                                                #
#        Aygun2024      :0.7910/DVN/HMZ5RG                                  #
#        Wang2022       :10.1016/j.trc.2022.103561                          #
#                                                                           #
# ************************************************************************* #

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.model_selection import train_test_split

# class in other files
from module.dataset import Datasets
from module.feat.feat_extract import feat_Wang2022
from module.feat.feat_sel import feat_sel_Wang2022
from module.classify_model import model_Wang2022

if __name__ == '__main__':
    start = time.time()

    # load dataset
    dataset = Datasets()
    data_raw, simlsl_raw = dataset.load_Aygun2024()

    # feat extraction
    simlsl_raw      = feat_Wang2022(simlsl_raw) 
    simlsl_event    = simlsl_raw.query('DRT_event==1').dropna()
    simlsl_baseline = simlsl_raw.query('DRT_baseline==1').dropna()
    print("feat")
    feats = feat_sel_Wang2022(simlsl_event, simlsl_baseline)
    #model = model_Wang2022(feats)

    X = pd.concat([simlsl_event[feats],         simlsl_baseline[feats]      ]) 
    y = pd.concat([simlsl_event.DRT_event - 1,  simlsl_baseline.DRT_baseline]) 
    X.to_csv("X.csv")
    y.to_csv("y.csv")
    sys.exit()

    X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size = 0.15,
            random_state = random.randint(0, 100)
            )
    X_train = X_train.to_numpy().reshape(len(X_train), 1, len(feats))
    y_train = y_train.to_numpy().reshape(len(X_train), 1)
    X_test  = X_test.to_numpy().reshape(len(X_test), 1, len(feats))
    y_test  = y_test.to_numpy().reshape(len(X_test), 1)

    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.33, epochs=5, batch_size=128)
    y_pred  = model.predict(X_test, verbose=0)

    auc = metrics.roc_auc_score(y_test, y_pred)
    print("AUC:",metrics.roc_auc_score(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_pred) #, pos_label=clf.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()

    print("elapse time", time.time()-start)
    sys.exit()

