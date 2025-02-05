# ************************************************************************* #
#                                                                           #
#        This is for the analysis of DDD system by Arefnezhad2019.          #
#        Dataset to be used for the analysis is Aygun2024.                  #
#                                                                           #
#        DOI                                                                #
#        Arefnezhad2019: 0.3390/s19040943                                   #
#        Aygun2024:      0.7910/DVN/HMZ5RG                                  #
#                                                                           #
# ************************************************************************* #

import warnings
warnings.filterwarnings("ignore")

import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from fuzzy_logic.anfis import Anfis

from tqdm import tqdm

# class in other files
from module.dataset import Datasets
from module.feat.feat_extract import feat_Arefnezhad2019
from module.classify_model import get_result_svm

if __name__ == '__main__':

    start = time.time()

    # load dataset
    dataset = Datasets()
    data_raw,_ = dataset.load_Aygun2024()

    # feature extraction
    data_feat, feat_list = feat_Arefnezhad2019(data_raw)
    data_feat = data_feat.dropna()

    # create ground truth
    eeg_gt = "alpha_p_beta"
    eeg_th_high = data_feat[eeg_gt].nlargest(int(len(data_feat)*0.15)).min()
    eeg_th_low  = data_feat[eeg_gt].nsmallest(int(len(data_feat)*0.15)).max()
    data_feat["drowsy"] = -1 
    data_feat["drowsy"][data_feat[eeg_gt] > eeg_th_high] = 1 
    data_feat["drowsy"][data_feat[eeg_gt] < eeg_th_low ] = 0 
    data_feat = data_feat[data_feat.drowsy != -1]
    Y = data_feat["drowsy"]

    data_feat.to_csv('./data_feat.csv')
    sys.exit()

    # feature evaluation (filtering method)
    index_l = []
    for feat in feat_list:
        fisher_id = abs((data_feat[feat][data_feat.drowsy == 0].mean() - data_feat[feat][data_feat.drowsy == 1].mean()) /
                        (data_feat[feat][data_feat.drowsy == 0].std() - data_feat[feat][data_feat.drowsy == 1].std()))
        corr_id = (data_feat[[feat, "drowsy"]].cov().iat[0,1]) / (data_feat[feat].std() - data_feat["drowsy"].std())
        ttest_id =\
                abs(data_feat[feat][data_feat.drowsy == 0].mean() - data_feat[feat][data_feat.drowsy == 1].mean()) /\
                math.sqrt(
                        (data_feat[feat][data_feat.drowsy == 0].std() / len(data_feat[feat][data_feat.drowsy == 0])) +\
                        (data_feat[feat][data_feat.drowsy == 1].std() / len(data_feat[feat][data_feat.drowsy == 1])) \
                        )
        mutual_info_id = mutual_info_score(data_feat[feat], data_feat["drowsy"])
        index_l.append([fisher_id] + [corr_id] + [ttest_id] + [mutual_info_id])

    index_df = pd.DataFrame(index_l,
            columns=['fisher', 'corr', 'ttest', 'mutual_info'],
            index=feat_list)
    print(index_df)
    print(index_df.to_numpy().T)
    print(np.ones(len(index_df.index)))
    print(np.array([random.randint(0,1) for i in range(len(index_df.index))]))
    y = np.array([random.randint(0,1) for i in range(len(index_df.index))])

    anfis: Anfis = Anfis(index_df.to_numpy().T, y)
    anfis.train()
    index_df["id"] = 0
    for i in range(len(index_df.index)):
        print(anfis.calculate(index_df.iloc[i,:-1].to_numpy()))
        index_df.id.iat[i] = anfis.calculate(index_df.iloc[i,:-1].to_numpy())
    print(index_df)
    feat_sel = index_df[index_df.id>0.5].index.tolist()
    print(feat_sel)
    X = data_feat[feat_sel]
    acc,_,_,_ = get_result_svm(X,Y)
    print(acc)
    sys.exit()

    # corr
    feat_corr = data_feat.corr().drowsy.loc[feat_list].sort_values(ascending=False)
    
    result = []
    for n in range(len(feat_list)):
        feat_sel = feat_corr[:n+1].index.to_list()
        X = data_feat[feat_sel]

        # classification
        result.append(get_result_svm(X,Y))

    result_df = pd.DataFrame(result,columns=['accuracy','precision','recall','AUC'])
    result_df['feat_n'] = range(len(feat_list))
    result_df.plot(x='feat_n',y=['accuracy','AUC'])
    plt.show()

    # https://github.com/Luferov/FuzzyLogicToolBox

    print("elapse time", time.time()-start)
