
import itertools as itr
import pandas as pd
import scipy as sp
import numpy as np
import scipy.stats as ss
from scipy.stats import pearsonr, ttest_ind

class CorrelationAnalysis:
    # 相関関数のコマンドテーブル
    __CORRFUNC = {
            'pearson':ss.pearsonr,
            'kendall':ss.kendalltau,
            'spearman':ss.spearmanr }

    def __init__(self,
            method='pearson',
            alpha =0.05,
            partial=False ):
        self.method_  = method  # {'pearson'|'kendall'|'spearman'}
        self.alpha_   = alpha   # significant level (default:0.05)
        self.partial_ = partial # computing partial correlation
        self.r_       = None    # correlation matrix @ Array
        self.p_       = None    # p-value matrix @ Array
        self.sigs_    = None    # significant correlation coefficients

    # 相関係数とp値を計算する
    def analyze(self,df):
        std_df = self.standardize(df)
        if self.partial_ is True :
            r,p = self.pcorr(std_df,method=self.method_)
        else:
            r,p = self.corr(std_df,method=self.method_)
        self.r_ = r
        self.p_ = p
        self.sigs_ = self.siglist(r,p,alpha=self.alpha_)
        return self.sigs_

    # 相関行列を返す
    def r_values(self):
        if self.r_ is not None :
            return self.r_
        else:
            print("c_coefs() call error : r is None")

    # p値行列を返す
    def p_values(self):
        if self.p_ is not None :
            return self.p_
        else:
            print("p_values() call error : p is None")

    # 有意な値だけをリストにする
    def significants(self):
        if self.sigs_ is not None :
            return self.sigs_
        else:
            print("significants() call error : p is None")

    #標準化
    #https://note.nkmk.me/python-list-ndarray-dataframe-normalize-standardize/
    @staticmethod
    def standardize(df,ddof=False):
        stddat = ( df - df.mean() ) / df.std(ddof=ddof)
        stddat.index = df.index
        stddat.columns = df.columns
        return stddat

    # p値付き相関係数行列の計算
    #
    # ARGS
    #  df   : pandas.DataFrame型のデータ
    #  method: 相関のタイプ {'pearson'|'kendall'|'spearman'}
    #
    # RETURN
    #  r_df  : 相関係数行列（pandas.DataFrame）
    #  p_df  : p値行列（pandas.DataFrame）
    #
    @staticmethod
    def corr(df,method="pearson"):
        # 空のDataFrameを用意する
        r_df = pd.DataFrame(index=df.columns, columns=df.columns)
        p_df = pd.DataFrame(index=df.columns, columns=df.columns)
        # ゼロで初期化する
        r_df.fillna(0.0,inplace=True)
        p_df.fillna(0.0,inplace=True)
        # 対角成分を1.0にする
        for i in r_df.columns:
            r_df.loc[[i],[i]] = 1.0
        # 非対角成分を計算する
        corrfunc = CorrelationAnalysis.__CORRFUNC[method]
        for i,j in itr.combinations(df,2):
            x = df.loc[:,[i]].values
            y = df.loc[:,[j]].values
            r,p = corrfunc(np.ravel(x),np.ravel(y))
            r_df.loc[[i],[j]] = r
            r_df.loc[[j],[i]] = r
            p_df.loc[[i],[j]] = p
            p_df.loc[[j],[i]] = p
        # 相関とp値を返却
        return r_df,p_df

    #p値つき編相関係数行列の計算
    #
    # ARGS
    #  df     : pandas.DataFrame型のデータ
    #  method : 相関のタイプ {'pearson'|'kendall'|'spearman'}
    #
    # RETURN
    #  pcor_df: 偏相関係数行列（pandas.DataFrame）
    #  p_df   : p値行列（pandas.DataFrame）
    #
    @staticmethod
    def pcorr(df,method="pearson"):
        cor = df.corr(method=method)
        n   = df.shape[0] # number of data
        l   = df.shape[1] # number of labels
        # 偏相関係数の計算
        cor_inv = sp.linalg.pinv(cor) # 従属変数が存在する場合に備えて、擬逆行列pinvを使用
        denom = 1 / np.sqrt(np.diag(cor_inv))
        numer = np.repeat(denom,l).reshape(l,l)
        pcor  = (-cor_inv) * denom * numer
        np.fill_diagonal(pcor,0) #ゼロ割防止のため対角成分を０にする
        # 統計検定量(t)の計算
        t = np.abs(pcor) * np.sqrt(n-3) / np.sqrt(1-pcor*pcor)#３変数間の相関なのでn-3
        np.fill_diagonal(pcor,1) #対角成分を自己相関値=1にする
        # p値の計算
        p = 2*sp.stats.t.cdf(-t,n-3) #両側、Array型
        # pandas.DataFrameへ変換
        pcor_df = pd.DataFrame(data=pcor,index=cor.index,columns=cor.columns)
        p_df=pd.DataFrame(data=p,index=cor.index,columns=cor.columns)
        p_df.to_csv("./p_df.csv")
        return pcor_df,p_df

    # 相関行列とp値行列から有意な相関係数リストを生成する
    #  r     : correlation matrix @ pandas.DataFrame
    #  p     : p-value matrix @ pandas.DataFrame
    #  alpha : significant level
    @staticmethod
    def siglist(r,p,alpha=0.05):
        clist = []
        n = r.columns.size
        j = 0
        while j < n:
            i = 0
            while i < j:
                if( p.iloc[i,j] < alpha ):
                    first = r.columns[i]
                    second= r.columns[j]
                    rvalue= r.iloc[i,j]
                    pvalue= p.iloc[i,j]
                    dat = { 'first':first, 'second':second, 'r':rvalue, 'p':pvalue }
                    clist.append(dat)
                i+=1
            j+=1
        return clist

    # 有意な相関のリスト
    #  df : pandas dataFrame format
    #  alpha : max of p-value
    @staticmethod
    def corlist_fast(df,alpha=0.05,method='pearson'):
        corrfunc = CorrelationAnalysis.__CORRFUNC[method]
        clist=[]
        for i,j in itr.combinations(df,2):
            x = df.loc[:,[i]].values
            y = df.loc[:,[j]].values
            r,p = corrfunc(np.ravel(x),np.ravel(y))
            if p < alpha :
                cpair = { 'first':i, 'second':j, 'r':r, 'p':p }
                clist.append(cpair)
        return clist

    # 有意な相関のリスト
    #  df : pandas dataFrame format
    #  alpha : max of p-value
    @staticmethod
    def corlist(df,alpha=0.05,method='pearson'):
        r,p = CorrelationAnalysis.corr(df,method=method)
        return CorrelationAnalysis.siglist(r,p,alpha)

    #有意な偏相関係数リスト
    #  df : pandas dataFrame format
    #  alpha : max of p-value
    @staticmethod
    def pcorlist(df,alpha=0.05,method="pearson"):
        r,p = CorrelationAnalysis.pcorr(df,method=method)
        return CorrelationAnalysis.siglist(r,p,alpha)

    # 有意な相関係数リストを表示する
    @staticmethod
    def print_clist(clist):
        for item in clist:
            print("r=%.2f, p=%.4f : %-16s <-> %-16s"%(item['r'],item['p'],item['first'],item['second']))

    # 分析結果を表示する
    def __str__(self):
        x  = "\n"
        x += "Computing Method   : " + self.method_ + "\n"
        x += "Significant Level  : alpha = " + str(self.alpha_) + "\n"
        x += "Enable Partial     : " + str(self.partial_) + "\n"
        x += "\n"
        if self.r_ is not None:
            pd.set_option('display.max_columns', len(self.r_.columns))
            pd.set_option('display.max_rows', len(self.r_.index))
            x += "Correlations:\n" + str(self.r_) + "\n\n"
        if self.p_ is not None:
            pd.set_option('display.max_columns', len(self.p_.columns))
            pd.set_option('display.max_rows', len(self.p_.index))
            x += "p-Values:\n" + str(self.p_) + "\n\n"
        if self.sigs_ is not None:
            x += "List of Significant Values:\n"
            for item in self.sigs_ :
                x += "r={0:.2f}, p={1:.4f} : {2:>16s} <-> {3:<16s}\n".format(item['r'],item['p'],item['first'],item['second'])
            return x

def feat_sel_Wang2022(simlsl_event, simlsl_baseline):
    feats = [
        'speed_mean','lon_acc_mean','lat_acc_mean','lane_off_mean','str_rate_mean',
        'speed_std','lon_acc_std','lat_acc_std','lane_off_std','str_rate_std',
        'speed_pe','lon_acc_pe','lat_acc_pe','lane_off_pe','str_rate_pe',
        ]
    ttest_result = pd.DataFrame(columns=['statistic','p_val'], index=feats)
    for f in feats:
        ttest_result.statistic.loc[f], ttest_result.p_val.loc[f] = ttest_ind(simlsl_event[f],simlsl_baseline[f])
    #ttest_result.to_csv('ttest.csv')
    feats = [
            'speed_mean','lon_acc_mean','lat_acc_mean','lane_off_mean',#'str_rate_mean',
            'speed_std','lon_acc_std','lat_acc_std','lane_off_std','str_rate_std',
            #'speed_pe','lon_acc_pe','lat_acc_pe','lane_off_pe','str_rate_pe',
            ]

    return feats


