import warnings
warnings.filterwarnings("ignore")

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from scipy.signal import butter,filtfilt
from torchsummary import summary

# class in other files
from module.preprocess import Preprocess
from module.dataset import Datasets

class PredictSimpleFormulaNet(nn.Module):
    '''class for net layers & input data size
    input_size: data size for LSTM input (not time series size)
    hideen_size: size of hidden vector
    batch_first: tensor form for LSTM input. 
                 default is False: (series size, batch size, input size)
                 in case of True: (batch size, series size, input size)
    '''
    def __init__(self, input_size, output_size, hidden_size, num_layer, batch_first):
        super(PredictSimpleFormulaNet, self).__init__()
        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           num_layers = num_layer,
                           batch_first = batch_first)
        self.output_layer = nn.Linear(hidden_size, output_size)
        # weight initialization
        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)

    def forward(self, inputs):
        ''' forward propagation
        arg of self.rnn input tensor format 
            -> (batch size, series size, input size) if batch_first=True 
        tensor form of h
            -> (batch size, series size, hidden vector size)
            -> (batch size, series size, hidden vector size * 2) for bidirectional
        _ is taple for hidden vector and cell
        input h[:,-1] into self.output_layer 
        tensor form of h[:,-1] -> (batch size, hidden vector size)
    
        e.g)
        input   = [[[sin(t0)] [sin(t1)] [sin(t2)]] 
                   [[sin(t1)] [sin(t2)] [sin(t3)]]] 
        h       = [[[h(t0)] [h(t1)] [h(t2)]]
                   [[h(t1)] [h(t2)] [h(t3)]]] 
        h[:,-1] = [[[h(t2)]] [[h(t3)]]] 
        '''
        h, _= self.rnn(inputs)
        output = self.output_layer(h[:, -1])
        return output

class Train():
    ''' class for training, dataset creation, comparison 
    between train model and label.
    '''
    def __init__(self, input_size, output_size, hidden_size, num_layer, batch_first, lr):
        # definition for model, loss function, optimization method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device:", self.device)
        self.net = PredictSimpleFormulaNet(input_size, 
                                           output_size, 
                                           hidden_size, 
                                           num_layer,
                                           batch_first
                                           ).to(self.device)
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.net.parameters(), 
                                    lr=lr, 
                                    betas=(0.9, 0.999), 
                                    amsgrad=True)
        #summary(self.net, [(1,input_size,input_size),(1,input_size,input_size)])
        self.hidden_size = hidden_size
        self.num_layer = num_layer

    def make_dataset(self, dataset_num, sequence_length, t_start, dataset, max_diff=10):
        ''' function for training dataset creation
        produce train & test datasets and separate train & test in main().

        - dataset_num       : # of dataset
        - sequence_length   : size of series
        - t_start           : start time of data creation
        - calc_mode         : decide sin or cos

        returns:
        - np.array(dataset_inputs) -> (dataset_num, sequence_length)
            e.g.) [[sin(t0), sin(t1), sin(t2)]
                   [sin(t1), sin(t2), sin(t3)]
                   ...]
        - np.array(dataset_labels) -> (dataset_num, network output size)
            e.g.) [sin(t3), sin(t4), ...]
        - np.array(dataset_times) -> (dataset_num, network output size)
            e.g.) [t3, t4, ...]

        splited in main() and test part will be used in pred_result_plt()
        '''
        dataset_inputs = []
        dataset_labels = []
        dataset_times = []
        mul_step_pred = 1
        for t in range(dataset_num - mul_step_pred):
            #if (max(np.diff(dataset[t_start + t:t_start + t + sequence_length])) < max_diff) and \
            #        (np.max(dataset[t_start + t:t_start + t + sequence_length]) > 0.001):
            #    dataset_inputs.append([dataset[t_start + t + i] for i in range(sequence_length)])
            #    dataset_labels.append(dataset[(t_start + t + sequence_length + mul_step_pred)])
            #    dataset_times.append(t_start + t + sequence_length)
                dataset_inputs.append([dataset[t_start + t + i] for i in range(sequence_length)])
                dataset_labels.append(dataset[(t_start + t + sequence_length + mul_step_pred)])
                dataset_times.append(t_start + t + sequence_length)
        print("test = {}, {}, {}".format(np.array(dataset_inputs).shape,\
                                         np.array(dataset_labels).shape,\
                                         np.array(dataset_times).shape))

        return (np.array(dataset_inputs), 
                np.array(dataset_labels), 
                np.array(dataset_times),)

    @torch.compile()
    def train_step(self, inputs, labels):
        ''' this function is for parameter update by train data and labels 
        input -> (batch size, series size, input size) 
            e.g.) [[[sin(t0)] [sin(t1)] [sin(t2)]]
                   [[sin(t14)] [sin(t15)] [sin(t16)]]
                   ...]
        labels -> (batch size, output size)
            e.g.) [[sin(t3)] [sin(t17)] ...]
        '''
        #inputs = torch.Tensor(inputs).to(self.device)
        #labels = torch.Tensor(labels).to(self.device)
        inputs = torch.from_numpy(inputs).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)
        self.net.train()
        preds = self.net(inputs)
        loss = self.criterion(preds, labels)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # limit 2.0 to prevent computational un-stable
        nn.utils.clip_grad_value_(self.net.parameters(), clip_value=2.0)
        self.optimizer.step()

        return loss, preds

    #@torch.no_grad()
    def test_step(self, inputs, labels):
        ''' function for calc loss from test data input & labels
        '''
        #inputs = torch.Tensor(inputs).to(self.device)
        #labels = torch.Tensor(labels).to(self.device)
        inputs = torch.from_numpy(inputs).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)
        self.net.eval()
        preds = self.net(inputs)
        loss = self.criterion(preds, labels)

        return loss, preds

    def train(self, train_inputs, train_labels, test_inputs, test_labels,\
              epochs, batch_size, sequence_length, input_size, plot=False):
        ''' make batch and calc loss
        '''
        torch.backends.cudnn.benchmark = True  

        train_epoch_loss = []
        valid_epoch_loss = []
        n_batches_train = int(train_inputs.shape[0] / batch_size)
        n_batches_test = int(test_inputs.shape[0] / batch_size)
        print("test_inputs.shape[0]", test_inputs.shape[0])
        print("n_batches_test      ", n_batches_test      )
        print("batch_size          ", batch_size          )
        for epoch in range(epochs):
            #print('Epoch {}/{}'.format(epoch+1, epochs))
            train_loss = 0.
            test_loss = 0.
            # shuffle the combination 
            # e.g.)
            # - train_inputs
            #       [[sin(t0), sin(t1), sin(t2)]
            #        [sin(t1), sin(t2), sin(t3)]]...]                    
            #                    v 
            #       [[sin(t22), sin(t23), sin(t24)]
            #        [sin(t14), sin(t15), sin(t16)]]...]                    
            # - train_inputs
            #       [sin(t3), sin(t4), ...]
            #                    v 
            #       [sin(t25), sin(t17), ...]
            train_inputs_shuffle, train_labels_shuffle = shuffle(train_inputs, train_labels)
            for batch in range(n_batches_train):
                start = batch * batch_size
                end = start + batch_size
                # change the form to model input
                # e.g.)
                #       [[sin(t22), sin(t23), sin(t24)]
                #        [sin(t14), sin(t15), sin(t16)]]...]                    
                #                        v 
                #       [[[sin(t22)] [sin(t23)] [sin(t24)]
                #        [[sin(t14)] [sin(t15)] [sin(t16)]]...]                    
                #
                #       [sin(t25), sin(t17), ...]
                #                        v 
                #       [[sin(t25)] [sin(t17)] ...]
                loss, _ = self.train_step(np.array(train_inputs_shuffle[start:end]).reshape(-1, sequence_length, input_size), 
                                          np.array(train_labels_shuffle[start:end]).reshape(-1, input_size))
                train_loss += loss.item()

            for batch in range(n_batches_test):
                start = batch * batch_size
                end = start + batch_size
                loss, _ = self.test_step(np.array(test_inputs[start:end]).reshape(-1, sequence_length, input_size), 
                                         np.array(test_labels[start:end]).reshape(-1, input_size))
                test_loss += loss.item()

            train_loss /= float(n_batches_train)
            test_loss /= float(n_batches_test)
            print('loss: {:.5f}, test_loss: {:.5f}'.format(train_loss, test_loss))
            train_epoch_loss.append(train_loss)
            valid_epoch_loss.append(test_loss)

        if plot:
            plt.plot(train_epoch_loss, label="train_loss")
            plt.plot(valid_epoch_loss, label="val_loss")
            plt.legend()
            #plt.savefig("loss.pdf")
            plt.title(str(self.hidden_size)+' '+str(self.num_layer)+' '+str(train_loss))
            #plt.savefig('loss_len_'+str(sequence_length)+'hid-si_'+str(self.hidden_size)+'_layer_'+str(self.num_layer)+'.pdf')
            plt.show()

    @torch.no_grad()
    def pred_result(self, test_inputs, test_labels, test_times, curv_cnt, 
                    sequence_length, input_size, plot=True, sample_type="curve"):
        ''' this function is for prediction and comparison to labels
        '''
        self.net.eval()
        preds = []
        for i in tqdm(range(len(test_inputs))):
            input = np.array(test_inputs[i]).reshape(-1, sequence_length, input_size)
            #input = torch.Tensor(input).to(self.device)
            input = torch.from_numpy(input).to(self.device)
            pred = self.net(input).data.cpu().numpy()
            preds.append(pred[0].tolist())
        preds = np.array(preds)
        test_labels = np.array(test_labels)
        pred_epss = np.abs(test_labels - preds)
        print("pred_epss_min = {}".format(pred_epss.min()))
        print("pred_epss_max = {}".format(pred_epss.max()))

        if sample_type == "curve":
            epss_cnt_list = list(zip(pred_epss[:,1], curv_cnt))
            epss_cnt = pd.DataFrame(epss_cnt_list, columns=['epss','cnt'])
            #epss_ave =[[0] * 2 for i in range(int(min(curv_cnt)),int(max(curv_cnt)))] 
            #epss_ave = []
            index = 0

            epss_ave = [[epss_cnt["epss"][epss_cnt["cnt"]==i].mean(numeric_only=True)]+[i] for i in tqdm(range(int(min(curv_cnt)),int(max(curv_cnt))))]

            '''
            for i in tqdm(range(int(min(curv_cnt)),int(max(curv_cnt)))):
                pred_epss_curv = epss_cnt["epss"][epss_cnt["cnt"]==i].mean(numeric_only=True)
                #pred_epss_curv = epss_cnt["epss"][epss_cnt["cnt"]==i].max()
                if not np.isnan(pred_epss_curv):
                    pred_epss_curv_list = [pred_epss_curv]+[i]
                    epss_ave.append(pred_epss_curv_list)
                    #epss_ave[index] = pred_epss_curv_list
                    #index += 1
            '''
        elif sample_type == "full":
            epss_ave = pred_epss 

        if plot:
            plt.plot(test_times, preds, c='r', label="pred")
            #plt.plot(test_times, test_labels[:,0], c='b', label='label1')
            #plt.plot(test_times, test_labels[:,1], c='k', label='yawrate_a')
            plt.plot(test_times, test_labels[:,0], c='b', label='yawrate_i')
            plt.xlabel('t')
            plt.ylabel('y')
            plt.legend()
            plt.title('compare label and pred, max_dev: {:.5f}'.format(pred_epss.max()))
            #plt.savefig('compare_label_pred.png')
            #plt.savefig('loss_len_'+str(sequence_length)+'hid-si_'+str(self.hidden_size)+'_layer_'+str(self.num_layer)+'.pdf')
            plt.show()
        return np.array(epss_ave)[~np.isnan(np.array(epss_ave)).any(axis=1), :]
        #return np.array(epss_ave)

    def pred_result_UAH(self, test_inputs, test_labels, test_times, 
                    sequence_length, input_size, plot=True):
        ''' this function is for prediction and comparison to labels
        '''
        self.net.eval()
        preds = []
        for i in range(len(test_inputs)):
            input = np.array(test_inputs[i]).reshape(-1, sequence_length, input_size)
            #input = torch.Tensor(input).to(self.device)
            input = torch.from_numpy(input).to(self.device)
            pred = self.net(input).data.cpu().numpy()
            preds.append(pred[0].tolist())
        preds = np.array(preds)
        test_labels = np.array(test_labels)
        pred_epss = np.abs(test_labels - preds)
        print("pred_epss_max = {}".format(pred_epss.max()))
        epss_cnt_list = list(zip(pred_epss[:,1]))
        epss_cnt = pd.DataFrame(epss_cnt_list, columns=['epss'])
        epss_ave = []
#        for i in range(int(min(curv_cnt)),int(max(curv_cnt))):
#            pred_epss_curv = epss_cnt["epss"][epss_cnt["cnt"]==i].mean()
#            #pred_epss_curv = epss_cnt["epss"][epss_cnt["cnt"]==i].max()
#            pred_epss_curv_list = [pred_epss_curv]+[i]
#            epss_ave.append(pred_epss_curv_list)

        if plot:
            plt.plot(test_times, preds, c='r', label="pred")
            #plt.plot(test_times, test_labels[:,0], c='b', label='label1')
            plt.plot(test_times, test_labels[:,1], c='k', label='label2')
            plt.xlabel('t')
            plt.ylabel('y')
            plt.legend()
            plt.title('compare label and pred')
            #plt.savefig('compare_label_pred.png')
            plt.show()
        return np.array(epss_cnt_list)

