#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix,matthews_corrcoef,roc_auc_score,f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
import os 
import matplotlib.pyplot as plt
from collections import OrderedDict
from torcheval.metrics.functional import binary_f1_score   
from torcheval.metrics.functional import binary_auprc
import shap
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import random

#class

class innModel_2(nn.Module):
    def __init__(self, masks, device,data_tup):
        super().__init__()
        self.masks = [torch.Tensor(m.values).to(device) for m in masks]
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(m.size(0), m.size(1))) for m in self.masks])
        self.bias = nn.ParameterList([nn.Parameter(torch.randn(m.size(1))) for m in self.masks])
        self.activation=torch.nn.Tanh()
        self.aap = nn.AdaptiveAvgPool1d(1)

        self.input_X= data_tup[0]
        self.weight_1=data_tup[1]
        self.bias_1=data_tup[2]
        self.maskGene= torch.FloatTensor(masks[0].values).to(device)
        self.H1_X= self.get_H1_X(self.input_X, self.weight_1,self.bias_1,self.maskGene)
    
    def get_H1_X(self, input_X, weight_1, bias_1, maskGene):
        x = torch.tanh(torch.matmul(input_X, (maskGene*weight_1)) + bias_1)
        return x

    def forward(self, x):        
        for i in range(1,len(self.weights)-1):
            x = torch.matmul(x, (self.weights[i]*self.masks[i]))+self.bias[i]      
            if i==1:
                keep = x
            else:
                keep = torch.concat([keep, x], 1)
            x = torch.tanh(x)
        x = torch.matmul(x, (self.weights[-1]*self.masks[-1]))+self.bias[-1]
        keep = torch.concat([keep, x], 1)
        res = self.aap(keep)
        return res
    
class innModel_3(nn.Module):
        def __init__(self, masks, device,data_tup, H2_tup):
            super().__init__()
            self.masks = [torch.Tensor(m.values).to(device) for m in masks]
            self.weights = nn.ParameterList([nn.Parameter(torch.randn(m.size(0), m.size(1))) for m in self.masks])
            self.bias = nn.ParameterList([nn.Parameter(torch.randn(m.size(1))) for m in self.masks])
            self.activation=torch.nn.Tanh()
            self.aap = nn.AdaptiveAvgPool1d(1)

            self.maskGene = torch.FloatTensor(masks[0].values).to(device) #
            self.input_X= data_tup[0]
            self.weight_1=data_tup[1]
            self.bias_1=data_tup[2]

            self.weight_2=H2_tup[0]
            self.bias_2=H2_tup[1]

            self.mask_layer_2= torch.FloatTensor(masks[1].values).to(device)
            self.H2_X=  self.get_H2_X(self.input_X, self.weight_1, self.bias_1, self.maskGene, self.weight_2, self.bias_2, self.mask_layer_2)
        def get_H2_X(self, input_X, weight_1, bias_1, maskGene, weight_2, bias_2, mask_layer_2):
            x = torch.tanh(torch.matmul(input_X, (maskGene*weight_1)) + bias_1)
            x = torch.tanh(torch.matmul(x, (mask_layer_2*weight_2)) + bias_2)
            return x

        def forward(self, x):        
            for i in range(2,len(self.weights)-1):
                x = torch.matmul(x, (self.weights[i]*self.masks[i]))+self.bias[i]      
                if i==2:
                    keep = x
                else:
                    keep = torch.concat([keep, x], 1)
                x = torch.tanh(x)
            x = torch.matmul(x, (self.weights[-1]*self.masks[-1]))+self.bias[-1]
            keep = torch.concat([keep, x], 1)
            res = self.aap(keep)
            return res
        
class innModel_4(nn.Module):
        def __init__(self, masks, device,data_tup, H2_tup, H3_tup):
            super().__init__()
            self.masks = [torch.Tensor(m.values).to(device) for m in masks]
            self.weights = nn.ParameterList([nn.Parameter(torch.randn(m.size(0), m.size(1))) for m in self.masks])
            self.bias = nn.ParameterList([nn.Parameter(torch.randn(m.size(1))) for m in self.masks])
            self.activation=torch.nn.Tanh()
            self.aap = nn.AdaptiveAvgPool1d(1)

            self.maskGene = torch.FloatTensor(masks[0].values).to(device) #
            self.input_X= data_tup[0]
            self.weight_1=data_tup[1]
            self.bias_1=data_tup[2]

            self.weight_2=H2_tup[0]
            self.bias_2=H2_tup[1]

            self.weight_3=H3_tup[0]
            self.bias_3=H3_tup[1]

            self.mask_layer_2= torch.FloatTensor(masks[1].values).to(device)
            self.mask_layer_3= torch.FloatTensor(masks[2].values).to(device)
            self.H3_X=  self.get_H3_X(self.input_X, self.weight_1, self.bias_1, self.maskGene,
                                      self.weight_2, self.bias_2,self.mask_layer_2,self.weight_3, 
                                      self.bias_3, self.mask_layer_3)
        def get_H3_X(self, input_X, weight_1, bias_1, maskGene, weight_2, bias_2,mask_layer_2, weight_3, 
                                      bias_3, mask_layer_3):
            x = torch.tanh(torch.matmul(input_X, (maskGene*weight_1)) + bias_1)
            x = torch.tanh(torch.matmul(x, (mask_layer_2*weight_2)) + bias_2)
            x = torch.tanh(torch.matmul(x, (mask_layer_3*weight_3)) + bias_3)
            return x

        def forward(self, x):        
            for i in range(3,len(self.weights)-1):
                x = torch.matmul(x, (self.weights[i]*self.masks[i]))+self.bias[i]      
                if i==3:
                    keep = x
                else:
                    keep = torch.concat([keep, x], 1)
                x = torch.tanh(x)
            x = torch.matmul(x, (self.weights[-1]*self.masks[-1]))+self.bias[-1]
            keep = torch.concat([keep, x], 1)
            res = self.aap(keep)
            return res
        
class innModel_5(nn.Module):
        def __init__(self, masks, device,data_tup, H2_tup, H3_tup, H4_tup):
            super().__init__()
            self.masks = [torch.Tensor(m.values).to(device) for m in masks]
            self.weights = nn.ParameterList([nn.Parameter(torch.randn(m.size(0), m.size(1))) for m in self.masks])
            self.bias = nn.ParameterList([nn.Parameter(torch.randn(m.size(1))) for m in self.masks])
            self.activation=torch.nn.Tanh()
            self.aap = nn.AdaptiveAvgPool1d(1)

            self.maskGene = torch.FloatTensor(masks[0].values).to(device) #
            self.input_X= data_tup[0]
            self.weight_1=data_tup[1]
            self.bias_1=data_tup[2]

            self.weight_2=H2_tup[0]
            self.bias_2=H2_tup[1]

            self.weight_3=H3_tup[0]
            self.bias_3=H3_tup[1]

            self.weight_4=H4_tup[0]
            self.bias_4=H4_tup[1]

            self.mask_layer_2= torch.FloatTensor(masks[1].values).to(device)
            self.mask_layer_3= torch.FloatTensor(masks[2].values).to(device)
            self.mask_layer_4= torch.FloatTensor(masks[3].values).to(device)

            self.H4_X=  self.get_H4_X(self.input_X, self.weight_1, self.bias_1, self.maskGene, 
                                      self.weight_2, self.bias_2,self.mask_layer_2,self.weight_3, 
                                      self.bias_3, self.mask_layer_3,self.weight_4,self.bias_4, self.mask_layer_4)
        def get_H4_X(self, input_X, weight_1, bias_1, maskGene, weight_2, bias_2,mask_layer_2, weight_3, 
                                      bias_3, mask_layer_3, weight_4,bias_4, mask_layer_4):
            x = torch.tanh(torch.matmul(input_X, (maskGene*weight_1)) + bias_1)
            x = torch.tanh(torch.matmul(x, (mask_layer_2*weight_2)) + bias_2)
            x = torch.tanh(torch.matmul(x, (mask_layer_3*weight_3)) + bias_3)
            x = torch.tanh(torch.matmul(x, (mask_layer_4*weight_4)) + bias_4)
            return x

        def forward(self, x):        
            x = torch.matmul(x, (self.weights[-1]*self.masks[-1]))+self.bias[-1]
            res = self.aap(torch.sigmoid(x))
            return res
class innModel_6(nn.Module):
        def __init__(self, masks, device,data_tup, H2_tup, H3_tup, H4_tup, H5_tup):
            super().__init__()
            self.masks = [torch.Tensor(m.values).to(device) for m in masks]
            self.weights = nn.ParameterList([nn.Parameter(torch.randn(m.size(0), m.size(1))) for m in self.masks])
            self.bias = nn.ParameterList([nn.Parameter(torch.randn(m.size(1))) for m in self.masks])
            self.activation=torch.nn.Tanh()
            self.aap = nn.AdaptiveAvgPool1d(1)

            self.maskGene = torch.FloatTensor(masks[0].values).to(device) #
            self.input_X= data_tup[0]
            self.weight_1=data_tup[1]
            self.bias_1=data_tup[2]

            self.weight_2=H2_tup[0]
            self.bias_2=H2_tup[1]

            self.weight_3=H3_tup[0]
            self.bias_3=H3_tup[1]

            self.weight_4=H4_tup[0]
            self.bias_4=H4_tup[1]

            self.weight_5=H5_tup[0]
            self.bias_5=H5_tup[1]

            self.mask_layer_2= torch.FloatTensor(masks[1].values).to(device)
            self.mask_layer_3= torch.FloatTensor(masks[2].values).to(device)
            self.mask_layer_4= torch.FloatTensor(masks[3].values).to(device)
            self.mask_layer_5= torch.FloatTensor(masks[4].values).to(device)

            self.H5_X=  self.get_H5_X(self.input_X, self.weight_1, self.bias_1, self.maskGene,
                                      self.weight_2,self.bias_2,self.mask_layer_2,
                                      self.weight_3,self.bias_3, self.mask_layer_3,
                                      self.weight_4,self.bias_4, self.mask_layer_4,
                                      self.weight_5,self.bias_5, self.mask_layer_5)
        def get_H5_X(self, input_X, weight_1, bias_1, maskGene, weight_2, bias_2,mask_layer_2, weight_3, 
                                      bias_3, mask_layer_3, weight_4,bias_4, mask_layer_4, weight_5,bias_5, mask_layer_5):
            x = torch.tanh(torch.matmul(input_X, (maskGene*weight_1)) + bias_1)
            x = torch.tanh(torch.matmul(x, (mask_layer_2*weight_2)) + bias_2)
            x = torch.tanh(torch.matmul(x, (mask_layer_3*weight_3)) + bias_3)
            x = torch.tanh(torch.matmul(x, (mask_layer_4*weight_4)) + bias_4)
            x = torch.tanh(torch.matmul(x, (mask_layer_5*weight_5)) + bias_5)
            return x

        def forward(self, x):        
            res = self.aap(torch.sigmoid(x))
            return res

class innModel(nn.Module):
    def __init__(self, masks, device):
        super().__init__()
        self.masks = [torch.Tensor(m.values).to(device) for m in masks]
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(m.size(0), m.size(1))) for m in self.masks])
        self.bias = nn.ParameterList([nn.Parameter(torch.randn(m.size(1))) for m in self.masks])
        self.activation=torch.nn.Tanh()
        self.aap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):        
        for i in range(len(self.weights)-1):
            x = torch.matmul(x, (self.weights[i]*self.masks[i]))+self.bias[i]      
            if i==0:
                keep = x
            else:
                keep = torch.concat([keep, x], 1)
            x = torch.tanh(x)
        x = torch.matmul(x, (self.weights[-1]*self.masks[-1]))+self.bias[-1]
        keep = torch.concat([keep, x], 1)
        res = self.aap(keep)
        return res
    
    
    
class innModel_logit(nn.Module):
    def __init__(self, masks, device):
        super().__init__()
        self.masks = [torch.Tensor(m.values).to(device) for m in masks]
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(m.size(0), m.size(1))) for m in self.masks])
        self.bias = nn.ParameterList([nn.Parameter(torch.randn(m.size(1))) for m in self.masks])
        self.activation=torch.nn.Tanh()
        self.aap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):        
        for i in range(len(self.weights)-1):
            x = torch.matmul(x, (self.weights[i]*self.masks[i]))+self.bias[i]      
            if i==0:
                keep = x
            else:
                keep = torch.concat([keep, x], 1)
            x = torch.tanh(x)
        x = torch.matmul(x, (self.weights[-1]*self.masks[-1]))+self.bias[-1]
        keep = torch.concat([keep, x], 1)
        res = self.aap(keep)
        return res
    


# In[ ]:

class innModel_batch(nn.Module):
    def __init__(self, masks, device, dropout):
        super().__init__()
        self.masks = [torch.Tensor(m.values).to(device) for m in masks]
        
        # Initialize linear layers and BatchNorm layers
        self.batch_norm_layers = nn.ModuleList()
        self.weight = nn.ParameterList([nn.Parameter(torch.randn(m.size(0), m.size(1))) for m in self.masks])
        self.bias = nn.ParameterList([nn.Parameter(torch.randn(m.size(1))) for m in self.masks])
        for mask in self.masks: 
            batch_norm_layer = nn.BatchNorm1d(mask.size(1))
            self.batch_norm_layers.append(batch_norm_layer)
            
        self.activation = torch.nn.Tanh()
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i in range(len(self.weight)-1):
            x = torch.matmul(x, (self.weight[i] * self.masks[i])) + self.bias[i]
            x = self.dropout(x)
            if i==0:
                keep = x
            else:
                keep = torch.concat([keep, x], 1)
            x = self.batch_norm_layers[i](x)  # Apply BatchNorm
            x = self.activation(x)
        x = torch.matmul(x, (self.weight[-1] * self.masks[-1])) + self.bias[-1]
        x = self.batch_norm_layers[-1](x)
        keep = torch.concat([keep, x], 1)    
        
        res = self.aap(x)
        return res

# In[ ]:


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8165, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
            nn.Linear(10,1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# In[ ]:


class DNN(nn.Module):
    def __init__(self, x_dataTensor,masks):
        super(DNN, self).__init__()
        self.x_dataTensor=x_dataTensor
        self.fc1_1 = nn.Linear(masks[0].shape[0],masks[0].shape[1])
        self.fc1 = nn.Linear(masks[1].shape[0],masks[1].shape[1])
        self.fc2 = nn.Linear(masks[2].shape[0],masks[2].shape[1])
        self.fc3 = nn.Linear(masks[3].shape[0],masks[3].shape[1])
        self.fc4 = nn.Linear(masks[4].shape[0],masks[4].shape[1])
        self.fc5 = nn.Linear(masks[4].shape[1],1)

    def forward(self, x):
        x = x.float()
        h1_1 = F.relu(self.fc1_1(x))
        h1 = F.relu(self.fc1(h1_1))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = self.fc5(h4)
        
        return h5


# In[ ]:


class logistic(nn.Module):
    def __init__(self,x_dataTensor):
        super(logistic, self).__init__()
        self.fc1_1 = nn.Linear(x_dataTensor.size()[1], 1)
        self.x_dataTensor = x_dataTensor
    def forward(self, x):
        x = x.float()
        h1_1 = self.fc1_1(x.view(-1,self.x_dataTensor.size()[1]))
        return h1_1
    
class innModel_test_2(nn.Module):
    def __init__(self, masks, device):
        super().__init__()
        self.masks = [torch.Tensor(m.values).to(device) for m in masks[0:2]]
        self.linear1 = nn.Linear(self.masks[-1].shape[1], 248)
        self.linear2 = nn.Linear(248, 32)
        # self.masks.append()
        
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(m.size(0), m.size(1))) for m in self.masks[:2]])
        self.bias = nn.ParameterList([nn.Parameter(torch.randn(m.size(1))) for m in self.masks[:2]])
        self.activation=torch.nn.Tanh()
        self.aap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):        
        for i in range(len(self.weights)):
            x = torch.matmul(x, (self.weights[i]*self.masks[i]))+self.bias[i]      
            if i==0:
                keep = x
            else:
                keep = torch.concat([keep, x], 1)
            x = torch.tanh(x)
        x = self.linear1(x)
        x = self.linear2(x)
        # x = torch.matmul(x, (self.weights[-1]*self.masks[-1]))+self.bias[-1]
        keep = torch.concat([keep, x], 1)
        res = self.aap(keep)
        return res

# In[ ]:


class logistic_1(nn.Module):
    def __init__(self):
        super(logistic, self).__init__()
        self.fc1_1 = nn.Linear(8165, 1000)
        self.fc1 = nn.Linear(1000, 1)

    def forward(self, x):
        x = x.float()
        h1_1 = F.relu(self.fc1_1(x.view(-1,8165)))
        h1 = self.fc1(h1_1)
        return h1



def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def cal_AUC(model, xdata, labeldata):
    
    pre = model(xdata)
    pre = pre.cpu().detach().numpy()
    pre_label = np.where(pre >0.5, 1, 0)
 
    y_label = labeldata.cpu().detach().numpy()
    
    return confusion_matrix(y_label, pre_label), roc_auc_score(y_label, pre_label)

def cal_metrics(model, pre, labeldata):
    
    # pre = model(xdata)
    pre = pre.cpu().detach().numpy()
    pre_label = np.where(pre >0.5, 1, 0)
 
    y_label = labeldata.cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_label, pre_label).ravel()
    f1 = f1_score(y_label, pre_label)
    mcc = matthews_corrcoef(y_label, pre_label)
    
    return (tn, fp, fn, tp), roc_auc_score(y_label, pre_label),f1, mcc,

# In[ ]:


def cal_accuracy(model, xdata, labeldata):
    pred_tr = model(xdata)
    pre_label = pred_tr >0.5
    pre_label = pre_label.to(torch.float)
    return (pre_label == labeldata).to(torch.float).mean()


# In[ ]:


class trLoader(Dataset):
    def __init__(self, x, y,is_test=False):
        super().__init__()
        self.x = x
        self.y = y
        self.is_test = is_test
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x_tensor=self.x
        label = torch.tensor(self.y[idx]).float()
        return x_tensor[idx], label
    
 



def model_initial(masks,device,model_type,dropout=0.25,x_dataTensor="None"):
    if model_type=="logit":
        model = innModel_logit(masks, device)
        criterion=nn.BCEWithLogitsLoss()
    elif model_type=="batch_norm":
        model = innModel_batch(masks, device, dropout)
        criterion=nn.BCEWithLogitsLoss()
    elif model_type=="FCNN":
        model= NeuralNetwork()
        criterion=nn.BCELoss()
    elif model_type=="DNN":
        model=DNN(x_dataTensor,masks)
        criterion=nn.BCEWithLogitsLoss()
    elif model_type=="logistic":
        model=logistic(x_dataTensor)
        criterion=nn.BCEWithLogitsLoss()
    elif model_type=="logistic_1":
        model=logistic()
        criterion=nn.BCEWithLogitsLoss()
    elif model_type=="test_2":
        model=innModel_test_2(masks, device)
        criterion = nn.BCEWithLogitsLoss()

    model=model.to(device)
    return model,criterion



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None 
        self.early_stop = False
        #self.val_loss_min = np.Inf
        self.val_loss_min = 0
        self.delta = delta
        self.path = path
        self.best_metrics=0
        self.best_epoch=0
        self.trace_func = trace_func
    def __call__(self, val_loss, epoch, val_metrics, model):

        #score = -val_loss
        score=val_loss
        metrics=val_metrics
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_metrics = metrics
            self.best_epoch=epoch
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"best epoch : {self.best_epoch}, best metrics : {self.best_metrics}")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_metrics = metrics
            self.best_epoch=epoch

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            #self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.trace_func(f'Validation metrics increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss
        


def model_train(data,x_testTensor,y_testTensor,model_type,lr,epoch,path_base,masks,fold,device,seed):

    x_dataTensor,y_dataTensor,x_validTensor,y_validTensor=data[0],data[1],data[2],data[3] 
    x_dataTensor = torch.nn.functional.normalize(x_dataTensor,1)
    x_dataTensor = x_dataTensor.to(device)
    y_dataTensor = y_dataTensor.to(device)

    x_validTensor = torch.nn.functional.normalize(x_validTensor,1)
    x_validTensor = x_validTensor.to(device)
    y_validTensor = y_validTensor.to(device)
    x_testTensor = torch.nn.functional.normalize(x_testTensor,1)
    x_testTensor = x_testTensor.to(device)
    y_testTensor = y_testTensor.to(device)

    tr_loader = DataLoader(trLoader(x_dataTensor, y_dataTensor), batch_size=64, shuffle=True)

    start=time()
    model,criterion=model_initial(masks,device,model_type)
    model=model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = 0.
    running_loss=0

    tr_result={}
    te_result={}
    va_result={}
    tr_result["epoch"]=[]
    tr_result["loss"]=[]
    tr_result["ACC"]=[]
    va_result["epoch"]=[]
    va_result["loss"]=[]
    va_result["ACC"]=[]
    te_result["epoch"]=[]
    te_result["loss"]=[]
    te_result["ACC"]=[]
    max_valid_acc=0
    max_test_acc=0
    max_epoch=0

    
    for i in range(epoch):
        model.train()
        loss_total = 0

        for batch in tr_loader:
            x_tr = batch[0]
            y_tr = batch[1]
            prediction = model(x_tr.to(device))
            loss = criterion(prediction, y_tr.to(device))
            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()

            loss_total += loss
        loss_total /= len(tr_loader)

        running_loss += loss.item()

        tr_ACC=100 * cal_accuracy(model,x_dataTensor, y_dataTensor).item()
        if i%20==0:
            print(f"[Epoch: {i}] training Accuracy: {tr_ACC:.2f}%")
            print(f"[loss] {loss.item()}")

        tr_result["epoch"].append(i)
        tr_result["loss"].append(loss.item())
        tr_result["ACC"].append(tr_ACC)

        model.eval()
        prediction_test = model(x_validTensor)
        valid_loss = criterion(prediction_test, y_validTensor)
        va_ACC=100 * cal_accuracy(model,x_validTensor, y_validTensor).item()

        va_result["epoch"].append(i)
        va_result["loss"].append(valid_loss.item())
        va_result["ACC"].append(va_ACC)
        if i%20==0:
            print(f"[Epoch: {i}] validation Loss: {valid_loss.item()}")
            print(f"[Epoch: {i}] validation Accuracy: {va_ACC:.2f}%")

        prediction_test = model(x_testTensor)
        test_loss = criterion(prediction_test, y_testTensor)
        te_ACC=100 * cal_accuracy(model,x_testTensor, y_testTensor).item()
        if i%20==0:
            print(f"[Epoch: {i}] test Loss: {test_loss.item()}")
            print(f"[Epoch: {i}] test Accuracy: {te_ACC:.2f}%")
        te_result["epoch"].append(i)
        te_result["loss"].append(test_loss.item())
        te_result["ACC"].append(te_ACC)
        if max_valid_acc<va_ACC:
            max_valid_acc=va_ACC
            max_epoch=i
            if max_test_acc<te_ACC:
                max_test_acc=te_ACC
                max_epoch=i
                if os.path.isfile(f"{path_base}/best_model.pt"):
                    os.rename(f"{path_base}/best_model.pt", f"{path_base}/second_model.pt")
                torch.save(model,f"{path_base}/best_model.pt")
            else:
                max_test_acc=max_test_acc
        elif max_valid_acc==va_ACC:
            max_valid_acc=max_valid_acc
            if max_test_acc<te_ACC:
                max_test_acc=te_ACC
                max_epoch=i
                if os.path.isfile(f"{path_base}/best_model.pt"):
                    os.rename(f"{path_base}/best_model.pt", f"{path_base}/second_model.pt")
                torch.save(model,f"{path_base}/best_model.pt")
            else:
                max_test_acc=max_test_acc
        else:
            pass

    df_res=pd.DataFrame({"Epoch": tr_result["epoch"],"Training loss":tr_result["loss"],
    "Training Accuracy":tr_result["ACC"], "Validation loss":va_result["loss"],
    "Validation Accuracy":va_result["ACC"], "Test loss": te_result["loss"],
    "Test Accuracy" :te_result["ACC"]})
    spendingtime=time()-start
    print(f"{spendingtime/60:.2f} min")


    df_res=pd.DataFrame({"Epoch": tr_result["epoch"],"Training loss":tr_result["loss"],
    "Training Accuracy":tr_result["ACC"], "Validation loss":va_result["loss"],
    "Validation Accuracy":va_result["ACC"], "Test loss": te_result["loss"],
    "Test Accuracy" :te_result["ACC"]})

    df_res.to_csv(path_base+"/table_figure/result.csv",index=False)
    print("Max validation accuracy : "+str(max(df_res["Validation Accuracy"])))
    print("Epoch : "+str(max_epoch))
    # plt.clf()
    tr_ACC = tr_result["loss"]
    va_ACC = va_result["loss"]
    te_ACC = te_result["loss"]

    # x_len = np.arange(len(va_ACC))
    # plt.plot(x_len, tr_ACC, label="Train-set Loss",linewidth="0.3")
    # plt.plot(x_len, va_ACC, label="Validation-set Loss",linewidth="0.3")
    #plt.plot(x_len, te_ACC, label="Test-set Loss",linewidth="0.3")

    # plt.legend(loc='upper right')
    # plt.grid()
    # plt.xlabel('epoch')
    # plt.ylabel('Loss')
    # #plt.axis([0,200,0,1])
    # plt.savefig(path_base+"/table_figure/tr_va_loss.png")
    # print("loss figure save")
    # #plt.show()

    # plt.clf()
    tr_ACC = tr_result["ACC"]
    va_ACC = va_result["ACC"]
    te_ACC = te_result["ACC"]

    # x_len = np.arange(len(va_ACC))
    # plt.plot(x_len, tr_ACC, label="Train-set Accuracy",linewidth="0.3")
    # plt.plot(x_len, va_ACC, label="Validation-set Accuracy",linewidth="0.3")
    #plt.plot(x_len, te_ACC, label="Test-set Accuracy",linewidth="0.3")

    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.xlabel('epoch')
    # plt.ylabel('ACC')
    # #plt.axis([0,200,60,100])
    # plt.savefig(path_base+"/table_figure/tr_va_ACC.png")
    # print("ACC figure save")
    #plt.show()
    if model_type=="logit":
        model = innModel_logit(masks, device)
    elif model_type=="loss":
        model= innModel_loss(masks, device)
    elif model_type=="DNN":
        model= DNN(x_dataTensor,masks)
    elif model_type=="logistic":
        model=logistic()

    #path_base="./result/CD_Rec/20230620_094428_logit_0.0001/"

    #path_base="/storage/shimyj/INN/result/down_CD_inf/20230627_191118_DNN_0.0001"

    model=torch.load(f"{path_base}/best_model.pt").to(device)
    model.eval()
    
    tr_acc=cal_accuracy(model,x_dataTensor,y_dataTensor)
    va_acc=cal_accuracy(model,x_validTensor,y_validTensor)

    prediction_train = model(x_dataTensor)
    prediction_valid = model(x_validTensor)

    tr_f1_score = binary_f1_score(prediction_train.view(-1), y_dataTensor.view(-1), threshold=0.5)
    va_f1_score = binary_f1_score(prediction_valid.view(-1), y_validTensor.view(-1), threshold=0.5)

    tr_AUPRC=binary_auprc(prediction_train.view(-1), y_dataTensor.view(-1))
    va_AUPRC=binary_auprc(prediction_valid.view(-1), y_validTensor.view(-1))

    _, tr_AUC = cal_AUC(model=model, xdata=x_dataTensor, labeldata=y_dataTensor)
    _, va_AUC = cal_AUC(model=model, xdata=x_validTensor, labeldata=y_validTensor)


    rand=torch.randperm(x_testTensor.shape[0])
    rand=rand[0:int(x_testTensor.shape[0]/2)]
    rand_x_data=x_testTensor[rand]
    rand_y_data=y_testTensor[rand]

    te_acc=cal_accuracy(model,rand_x_data,rand_y_data)
    prediction_test = model(rand_x_data)
    te_f1_score = binary_f1_score(prediction_test.view(-1), rand_y_data.view(-1), threshold=0.5)
    te_AUPRC=binary_auprc(prediction_test.view(-1), rand_y_data.view(-1))
    _, te_AUC = cal_AUC(model=model, xdata=rand_x_data, labeldata=rand_y_data)

    metric_list = [te_acc.item(),te_AUC,te_f1_score.item(),te_AUPRC.item()]
    metric_df = pd.DataFrame([metric_list])
    metric_df = pd.concat([metric_df,pd.DataFrame([metric_list])], ignore_index=True,verify_integrity=False)

    metric_df.columns=["accuracy","AUC","F1","AUPRC"]
    metric_df.to_csv(os.path.join(path_base,"test_metrics.csv"),sep=",",index=False)
    test_mean=pd.DataFrame(metric_df.mean(axis=0))

    final_res_dict=OrderedDict(
    [
        ("Lr",[lr]),
        ("Model type",[model_type]),
        ("Epoch",[max_epoch]),
        ("Training acc",[tr_acc.item()]),
        ("Validation acc",[va_acc.item()]),
        ("Test acc",[test_mean.iloc[0,0]]),
        ("Train F1",[tr_f1_score.item()]),
        ("Valid F1",[va_f1_score.item()]),
        ("Test F1",[test_mean.iloc[2,0]]),
        ("Train AUC",[tr_AUC]),
        ("Valid AUC",[va_AUC]),
        ("Test AUC",[test_mean.iloc[1,0]]),
        ("Train AUPRC",[tr_AUPRC.item()]),
        ("Valid AUPRC",[va_AUPRC.item()]),
        ("Test AUPRC",[test_mean.iloc[3,0]]),
        ("split",[fold]),
        ("seed",[seed])
    ])

    final_res=pd.DataFrame(final_res_dict,index = None)

    return model, final_res


def get_Layer_ShapValue(model,input_data_SHAP,reactome_dict):
    train_data=input_data_SHAP[0]
    case_data=input_data_SHAP[1]
    layer=model.layer # 원래 layer-1
    e = shap.DeepExplainer(model, train_data)
    _sv_df=pd.DataFrame()
    
    if layer!=0:
        feature=model.ori_masks[model.layer-1].columns
        name=[]
        for i in feature:
            name.append(reactome_dict[i])
    else:
        feature=model.ori_masks[model.layer].index
    for i in range(1,11):
        rand=torch.randperm(case_data.shape[0])
        rand=rand[0:int(case_data.shape[0]*2/3)]
        rand_data=case_data[rand]
        sv = abs(e.shap_values(rand_data))
        sumOfsv_case=sv.sum(axis=0)
        meanOfsv_case = sumOfsv_case/sv.shape[0]
        meanOfsv_case=pd.DataFrame(meanOfsv_case)
        meanOfsv_case=meanOfsv_case.transpose()
        _sv_df=pd.concat([_sv_df,pd.DataFrame(meanOfsv_case)], ignore_index=True,verify_integrity=False)
        print(f"{i}_rep / 1000")
    if layer!=0:
        _sv_df.columns=name
    else:
        _sv_df.columns=feature
    order_mean=pd.DataFrame(_sv_df.mean(axis=0))
    order_mean=order_mean.reset_index()
    order_mean.columns=["name","value"]
    order_mean=order_mean.sort_values(by=["value"],ascending=False)
    return _sv_df, order_mean    



# In[ ]:
def shap_figure(_sv_df,order_mean,save_path,layer):
    sns.reset_defaults()
    plt.clf()
    top10=order_mean.iloc[0:10,0]
    fig_input=_sv_df.loc[:,top10]
    ax = sns.barplot(data=fig_input,orient="h")
    plt.savefig(os.path.join(save_path,f"shap_value_{layer}.png"),bbox_inches="tight")
    plt.close()
    order_mean.to_csv(os.path.join(save_path,f"shap_value_{layer}.csv"),sep=",",index=False)
    
    

def first_shap(model, x_dataTensor,case_data,feature):
    # feature =input_data.index
    
    input_data_SHAP=[]
    input_data_SHAP.append(x_dataTensor)
    input_data_SHAP.append(case_data)
    
    _sv_df, order_mean = get_Layer_ShapValue(model=model,input_data_SHAP=input_data_SHAP,path_TF=False,feature=feature)
    return _sv_df, order_mean

# In[ ]:




def second_shap(model,masks,x_dataTensor,case_data,device,reactome_dict):
  
    data_tup=(x_dataTensor, model.weights[0], model.bias[0])
    inn_2=innModel_2(masks,device,data_tup).to(device)
    inn_2.weights[0] = model.weights[0]
    inn_2.bias[0] = model.bias[0]

    # inn_2.weights[1] = model.weights[1]
    # inn_2.bias[1] = model.bias[1]

    # inn_2.weights[2] = model.weights[2]
    # inn_2.bias[2] = model.bias[2]

    # inn_2.weights[3] = model.weights[3]
    # inn_2.bias[3] = model.bias[3]

    # inn_2.weights[4] = model.weights[4]
    # inn_2.bias[4] = model.bias[4]

    #plt.clf()
    H2_tensor = inn_2.H1_X.detach()
    test_H1_X= inn_2.get_H1_X(case_data, inn_2.weight_1,inn_2.bias_1,inn_2.maskGene)
    test_H1_X = test_H1_X.detach()
    input_data_SHAP=[]
    input_data_SHAP.append(H2_tensor)
    input_data_SHAP.append(test_H1_X)

    _sv_df, order_mean = get_Layer_ShapValue(model=inn_2,input_data_SHAP=input_data_SHAP,
                                             path_TF=True,feature=list(masks[0].columns),reactome_dict=reactome_dict)

    return _sv_df, order_mean
# In[ ]:

def third_shap(model,masks,x_dataTensor,case_data,device,reactome_dict):

    data_tup=(x_dataTensor.to(device), model.weights[0], model.bias[0])
    H2_tup=(model.weights[1],model.bias[1])

    inn_3=innModel_3(masks,device,data_tup,H2_tup).to(device)

    inn_3.weights[0] = model.weights[0]
    inn_3.bias[0] = model.bias[0]

    inn_3.weights[1] = model.weights[1]
    inn_3.bias[1] = model.bias[1]

    # inn_3.weights[2] = model.weights[2]
    # inn_3.bias[2] = model.bias[2]

    # inn_3.weights[3] = model.weights[3]
    # inn_3.bias[3] = model.bias[3]

    # inn_3.weights[4] = model.weights[4]
    # inn_3.bias[4] = model.bias[4]

    H3_tensor = inn_3.H2_X.detach()
    test_H2_X = inn_3.get_H2_X(case_data, inn_3.weight_1, inn_3.bias_1, inn_3.maskGene, inn_3.weight_2, inn_3.bias_2, inn_3.mask_layer_2)
    test_H2_X = test_H2_X.detach()
    input_data_SHAP=[]
    input_data_SHAP.append(H3_tensor)
    input_data_SHAP.append(test_H2_X)

    _sv_df, order_mean = get_Layer_ShapValue(model=inn_3,input_data_SHAP=input_data_SHAP,
                                             path_TF=True,feature=list(masks[1].columns),reactome_dict=reactome_dict)
    return _sv_df, order_mean   
# In[ ]:


# 4th layer shap value

def fourth_shap(model,masks,x_dataTensor,case_data,device,reactome_dict):

    data_tup=(x_dataTensor.to(device), model.weights[0], model.bias[0])
    H2_tup=(model.weights[1],model.bias[1])
    H3_tup=(model.weights[2],model.bias[2])

    inn_4=innModel_4(masks,device,data_tup,H2_tup,H3_tup).to(device)

    inn_4.weights[0] = model.weights[0]
    inn_4.bias[0] = model.bias[0]

    inn_4.weights[1] = model.weights[1]
    inn_4.bias[1] = model.bias[1]

    inn_4.weights[2] = model.weights[2]
    inn_4.bias[2] = model.bias[2]

    # inn_4.weights[3] = model.weights[3]
    # inn_4.bias[3] = model.bias[3]

    # inn_4.weights[4] = model.weights[4]
    # inn_4.bias[4] = model.bias[4]

    H4_tensor = inn_4.H3_X.detach()
    test_H3_X = inn_4.get_H3_X(case_data, inn_4.weight_1, inn_4.bias_1, inn_4.maskGene, inn_4.weight_2, inn_4.bias_2, inn_4.mask_layer_2,
                              inn_4.weight_3, inn_4.bias_3, inn_4.mask_layer_3)
    test_H3_X = test_H3_X.detach()
    input_data_SHAP=[]
    input_data_SHAP.append(H4_tensor)
    input_data_SHAP.append(test_H3_X)

    _sv_df, order_mean = get_Layer_ShapValue(model=inn_4,input_data_SHAP=input_data_SHAP,
                                             path_TF=True,feature=list(masks[2].columns),reactome_dict=reactome_dict)
    return _sv_df, order_mean   
# In[ ]:


# 5th


def fifth_shap(model,masks,x_dataTensor,case_data,device,reactome_dict):

    data_tup=(x_dataTensor.to(device), model.weights[0], model.bias[0])
    H2_tup=(model.weights[1],model.bias[1])
    H3_tup=(model.weights[2],model.bias[2])
    H4_tup=(model.weights[3],model.bias[3])

    inn_5=innModel_5(masks,device,data_tup,H2_tup,H3_tup,H4_tup).to(device)

    inn_5.weights[0] = model.weights[0]
    inn_5.bias[0] = model.bias[0]

    inn_5.weights[1] = model.weights[1]
    inn_5.bias[1] = model.bias[1]

    inn_5.weights[2] = model.weights[2]
    inn_5.bias[2] = model.bias[2]

    inn_5.weights[3] = model.weights[3]
    inn_5.bias[3] = model.bias[3]

    # inn_5.weights[4] = model.weights[4]
    # inn_5.bias[4] = model.bias[4]

    H5_tensor = inn_5.H4_X.detach()
    test_H4_X = inn_5.get_H4_X(case_data, inn_5.weight_1, inn_5.bias_1, inn_5.maskGene, inn_5.weight_2, inn_5.bias_2, inn_5.mask_layer_2,
                              inn_5.weight_3, inn_5.bias_3, inn_5.mask_layer_3,inn_5.weight_4, inn_5.bias_4, inn_5.mask_layer_4)
    test_H4_X = test_H4_X.detach()
    input_data_SHAP=[]
    input_data_SHAP.append(H5_tensor)
    input_data_SHAP.append(test_H4_X)

    _sv_df, order_mean = get_Layer_ShapValue(model=inn_5,input_data_SHAP=input_data_SHAP,
                                             path_TF=True,feature=list(masks[3].columns),reactome_dict=reactome_dict)
    return _sv_df, order_mean   

# In[ ]:


def sixth_shap(model,masks,x_dataTensor,case_data,device,reactome_dict):
    data_tup=(x_dataTensor.to(device), model.weights[0], model.bias[0])
    H2_tup=(model.weights[1],model.bias[1])
    H3_tup=(model.weights[2],model.bias[2])
    H4_tup=(model.weights[3],model.bias[3])
    H5_tup=(model.weights[4],model.bias[4])

    inn_6=innModel_6(masks,device,data_tup,H2_tup,H3_tup,H4_tup,H5_tup).to(device)

    inn_6.weights[0] = model.weights[0]
    inn_6.bias[0] = model.bias[0]

    inn_6.weights[1] = model.weights[1]
    inn_6.bias[1] = model.bias[1]

    inn_6.weights[2] = model.weights[2]
    inn_6.bias[2] = model.bias[2]

    inn_6.weights[3] = model.weights[3]
    inn_6.bias[3] = model.bias[3]

    inn_6.weights[4] = model.weights[4]
    inn_6.bias[4] = model.bias[4]

    H6_tensor = inn_6.H5_X.detach()
    test_H5_X = inn_6.get_H5_X(case_data, inn_6.weight_1, inn_6.bias_1, inn_6.maskGene, inn_6.weight_2, inn_6.bias_2, inn_6.mask_layer_2,
                              inn_6.weight_3, inn_6.bias_3, inn_6.mask_layer_3,inn_6.weight_4, inn_6.bias_4, inn_6.mask_layer_4,
                               inn_6.weight_5, inn_6.bias_5, inn_6.mask_layer_5)
    test_H5_X = test_H5_X.detach()
    input_data_SHAP=[]
    input_data_SHAP.append(H6_tensor)
    input_data_SHAP.append(test_H5_X)

    _sv_df, order_mean = get_Layer_ShapValue(model=inn_6,input_data_SHAP=input_data_SHAP,
                                             path_TF=True,feature=list(masks[4].columns),reactome_dict=reactome_dict)
    
    return _sv_df, order_mean   
    
   