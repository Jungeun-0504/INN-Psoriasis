# %%
import os
import pickle
import sys
sys.path.append('/centaurus/project/Playground/CT_Psoriasis/INN/src')
import util_INN_device as UT
import pandas as pd
from time import time,localtime,strftime
import argparse
import torch
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import copy
torch.set_num_threads(40) # set the number of threads for pytorch

import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix,matthews_corrcoef,roc_auc_score,f1_score
import numpy as np
import os 
import matplotlib.pyplot as plt
from collections import OrderedDict
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import random

# %%
parser = argparse.ArgumentParser(
                    prog='INN model training',
                    description='train model and write result',
                    epilog='model_type, lr, epoch, input_file, GPU_num, seed, save_path')


# parser.add_argument('-l', '--lr',type=float)
# parser.add_argument('-y', '--model_type',type=str)
# parser.add_argument('-e', '--epoch',type=int)
#parser.add_argument('-r', '--result',type=str)
#parser.add_argument('-i', '--input',type=str)
# parser.add_argument('-d', '--disease',type=str)
parser.add_argument('-g', '--GPU',type=int)
# parser.add_argument('-s', '--seed', type=int, default=False)
parser.add_argument('-p', '--path', type=str, default=False)
parser.add_argument('-n', '--number', type=str, default=False)

args = parser.parse_args()

# %%
path=args.path
input_dir= f"{path}/psoriasis/input/"
pathway = f"{path}/pathway/ReactomePathways.txt"
result_path = f"{path}/psoriasis/result/"

# %%
################################## Fix Me ##################################
config={
    "model":{
        "epoch": 500,
        "type": "batch_norm",
        "batch_size": 20,
        "rep_size": 10,
        "lr": [0.01,0.015,0.02,0.025,0.03],
        "dropout": [0.1,0.2,0.25,0.3],
        "GPU":args.GPU,
        "rep":10,
        "earlyStopping":15,
        "seed": 42},
    "result":{
        "log_interval":1,
        "cv_metrics":"cv_metrics",
        "shap":"shap_res/",
        "number":args.number
    }
}
############################################################################


# %%

# %%
device = torch.device(f'cuda:{config["model"]["GPU"]}')
torch.cuda.set_per_process_memory_fraction(0.5, device=device)
torch.cuda.empty_cache()
# %%

with open(os.path.join(input_dir,"test_X_Tensor.pickle"),"rb") as f:
    x_testTensor=pickle.load(f)
with open(os.path.join(input_dir,"test_y_Tensor.pickle"),"rb") as f:
    y_testTensor=pickle.load(f)
with open(os.path.join(input_dir,"lab_data.pickle"),"rb") as f:
    lab_data=pickle.load(f)
with open(os.path.join(input_dir,"input_data.pickle"),"rb") as f:
    input_data=pickle.load(f)
with open(os.path.join(input_dir,"masks.pickle"),"rb") as f:
    masks=pickle.load(f)
with open(os.path.join(input_dir,"split.pickle"),"rb") as f:
    split=pickle.load(f)

# %%
t_exp=input_data.T
#ac=torch.nn.Sigmoid()

# %%
reactome_data=pd.read_csv(pathway,sep="\t",header=None)
reactome_dict={}
for i,j in enumerate(reactome_data[0]):
    reactome_dict[j]=reactome_data[1][i]

# %%

# %%
def cal_metrics(model, pre, labeldata):
    
    # pre = model(xdata)
    pre = pre.cpu().detach().numpy()
    pre_label = np.where(pre >0.5, 1, 0)
    y_label = labeldata.cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_label, pre_label).ravel()
    f1 = f1_score(y_label, pre_label)
    mcc = matthews_corrcoef(y_label, pre_label)
    
    return (tn, fp, fn, tp), roc_auc_score(y_label, pre_label),f1, mcc,

# %%
def train_loop(dataloader,model,optimizer):
    # y_tr_total = torch.zeros(y_dataTensor.shape[0])
    # pred_tr_total = torch.zeros(y_dataTensor.shape[0])
    # num_bs=0
    num_batches = len(dataloader)
    model.train()
    loss_per_epoch=0
    
    for (inputs,targets) in dataloader:
        x_tr = inputs.to(device)
        y_tr = targets.to(device)
        prediction = model(x_tr.to(device))
        loss = criterion(prediction, y_tr.to(device))
        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()
        loss_per_epoch += loss.item()
        #y_tr_total[num_bs:num_bs+len(x_tr)]=y_tr.squeeze()
        #pred_tr_total[num_bs:num_bs+len(x_tr)]=prediction.squeeze()
        #num_bs=num_bs+len(x_tr)
        
    loss_per_batch=loss_per_epoch/num_batches
    #_, tr_AUC, tr_f1, tr_mcc=cal_metrics(model,pred_tr_total,y_tr_total)
    #tr_ACC=100 * cal_accuracy(model,x_dataTensor, y_dataTensor).item()
    # metrics={"ACC":ACC,
            #  "AUC":AUC,
            #  "f1":f1,
            #  "mcc":mcc}
    return loss_per_batch #, metrics

# %%
def test_loop(dataloader,model,is_test=False):
    global test_loader
    global valid_loader
    global device

    if is_test:
        y_val_total=torch.zeros(test_loader.dataset.y.shape[0])
        pred_val_total=torch.zeros(test_loader.dataset.y.shape[0])
        
    else:
        y_val_total=torch.zeros(valid_loader.dataset.y.shape[0])
        pred_val_total=torch.zeros(valid_loader.dataset.y.shape[0])
    model.eval()
    total_loss = 0
    num_bs=0
    for (inputs, targets) in dataloader:
        x_val = inputs.to(device)
        y_val = targets.to(device)
            
        output = model(x_val)
        loss = criterion(output, y_val)
        y_val_total[num_bs:num_bs+len(x_val)]=y_val.squeeze()
        pred_val_total[num_bs:num_bs+len(x_val)]=output.squeeze()
        num_bs=num_bs+len(x_val)
        total_loss += loss.item()
    
    _, AUC,f1, mcc=cal_metrics(model,pred_val_total,y_val_total)
    #ACC=100 * UT.cal_accuracy(model,x_validTensor, y_validTensor).item() # Acrracy쓰려면 코드 고쳐야함 지금은 validation기준으로만 작동
    metrics={#"ACC":[ACC],
            "AUC":[AUC],
            "f1":[f1],
            "mcc":[mcc]}
    
    return loss,metrics
    

# %%
def CV_data_prepare(data):
    x_dataTensor,y_dataTensor,x_validTensor,y_validTensor=data[0],data[1],data[2],data[3] 
    x_dataTensor = torch.nn.functional.normalize(x_dataTensor,1)
    x_dataTensor = x_dataTensor.to(device)
    y_dataTensor = y_dataTensor.to(device)

    x_validTensor = torch.nn.functional.normalize(x_validTensor,1)
    x_validTensor = x_validTensor.to(device)
    y_validTensor = y_validTensor.to(device)

    global x_testTensor,y_testTensor
    x_testTensor = torch.nn.functional.normalize(x_testTensor,1)
    x_testTensor = x_testTensor.to(device)
    y_testTensor = y_testTensor.to(device)

    train_dataset=UT.trLoader(
        x=x_dataTensor,
        y=y_dataTensor)

    valid_dataset=UT.trLoader(
        x=x_validTensor,
        y=y_validTensor)

    test_dataset=UT.trLoader(
        x=x_testTensor,
        y=y_testTensor)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["model"]["batch_size"],
        shuffle=True   
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config["model"]["batch_size"],
        shuffle=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config["model"]["batch_size"],
        shuffle=False
    )
    return train_loader,valid_loader,test_loader

# %%
def concat_dict(dict1,dict2):
    for key in dict1.keys():
        if isinstance(dict1[key], dict):
            for element in dict1[key].keys():
                dict1[key][element].extend(dict2[key][element])
        else:
            dict1[key].extend(dict2[key])
        
    return dict1

# %%
#new shap class 
class shap_value_2(nn.Module):
    def __init__(self, ori_model, masks, device, dropout, data_tup, layer):# layer는 1부터 시작
        super().__init__()
        self.ori_model=ori_model
        self.layer=layer-1
        self.ori_masks = masks
        self.masks = [torch.Tensor(m.values).to(device) for m in self.ori_masks[self.layer:]]
        self.weight= self.ori_model.weight[self.layer:].to(device) 
        self.bias= self.ori_model.bias[self.layer:].to(device)
        if self.layer==len(self.ori_model.weight): # layer=6 slf.layer=5
            self.batch_norm_layers=self.ori_model.batch_norm_layers[self.layer-1:].to(device)    
        else: # layer= 1~5 self.layer=0~4
            self.batch_norm_layers=self.ori_model.batch_norm_layers[self.layer:].to(device)
        # self.batch_norm_layers = nn.ModuleList().to(device)
        # for mask in self.masks[self.layer:]: 
        #     batch_norm_layer = nn.BatchNorm1d(mask.size(1))
        #     self.batch_norm_layers.append(batch_norm_layer)
        
        self.activation = torch.nn.Tanh()
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.input_X= data_tup[0]
        self.test_X= data_tup[1]
        #self.weight = nn.ParameterList([nn.Parameter(torch.randn(m.size(0), m.size(1))) for m in self.masks])
        # Initialize linear layers and BatchNorm layers
        #self.bias = nn.ParameterList([nn.Parameter(torch.randn(m.size(1))) for m in self.masks])

        self.maskGene= torch.FloatTensor(masks[0].values).to(device)
#        self.H1_X= self.get_H1_X()
    
    def get_layer_X(self):
        x=self.input_X
        for i in range(self.layer): # layer 0일 경우 안돌아감
            # print(i)
            # print(self.input_X.shape)
            # print(self.maskGene.shape)
            # print(self.ori_model.weight[i].shape)
            # print(self.ori_model.bias[i].shape)
            
            x = torch.matmul(x, (self.ori_model.weight[i]*self.ori_model.masks[i])) + self.ori_model.bias[i]
            x = self.ori_model.batch_norm_layers[i](x) 
            x = torch.tanh(x)
        return x
    
    def get_test_layer_X(self):
        x=self.test_X
        for i in range(self.layer): # layer 0일 경우 안돌아감
            x = torch.matmul(x, (self.ori_model.weight[i]*self.ori_model.masks[i])) + self.ori_model.bias[i]
            x = self.ori_model.batch_norm_layers[i](x) 
            x = torch.tanh(x)
        return x
    

    def forward(self, x): # keep 구현 어떻게...?
        if self.layer<4:
            #print(len(self.ori_model.weight)-self.layer-1)
            for i in range(len(self.ori_model.weight)-self.layer-1): #1일때 for문 4번
                #print(f"x={x.shape}")
                #print(f"weight={self.weight[i].shape}")
                x = torch.matmul(x, (self.weight[i] * self.masks[i])) + self.bias[i]
                #print(f"x={x.shape}")
                x = self.dropout(x)
                #print(i)
                # if i==0:
                #     keep = x
                # else:
                #     keep = torch.concat([keep, x], 1)
                #print(f"batch={self.batch_norm_layers[i]}")
                x = self.batch_norm_layers[i](x)  # Apply BatchNorm
                #x= torch.tanh(x)
            # print(x.shape)
            # print(self.weight[-1].shape)
            # print(self.masks[-1].shape)
            #   print(f"cal{i}")
        if self.layer<5:
            x = torch.matmul(x, (self.weight[-1] * self.masks[-1])) + self.bias[-1]
        
        x = self.batch_norm_layers[-1](x)
        #if self.layer<:
            # keep = torch.concat([keep, x], 1)
        
        res = self.aap(x)

        
        return res


# %%

# cv_matrics에서 AUC가장 높은 모델의 lr 가져오기 (추후에 batch_size,dropout도 추가)
cv_result_path=os.path.join(result_path,f"{config['result']['cv_metrics']}.csv")
new_cv_path=os.path.join(result_path,f"{config['result']['cv_metrics']}_{config['result']['number']}.csv")

cv_res_load=pd.read_csv(cv_result_path,index_col=0)
dr=cv_res_load.loc[cv_res_load["Valid AUC"]==max(cv_res_load["Valid AUC"])]["dropout"].values[0]
lr=cv_res_load.loc[cv_res_load["Valid AUC"]==max(cv_res_load["Valid AUC"])]["lr"].values[0]

# %%
use_gpu=True

#rep_result_path=os.path.join(result_path,f"{config['result']['rep_metrics']}_{config['result']['number']}.csv")

for rep in range(config["model"]["rep_size"]): # 50
    seed = random.randint(1, 40000) # training 동안 seed 고정
    UT.seed_everything(seed)
    for fold,data in enumerate(split):
        log = f"{str(config['model']['type'])}_{str(lr)}_{str(fold)}"
        print(log)
        train_loader,valid_loader,test_loader= CV_data_prepare(data)
        tm = localtime(time())
        model_file_name = strftime("%Y%m%d_%H%M%S", tm)
        model_file_name += f"_{log}"
        # model_file_name += f"{str(model_type)}_{str(lr)}_{str(fold)}"
        # path_base = os.path.join(result_path, model_file_name)
        # if not os.path.isdir(path_base):
        #     os.mkdir(path_base)
        #model_init
        model,criterion = UT.model_initial(masks,device,config["model"]["type"],dropout=dr)
        model=model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)     
        # model train, valid, test
        max_valid_acc=0
        max_test_acc=0
        switch_early_stop=False
        early_stopping = UT.EarlyStopping(patience = config["model"]["earlyStopping"], verbose = True,path=os.path.join(result_path,f"checkpoint_{config['result']['number']}.pt"))
        
        # epoch
        for i in range(config["model"]["epoch"]): # epoch
            print(f"epoch {i}")
            loss=train_loop(train_loader,model,optimizer)
            # if i%config["result"]["log_interval"]==0:
                # print(f"[train_loss] {loss}")
            valid_loss,valid_metrics=test_loop(valid_loader,model)
            # if i%config["result"]["log_interval"]==0:
            #     print(f"[valid loss] {valid_loss}")
            #     #print(f"[Epoch: {i}] validation Accuracy: {(valid_metrics['ACC'][0]):.2f}%")
            #     print(f"[Epoch: {i}] validation AUC: {valid_metrics['AUC'][0]}")
            #     print(f"[Epoch: {i}] validation F1: {valid_metrics['f1'][0]}")
            #     print(f"[Epoch: {i}] validation MCC: {valid_metrics['mcc'][0]}")
                
            test_loss,test_metrics=test_loop(test_loader,model,is_test=True)
            # if i%config["result"]["log_interval"]==0:
                # print(f"[test loss] {test_loss.item()}")
                # #print(f"[Epoch: {i}] test Accuracy: {test_metrics['ACC'][0]:.2f}%")
                # print(f"[Epoch: {i}] test AUC: {test_metrics['AUC'][0]}")
                # print(f"[Epoch: {i}] test F1: {test_metrics['f1'][0]}")
                # print(f"[Epoch: {i}] test MCC: {test_metrics['mcc'][0]}")
            
            metrics={"epoch":[i],"train":[loss],"valid":copy.deepcopy(valid_metrics),"test":copy.deepcopy(test_metrics)}
        #best model save
            if valid_metrics['AUC'][0]>0.7:
                switch_early_stop=True
            if (config["model"]["earlyStopping"]!=0)&(switch_early_stop==True):
                early_stopping(valid_metrics['AUC'][0], i, metrics, model) # 현재 과적합 상황 추적
                #early_stopping(valid_metrics['AUC'][0], i, metrics, model) # 현재 과적합 상황 추적
                if early_stopping.early_stop: # 조건 만족 시 조기 종료
                    max_metrics=early_stopping.best_metrics
                    max_epoch=early_stopping.best_epoch
                    break
                elif i==config["model"]["epoch"]-1:
                    max_metrics=metrics
                    max_epoch=i
                    
            elif switch_early_stop==False:
                if max_valid_acc<valid_metrics['AUC'][0]:
                    max_valid_acc=valid_metrics['AUC'][0]
                    max_test_acc=test_metrics['AUC'][0]
                    max_metrics=metrics
                    max_epoch=i
                    if max_test_acc<test_metrics['AUC'][0]:
                        max_test_acc=test_metrics['AUC'][0]
                        max_metrics=metrics
                        #if os.path.isfile(f"{path_base}/best_model.pt"):
                            #os.rename(f"{path_base}/best_model.pt", f"{path_base}/second_model.pt")
                        torch.save(model,os.path.join(result_path,f"checkpoint_{config['result']['number']}.pt"))
                    else:
                        max_test_acc=max_test_acc
                elif max_valid_acc==valid_metrics['AUC'][0]:
                    max_valid_acc=max_valid_acc
                    max_test_acc=test_metrics['AUC'][0]
                    if max_test_acc<test_metrics['AUC'][0]:
                        max_test_acc=test_metrics['AUC'][0]
                        max_metrics=metrics
                        #if os.path.isfile(f"{path_base}/best_model.pt"):
                            #os.rename(f"{path_base}/best_model.pt", f"{path_base}/second_model.pt")
                        torch.save(model,os.path.join(result_path,f"checkpoint_{config['result']['number']}.pt"))
                    else:
                        max_test_acc=max_test_acc
                else:
                    pass
            
        shap_path=os.path.join(result_path,config["result"]["shap"])
        if not os.path.isdir(shap_path):
            os.mkdir(shap_path)
            
        model=torch.load(os.path.join(result_path,f"checkpoint_{config['result']['number']}.pt"))
        data_tup=(train_loader.dataset.x, test_loader.dataset.x)
        for i in range(1,len(model.weight)+2):
            print(f"shap_layer : {i}")
            shap_model=shap_value_2(model, masks, device, 0.2, data_tup, i)
            # print(shap_model.weight)
            # print(shap_model.batch_norm_layers)
            shap_train_input = shap_model.get_layer_X()
            shap_test_input = shap_model.get_test_layer_X()
            input_data_SHAP=[]
            input_data_SHAP.append(shap_train_input)
            input_data_SHAP.append(shap_test_input)
            _sv_df, order_mean = UT.get_Layer_ShapValue(model=shap_model,input_data_SHAP=input_data_SHAP,reactome_dict=reactome_dict)
            order_mean.sort_values(by="name", inplace=True)
            order_mean.reset_index(drop=True, inplace=True)
            if not os.path.exists(os.path.join(shap_path,f"shap_value_{i}_{config['result']['number']}.csv")):
                print("dataframe new write and save")
                order_mean.to_csv(os.path.join(shap_path,f"shap_value_{i}_{config['result']['number']}.csv"),sep=",",index=False)
            
            else:
                print("dataframe open and save")
                order_mean_load=pd.read_csv(os.path.join(shap_path,f"shap_value_{i}_{config['result']['number']}.csv"))
                # Perform a single merge operation outside the loop
                order_mean = pd.concat([order_mean_load,order_mean["value"]], axis=1)
                col_list=[]
                for col in range(len(order_mean.columns)-1):
                    col_list.append(f"value_{col}")
                order_mean.columns=["name"]+col_list
                order_mean.to_csv(os.path.join(shap_path,f"shap_value_{i}_{config['result']['number']}.csv"),sep=",",index=False)
            
            #UT.shap_figure(_sv_df,order_mean,shap_path,i)
        
        # fold statistics save
        if fold==0:
            cv_res_dict=copy.deepcopy(max_metrics)
        elif fold==(len(split)-1):
            # best performance model metrics save

            final_res_dict=OrderedDict([
                
                #("Validation acc",cv_res_dict["valid"]["ACC"]),
                #("Test acc",cv_res_dict["test"]["ACC"]), # accuracy 쓰려면 코드 고치기 -> cal_accuracy함수 수정해야함
                
                ("Valid AUC",cv_res_dict["valid"]["AUC"]),
                ("Test AUC",cv_res_dict["test"]["AUC"]),
                
                ("Valid F1",cv_res_dict["valid"]["f1"]),
                ("Test F1",cv_res_dict["test"]["f1"]),
                
                ("Valid MCC",cv_res_dict["valid"]["mcc"]),
                ("Test MCC",cv_res_dict["test"]["mcc"]) 
            ])
            
            df=pd.DataFrame(final_res_dict)
            mean_cv=df.mean().to_frame().T
            info_state=pd.DataFrame({"lr":[lr],"dropout":[dr],"seed":[seed]})
            lr_mean_cv=pd.concat([info_state,mean_cv],axis=1)
            
            if not os.path.exists(new_cv_path):
                print("metrics new write and save")
                lr_mean_cv.to_csv(new_cv_path)
            
            else:
                print("metrics open and save")
                cv_res_load=pd.read_csv(new_cv_path,index_col=0)
                lr_mean_cv=pd.concat([cv_res_load,lr_mean_cv])
                lr_mean_cv.to_csv(new_cv_path)
                
        else: # fold!=0 and fold!=(len(split)-1)
            cv_res_dict=concat_dict(cv_res_dict,max_metrics)
        
