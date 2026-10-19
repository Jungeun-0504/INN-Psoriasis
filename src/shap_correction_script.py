import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import argparse

args=argparse.ArgumentParser()
args.add_argument('-i',"--input_path",type=str, help="input path") 

args2= args.parse_args()
path = args2.input_path

merge_path=f"{path}/psoriasis/result/shap_res/value_merge.pickle"
with open(merge_path,'rb') as f:
    shap_dict = pickle.load(f)

reactome=f"{path}/pathway/ReactomePathways.txt"

reactome_data=pd.read_csv(reactome,sep="\t",header=None)
reactome_dict={}
for i,j in enumerate(reactome_data[0]):
    reactome_dict[j]=reactome_data[1][i]

temp_dict={"group":reactome_dict.keys(),"description":reactome_dict.values()}

all_path=pd.DataFrame(temp_dict)

with open(f"{path}/psoriasis/input/masks.pickle", 'rb') as f:
    masks = pickle.load(f)


all_path.index=all_path.group

for i in range(5):
    melt_temp = pd.melt(masks[i],ignore_index=False)
    melt_temp = melt_temp[melt_temp.value==1].reset_index()
    melt_temp.columns = [f'{i}_layer',f'{i+1}_layer', '-']
    melt_temp = melt_temp.iloc[:,:-1]
    if i==0:
        _total = melt_temp
    else:
        _total = pd.merge(_total, melt_temp, on=f'{i}_layer', how='right')
    
for i in range(1,6):
    _total[f'{i}_layer']=_total[f'{i}_layer'].replace(all_path.to_dict()['description'])    

_total.to_csv(f"{path}/psoriasis/result/shap_res/total_pathway_info.csv",index=False)
all_path=pd.read_csv(f"{path}/psoriasis/result/shap_res/total_pathway_info.csv")
all_path=_total

from scipy import stats

class node_degree:
    def __init__(self, df_dict, layer: int, pathway: pd.DataFrame):
        
        self.df=df_dict
        self.layer=layer
        self.node=list(self.df[self.layer]['name'])
        self.value=np.zeros(len(self.node))
        self.pathway=pathway
        self.value=self.df[self.layer]
        #for j, i in enumerate(self.node):
            
            # self.value[j]=self.df.loc[self.df[f"{self.layer}_layer"]==i][f"{self.layer}_value"].mean()
        
        
    def get_degree(self):
        _input_list=np.zeros(len(self.node))
        _output_list=np.zeros(len(self.node))
        _degree_list=np.zeros(len(self.node))
        for j,i in enumerate (self.node):
            _sub=self.pathway.loc[self.pathway[f"{self.layer-1}_layer"]==i]# self.layer=2 layer -> 1layer
            if self.layer==1:
                _output_list[j]=len(_sub[f"{self.layer}_layer"].unique())
                _degree_list[j]=_output_list[j]
            elif (self.layer>1)&(self.layer<6):
                _input_list[j]=len(_sub[f"{self.layer-2}_layer"].unique())
                _output_list[j]=len(_sub[f"{self.layer}_layer"].unique())
                _degree_list[j]=_input_list[j]+_output_list[j]
            elif self.layer==6:
                _input_list[j]=len(_sub[f"{self.layer-2}_layer"].unique())
                _degree_list[j]=_input_list[j]
        return self.node, _input_list,_output_list,_degree_list
    
    def get_total_connection(self):
        _con_list=np.zeros(len(self.node))
        for j,i in enumerate (self.node):
            _sub=self.pathway.loc[self.pathway[f"{self.layer-1}_layer"]==i]
            _con_list[j]=_sub.shape[0]
        return _con_list
    
    def get_cor(self,degree_list):
        test_df=pd.DataFrame({"node":self.node,"cal_degree":degree_list})
        t_shap=self.df[self.layer].T.iloc[1:,:]
        t_shap.columns=self.df[self.layer]["name"]
        value_degree=pd.melt(t_shap)
        value_degree=pd.merge(test_df,value_degree,left_on="node",right_on="name",how="right")
        value_degree.drop("name",axis=1,inplace=True)
        correlation, p_value = stats.pearsonr(value_degree["value"], value_degree["cal_degree"])
        return correlation, p_value

dict_cor={}
for i in range(1,7):
    cl_node_degree=node_degree(shap_dict,i,_total)
    node_list, input_list,output_list,degree_list=cl_node_degree.get_degree()
    connection=cl_node_degree.get_total_connection()
    print(f"layer:{i}")
    print(f"degree:{cl_node_degree.get_cor(degree_list)}")
    print(f"input:{cl_node_degree.get_cor(input_list)}")
    print(f"output:{cl_node_degree.get_cor(output_list)}")
    print(f"total:{cl_node_degree.get_cor(connection)}")
    dict_cor[i]={"fan_in": cl_node_degree.get_cor(input_list),
                "fan_out": cl_node_degree.get_cor(output_list),
                "degree": cl_node_degree.get_cor(degree_list),
                "total_connect":cl_node_degree.get_cor(connection),
                "std":{}}
    
    for j in range(1,7):
        try:
            _value=cl_node_degree.value[degree_list>np.mean(degree_list)+j*(np.std(degree_list))]
            _degree=degree_list[degree_list>np.mean(degree_list)+j*(np.std(degree_list))]
            test_df=pd.DataFrame({"node":_value["name"],"cal_degree":_degree})
            t_shap=_value.T.iloc[1:,:]
            t_shap.columns=_value["name"]
            value_degree=pd.melt(t_shap)
            value_degree=pd.merge(test_df,value_degree,left_on="node",right_on="name",how="right")
            value_degree.drop("name",axis=1,inplace=True)
            _res = stats.pearsonr(value_degree["value"], value_degree["cal_degree"])
            print(f"mean_{j}_std:{_res[0]}")
            dict_cor[i]["std"][j]=_res[0]
        except:
            print("error")
#이 다음 NA처리

for i in range(1,7):
    for j in range(1,7):
        try:
            print(dict_cor[i]["std"][j])
        except:
            dict_cor[i]["std"][j]=np.nan


df_cor_ori=pd.DataFrame()
for i in range(1,7):
    df_cor=pd.DataFrame([dict_cor[i]['fan_in'][0],
                        dict_cor[i]['fan_out'][0],
                        dict_cor[i]['degree'][0],
                        dict_cor[i]['total_connect'][0],
                        dict_cor[i]['std'][1],
                        dict_cor[i]['std'][2],
                        dict_cor[i]['std'][3],
                        dict_cor[i]['std'][4],
                        dict_cor[i]['std'][5],
                        dict_cor[i]['std'][6]],
                        index=["fan_in","fan_out","degree","total_connect","std1","std2","std3","std4","std5","std6"],columns=[f"{i}_layer"])
    df_cor_ori=pd.concat([df_cor_ori,df_cor],axis=1)
    
df_cor_ori.columns=["1_layer","2_layer","3_layer","4_layer","5_layer","6_layer"]
df=df_cor_ori.fillna(0)
df=df.replace("Nan",0)


# shap value 보정 correlation 0.5이상인 것만

for i in range(1,7):
    print(i)
    for j in range(1,7):
        if abs(dict_cor[i]["std"][j])>0.5:
            print(f"correction_{i}_layer_{j}_std")
            cl_node_degree=node_degree(shap_dict,i,_total)
            node_list, input_list,output_list,degree_list=cl_node_degree.get_degree()
            
            cor_value=cl_node_degree.value[degree_list>np.mean(degree_list)+j*(np.std(degree_list))]
            cor_list=[]
            for item in cor_value["name"]:
                if item in node_list:
                    index = node_list.index(item)
                    cor_list.append(index)

            cor_degree=degree_list[cor_list]
            for k,name in enumerate(cor_value["name"]):
                selected_row=shap_dict[i].loc[shap_dict[i]["name"]==name].index.values
                shap_dict[i].iloc[selected_row,1:]=shap_dict[i].iloc[selected_row,1:]/cor_degree[k]
            
            break

save_path=f"{path}/psoriasis/result/shap_res/value_merge_corrected.pickle"
with open(save_path,'wb') as f:
    pickle.dump(shap_dict,f)
