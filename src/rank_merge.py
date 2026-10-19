# %%
import pandas as pd
import pickle as pickle
import os
from sklearn.preprocessing import minmax_scale
import argparse

# %%

parser = argparse.ArgumentParser(
                    prog='INN model training',
                    description='train model and write result',
                    epilog='input_file')

parser.add_argument('-ip', '--input_path',type=str)
parser.add_argument('-is', '--input_string',type=str)

args = parser.parse_args()

input_path=args.input_path
input_str=args.input_string
res_dir=f"/{input_path}/shap_res"

# %%
res_dict=pickle.load(open(os.path.join(res_dir,"value_merge.pickle"),"rb"))

# %%
for i in range(1,7):
    res_dict[i]["value"]=res_dict[i].iloc[:,1:].astype(float).mean(axis=1).sort_values(ascending=False)
    res_dict[i]=res_dict[i].iloc[:,[0,-1]]


# %%
################################## Fix Me ##################################
input_base_path="/storage/seojiwon/INN_Psoriasis/INN_edit/"

reactome="/storage/seojiwon/INN_Psoriasis/INN_edit/pathway/ReactomePathways.txt"
############################################################################

reactome_data=pd.read_csv(reactome,sep="\t",header=None)
reactome_dict={}
for i,j in enumerate(reactome_data[0]):
    reactome_dict[j]=reactome_data[1][i]

temp_dict={"group":reactome_dict.keys(),"description":reactome_dict.values()}

all_path=pd.DataFrame(temp_dict)
values = {}
for i in range(1,7):
    #values[i-1]=pd.read_csv(os.path.join(res_dir,f"shap_value_{i}.csv"))
    values[i-1]=res_dict[i].sort_values(by=['value'],ascending=False)
    values[i-1]['rank']=res_dict[i].rank(method="min",ascending=True)['value']


# %%
all_path=all_path.loc[all_path["group"].str.startswith("R-HSA-")]

# %%
mask_path=os.path.join(os.path.join(input_base_path,input_str),"masks.pickle")

with open(mask_path, 'rb') as f:
    masks = pickle.load(f)
    
for i in range(1,6):
    values[i] = pd.merge(values[i], all_path, left_on='name',right_on='description', how='left')[['group','value','rank']]    

for i in range(4,-1,-1): #4 3 2 1 0
    melt_temp = pd.melt(masks[i],ignore_index=False)
    melt_temp = melt_temp[melt_temp.value==1].reset_index()
    melt_temp.columns = [f'{i}_layer',f'{i+1}_layer', '-']
    melt_temp = melt_temp.iloc[:,:-1]
    values[i].columns = ['group',f'{i}_value', f'{i}_rank']
    melt_temp = pd.merge(melt_temp,values[i], left_on=f'{i}_layer', right_on='group', how='left')
    melt_temp = melt_temp.drop(columns='group')
    if i==4:
        values[i+1].columns = ['group',f'{i+1}_value', f'{i+1}_rank']
        melt_temp = pd.merge(melt_temp,values[i+1], left_on=f'{i+1}_layer', right_on='group', how='left')
        melt_temp = melt_temp.drop(columns='group')
        
        df_total = melt_temp
    else:
        df_total = pd.merge(df_total, melt_temp, on=f'{i+1}_layer', how='left')



for i in range(0,6):
    df_total[f'{i}_value_norm'] = minmax_scale(df_total[f'{i}_value'])
    df_total[f'{i}_rank_norm'] = minmax_scale(df_total[f'{i}_rank'])

df_total = df_total.reindex(sorted(df_total.columns),axis=1)
df_total = df_total.dropna().reset_index()


# %%
for k in range(1,6):
    for i, j in enumerate(df_total[f"{k}_layer"]):
    #     print(j)
        df_total.loc[i,f'{k}_layer']=reactome_dict[j]

# %%
df_total.to_csv(os.path.join(res_dir,"rank_merge.csv"),index=False)

# %%



