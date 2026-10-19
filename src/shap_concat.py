# %%
import pandas as pd
import numpy as np
import os
import pickle
import argparse

parser = argparse.ArgumentParser(
                    prog='INN model training',
                    description='train model and write result',
                    epilog='model_type, lr, epoch, input_file, GPU_num, seed, save_path')

parser.add_argument('-i', '--inputname',type=str)

args = parser.parse_args()

# Importing the dataset

# %%

# Importing the dataset
shap_path=f"/{args.inputname}/shap_res/"

total_file=os.listdir(shap_path)

layer_df = {}

for layer in range(1, 7):
    print(f"layer: {layer}")
    matching_files = [file for file in total_file if file.startswith(f'shap_value_{layer}_') and file.endswith('.csv')]

    # Use list comprehension to read and store DataFrames in a dictionary
    dfs = {j + 1: pd.read_csv(os.path.join(shap_path, file)) for j, file in enumerate(matching_files)}
    for i in range(1, len(dfs)+1):
        dfs[i].sort_values(by="name", inplace=True)
        dfs[i].reset_index(drop=True, inplace=True)
    # Perform a single merge operation outside the loop
    merged_df = pd.concat(dfs.values(), axis=1)
    merged_df.index=merged_df.iloc[:,0]
    merged_df.drop(columns=["name"],inplace=True)
    temp_name=pd.DataFrame(merged_df.index)
    temp_name.reset_index(drop=True,inplace=True)
    merged_df.reset_index(drop=True,inplace=True)
    merged_df=pd.concat([pd.DataFrame(temp_name),merged_df],axis=1,ignore_index=True)
    col_list = [f"value_{col}" for col in range(len(merged_df.columns) - 1)]
    merged_df.columns = ["name"] + col_list

    # Store the merged DataFrame in the dictionary
    layer_df[layer] = merged_df

with open(os.path.join(shap_path,"value_merge.pickle"), 'wb') as handle:
    pickle.dump(layer_df, handle, protocol=pickle.HIGHEST_PROTOCOL)


