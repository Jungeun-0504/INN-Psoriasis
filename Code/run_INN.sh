#!/usr/bin/sh

### conda env setting ###
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate INN_env


gpu=(0 1)
num=0

################################## Fix Me ##################################
outpath=/centaurus/project/Playground/CT_Psoriasis/INN
############################################################################

input=${outpath}/psoriasis
result=${outpath}/psoriasis/result
scriptpath=${outpath}/src/


train_script=${scriptpath}'model_training_ver3_gpu_split.py'
shap_cal_script=${scriptpath}'shap_cal_gpu_split.py'
shap_concat_script=${scriptpath}'shap_concat.py'
shap_correction_script=${scriptpath}'shap_correction_script.py'

echo "python -u ${train_script} -g ${gpu[0]} -p ${outpath} > ${input}/output/psoriasis_training.log"
python -u ${train_script} -g ${gpu[0]} -p ${outpath} > ${input}/output/psoriasis_training.log

mkdir -p ${result}/shap_res/
echo "calculation shap value start"	
for g in ${gpu[@]}		
do  
    echo "python -u ${shap_cal_script} -g $g -n ${num} -p ${outpath} > ${result}/calculation_psoriasis_${num}.log 2>&1 &"
    python -u ${shap_cal_script} -g $g -n ${num} -p ${outpath} > ${result}/calculation_psoriasis_${num}.log 2>&1 &
    num=$((num+1))
done
wait

echo "python -u ${shap_concat_script} -i ${result}"
python -u ${shap_concat_script} -i ${result}

echo "python ${shap_correction_script} -i ${outpath}"
python ${shap_correction_script} -i ${outpath}

