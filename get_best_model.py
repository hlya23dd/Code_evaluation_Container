import argparse
import numpy as np

import sys
import os, datetime
import errno
import scipy.io as sio  #for mat file

#import matplotlib.pyplot as plt
import timeit


############################################################
# make a directory according to time
############################################################

mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d'))
try:
    os.makedirs(mydir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error..


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--patient', help='Patient number, 1 to 15 is available', type=int, default=1)
parser.add_argument('-mns', '--model_ensemble_size', help='Size of model ensemble tested for each feature combination', type=int, default=20)
patient_index_range=args.patient
pat='patient'+str(patient_index)

i_auc_criterion = 0

n_uvar=23
n_mvar=18
N_model =n_uvar*n_mvar
n_model_ensemble =args.model_ensemble_size


roc_valid=np.zeros(N_model*n_model_ensemble) 
pr_valid=np.zeros(N_model*n_model_ensemble) 
roc_train=np.zeros(N_model*n_model_ensemble) 
pr_train=np.zeros(N_model*n_model_ensemble) 

feature_combination=[] 
model_names=[] 

with open('auc_best_patient'+pat+'.dat') as f:
       lines = f.readlines()

for kk in range(N_model*n_model_ensemble): 
    roc_train[kk]= lines[kk].split()[1]
    pr_train[kk]= lines[kk].split()[2]
    roc_valid[kk]= lines[kk].split()[3]
    pr_valid[kk]= lines[kk].split()[4]

rocpr_valid_mean=np.zeros(N_model) 

for i in range(n_mvar):
    for j in range(n_uvar):
       k= i*n_mvar +j
       ROC_train=roc_train[k*n_model_ensemble:(k+1)*n_model_ensemble]
       PR_train=roc_train[k*n_model_ensemble:(k+1)*n_model_ensemble]
       ROC_valid=roc_train[k*n_model_ensemble:(k+1)*n_model_ensemble]
       PR_valid=roc_train[k*n_model_ensemble:(k+1)*n_model_ensemble]
   
       ROC_valid=ROC_valid[ROC_train*PR_train > 0.7]
       PR_valid=PR_valid[ROC_train*PR_train > 0.7]
       rocpr_valid_mean[k]=np.sum(ROC_valid*PR_valid)/len(ROC_valid)

idx_valid_best=rocpr_valid_mean.argsort()[::-1][0]

uvars =['c','e','E','p','y','cE','ce','cp','cy','ey','Ey','ep','Ep','py','cpe','cpE','epy','Epy','cey','cEy','cpy','cepy','cEpy']
mvars=['mpcH','mpcR','lincoh','corrfAmp','corrfPha','corr']
fea_levels=['a','b','ab']
opt_level=fea_levels[int(int(idx_valid_best/23)/6)]
opt_uvar=uvars[idx_valid_best%23]
opt_mvar=mvars[int(idx_valid_best/23)]

opt_model_name='grid_search/'+pat+'_'+opt_uvar+'_'+opt_mvar+'_'+opt_leve+'_mlp/'+pat+'_best.hd5'

print(rocpr_valid_mean[idx_valid_best],opt_model_name)

cwd = os.getcwd()

with open("Best_model.py", "w") as myfile:
    myfile.write('feature_select='+opt_uvar+'\n')      
    myfile.write('mvar_feature_select='+opt_mvar+'\n')      
    myfile.write('feature_level='+opt_level+'\n')      
    myfile.write('model_file_name='+opt_model_name+'\n')      

stop = timeit.default_timer()


