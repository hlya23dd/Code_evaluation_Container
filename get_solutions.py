import argparse
import numpy as np


import sys
import os, datetime
import errno
import scipy.io as sio  #for mat file

import timeit
import pandas as pd

from AR_v160_svm import *

############################################################
parser = argparse.ArgumentParser()
# Algorithm settings
parser.add_argument('-b', '--batch_size', help="batchsize", type=int, default=800)
parser.add_argument('-e', '--epochs', help="training epochs", type=int, default=500)
parser.add_argument('-v', '--verbose', help="keras verbosity", type=int, default=0, choices=[0, 1, 2])
parser.add_argument('-mf', '--model_file', help="Path to stored model file for model evaluation. "
                                                "If not specified, trained model for respective patient is expected in "
                                                "current working directory", default=None)

# Evaluation settings
parser.add_argument('-csv', '--path', help='path to the csv that includes the files', default='/CSVs')
parser.add_argument('-m', '--mode', help='Mode. 0: feature generation, Mode. 1: training, 2: validation, 3: test', type=int, default=-1,
                    choices=[-1, 0, 1, 2, 3])
parser.add_argument('-p', '--patient', help='Patient number, 1 to 15 is available', type=int, default=1)
parser.add_argument('-l', '--file_segment_length', help='Segment length in minutes, 1 or 10', type=int, default=10)
parser.add_argument('-sm', '--subtract_mean', help='Subtract channelwise mean of each file', type=int, default=1,
                    choices=[0, 1])

parser.add_argument('-ns', '--n_start', help='Starting index of segments to calculate features for', type=int, default=-1)
parser.add_argument('-ne', '--n_end', help='End index of segments to calculate features for', type=int, default=-1)
parser.add_argument('-mns', '--model_ensemble_size', help='Size of model ensemble tested for each feature combination', type=int, default=20)
parser.add_argument('-uf', '--uvar_fea', help='Univariate feature used', default='', choices=['c','e','E','p','y','cE','ce','cp','cy','ey','Ey','ep','Ep','py','cpe','cpE','epy','Epy','cey','cEy','cpy','cepy','cEpy'])
parser.add_argument('-bf', '--bvar_fea', help='bivariate feature used', default='', choices=['_mpcH_','_mpcR_','_lincoh_','_corrfAmp_','_corrfPha_','_corr_'])
parser.add_argument('-bv', '--bvar_var', help='variant of bivariate feature used', default='a', choices=['a','b','ab'])
parser.add_argument('-nt', '--n_train', help='For grid search of best feature combination. The train set is split into train and valid subsets. It is assumed that the first [n_train] segments are the train subset and the rest is for validation. If not specified the first 2/3 will be used for train.', type=int, default=None)
parser.add_argument('-wm', '--which_model', help='Which model should be used for train set. 0: best model from grid search, 1: retrain for the train set', type=int, default=0, choices=[0, 1])

############################################################
args = parser.parse_args()
print(args)

patient_index=args.patient
 
segment_length_minutes=args.file_segment_length

mode=args.mode

pat='patient'+str(patient_index)

which_model=parser.which_model
############################################################

Seer_name='hlya23dd'

num_channel=16
Num_neigh=1  # 1 channel prediction, univariant
time_delay=3 # order of AR model
poly_order=1 # polynomial order
regression_mode = 0 # 0:   model2 = LinearRegression(fit_intercept = False)

n_noisy=1 # 1 is used to show the noise part can be modeled as well
i_low_pass=1 # surrogate
F_threshold=0.0 # [4, 8, 12, 30] frequency threshold for filtering the EEG signal, f_threshold*noise_level = 0
#noise strength for generating surrogate signal, f_threshold*noise_level = 0
Noise_level=0.0 # do not change this, only if you want to add noise
# to randomize phase while keeping power spectrum, please set f_range to negative values !!! 

import os
cwd = os.getcwd()
feature_dir=cwd+'/features/'
i_standardization = -1 # no normalization

###############################
feature_select=args.uvar_fea
dim_feature_channelwise=14


mvar_feature_lib=['mpcH','mpcR','lincoh','corrfAmp','corrfPha','corr']
mvar_feature_range =['_mpcH_','_mpcR_','_lincoh_','_corrfAmp_','_corrfPha_','_corr_']

mvar_feature_select= args.bvar_fea

feature_level = args.bvar_var

feature_band_range = ['bandYang']

###############################
# mlp parameters

batch_size=args.batch_size
epochs=args.epochs

kernel_constraint_weight=0

i_feature_selection=0

hidden_layer_size=[16,8,4]

verbose =args.verbose # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

i_class_weight = 'balanced'

i_sample_weight =1
 
my_class_1_weight =100
###############################
from Best_model import *
#feature_select='+opt_uvar+'\n')      
#mvar_feature_select='+opt_mvar+'\n')      
#feature_level='+opt_level+'\n')      
#model_file_name='+opt_model_name+'\n')      
###############################

if mode <= 1: # train & feature_generation
   segment_file_name_lables='/CSVs/train_filenames_labels_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'
if mode == 2: # valid
   segment_file_name_lables='/CSVs/validation+_filenames_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'
if mode == 3: # test
   segment_file_name_lables='/CSVs/test_filenames_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'

df = pd.read_csv(segment_file_name_lables)
n_files=len(df)
segment_fnames=df['image']
i_file_range=range(n_files)

if mode == 1: # train
   labels=df['class']
else:
   labels=np.zeros(n_files)

for i_mat_segment in i_file_range:
    channel_range=range(0,num_channel) # univarinat
    pat_gData,pat_gLabel = eeg_pca_ica_mat_segment_Melborne(segment_fnames[i_mat_segment],labels[i_mat_segment],segment_length_minutes)
    print(pat_gData.shape)
    if pat_gData.shape[0] < 1:
       continue

############################################################
# generate uvar
#####

    if True:

        ar_poly_input={'poly_order': poly_order, 
        'time_delay': time_delay, 
        'regression_mode': regression_mode, 
        'num_neigh': Num_neigh,
        'pat_gData': pat_gData,
        'pat_gLabel': pat_gLabel, 
        'f_threshold': F_threshold,
        'n_noisy': n_noisy,
        'i_low_pass': i_low_pass,
        'noise_level': Noise_level
         }

        my_feature, my_label = ar_ps(ar_poly_input)

        print(my_feature.shape)
    dim_feature_channelwise=int(my_feature.shape[1]/num_channel)
    mask=get_feature_mask(feature_select,time_delay,dim_feature_channelwise)
    mask=np.asarray(mask)
    dim_feature=mask.sum()
    my_feature = (my_feature.T[mask>0.5]).T

    if sum(mask)>0:    
     feature_flag=1
     print(my_feature.shape)
    else:
     feature_flag=0

####
# get mvar
####
    pat_gData=pat_gData*1.0  # pat_gData seems to be integer, this cause trouble, 2018.11.28

    for feature, feature_function in zip(mvar_feature_lib, mvar_feature_function_lib):
           feature_ = '_'+feature+'_'
           if feature_ in mvar_feature_select:
                  print(feature)
                  pat_gData2=pat_gData.copy()*1.0

                  exec("dict_return = "+ feature_function)   
                  exec(feature+'_m' + " = dict_return['corr_y']")   
                  exec(feature+'_max' + " = dict_return['corr_max']")   
                  exec(feature+'_eva' + " = dict_return['e_values']")   
                  exec(feature+'_eve' + " = dict_return['e_vector']")   
                  exec(feature+'b' + " = np.c_["+feature+'_eva,' +feature+'_eve,' +feature+"_max]")   
                  exec(feature+'a' + " = "+feature+'_m')   
      
                  if feature_flag==0:
                     feature_flag=1
                     if 'a' in feature_level:
                        exec("my_feature ="+feature+'a')
                     if 'b' in feature_level:
                        exec("my_feature ="+feature+'b')
                     if 'ab' in feature_level:
                        exec("my_feature =np.c_["+feature+'a,'+feature+'b]')
                  else:
                     if 'a' in feature_level:
                        exec("my_feature =np.c_[my_feature,"+feature+'a]')
                     if 'b' in feature_level:
                        exec("my_feature =np.c_[my_feature,"+feature+'b]')

    print("feature calculation is done!")

    print(my_feature.shape)

############################################################
# collect features for different segments 
    print('feature collection')

    if i_mat_segment == i_file_range[0]:
       my_label_all = pat_gLabel
       my_feature_all=my_feature              
        
    else:
       my_label_all = np.r_[my_label_all,pat_gLabel]
       my_feature_all=np.r_[my_feature_all,my_feature]

####################################################### 2018-12-5
  
# replace NaN of samples
my_feature_all = feature_NaN(my_feature_all,my_feature_all.shape[1])
my_feature_all = feature_inf(my_feature_all,my_feature_all.shape[1])
    # mean centralization & standalization

if i_standarization >0:  
    for i_feature in range(my_feature_all.shape[1]):  
      yy=my_feature_all[:,i_feature]
      yy= yy-sum(yy)/len(yy)  # mean centralization
      if sum(yy*yy)>0:
         yy= yy/np.sqrt(sum(yy*yy)/len(yy))  # normalization
      my_feature_all[:,i_feature]=yy

######################################################

start = timeit.default_timer()


###
# mlp-prb
###
if which_model==0:
   roc_auc3, pr_auc3, test_label_10, probas2_10, model_file_name = keras_mlp_10m_prb_oldmodel_test(my_feature_all, my_label_all, model_file_name)
elif mode==1:

      n_1=sum(my_label_all)
      n_0=len(my_label_all)-n_1
      class_1_weight=n_0/n_1

      reset_keras() 
      roc_auc, roc_auc3, pr_auc, pr_auc3, model_file_name = keras_mlp_10m_prb(my_feature_all, my_label_all, my_feature_all, my_label_all, mydir, pat, segment_length_minutes, class_1_weight, batch_size, epochs, kernel_constraint_weight, verbose, hidden_layer_size, i_feature_selection)
      roc_auc3, pr_auc3, test_label_10, probas2_10, model_file_name = keras_mlp_10m_prb_oldmodel_test(my_feature_all, my_label_all, model_file_name)
      with open("Train_model.py", "w") as myfile:
           myfile.write('model_file_name='+model_file_name)       
else:
   from Train_model import *
   roc_auc3, pr_auc3, test_label_10, probas2_10, model_file_name = keras_mlp_10m_prb_oldmodel_test(my_feature_all, my_label_all, model_file_name)


print(roc_auc3, pr_auc3)

if not os.path.exists('solutions'):
    os.makedirs('solutions')

solution_fname='solutions/solution_['+Seer_Username+']_pat['+str(patient_index)+']_seg['+str(segment_length_minutes)+']_mode['+str(mode)+']_subtract['+str(subtract_mean)+'].csv'

solutions = pd.DataFrame({'image': df['image'], 'class': probas2_10})
solutions = solutions[['image','class']]

solutions.to_csv(solution_fname,index=0)

stop = timeit.default_timer()

print(pat+' Time: ', stop - start)  


