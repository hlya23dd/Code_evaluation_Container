import argparse
import numpy as np
import sys
import os, datetime
import errno
import scipy.io as sio  #for mat file


#import matplotlib as mpl
#mpl.use('Agg')

#import matplotlib.pyplot as plt
import timeit

from AR_v160_svm import *
#from Input import *
import pandas as pd

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

############################################################
args = parser.parse_args()
print(args)

patient_index=args.patient
 
segment_length_minutes=args.file_segment_length

mode=args.mode
n_start=args.n_start
n_end=args.n_end

pat='patient'+str(patient_index)

n_model_ensemble=args.model_ensemble_size

split_time='[PATH]/UTC_AB_CD_EF.mat'
split_time=None 
n_train=None

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
uvar_feature_select=args.uvar_fea
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


############################################################
# make a directory according to time
############################################################
mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d'))
try:
    os.makedirs(mydir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error..

if True:
#for uvar_feature_select in feature_range:
  #print(uvar_feature_select)

  channel_range=range(0,num_channel) # univarinat
 

###
# read features
###
  mask=get_feature_mask(uvar_feature_select,time_delay,dim_feature_channelwise)
  if sum(mask)>0:  
   feature_flag=1
   my_label_feature = np.loadtxt(os.path.join(feature_dir,pat+'_label_feature.dat'))
   my_label = my_label_feature[:,0]
   my_feature = my_label_feature[:,1:]
   #print(my_label.shape,my_feature.shape)

# use mask to select univarant feature
   mask=np.asarray(mask)
   dim_feature=mask.sum()
   my_feature = (my_feature.T[mask>0.5]).T
   #print(my_feature.shape)
  else:
   feature_flag=0

######################################################
# multivariant feature
####
# read features
  for feature in mvar_feature_lib:
       feature_ = '_'+feature+'_'
       if feature_ in mvar_feature_select:
                  print(feature)
                  if 'a' in feature_level:
                     feature_a = np.loadtxt(os.path.join(feature_dir,pat+'_a_'+feature+'_label_feature.dat'))
                  if 'b' in feature_level:
                     feature_b = np.loadtxt(os.path.join(feature_dir,pat+'_b_'+feature+'_label_feature.dat'))
                  if feature_flag==0:
                     feature_flag=1
                     if 'a' in feature_level:
                        my_feature = feature_a[:,1:]
                     if 'b' in feature_level:
                        my_feature = feature_b[:,1:]
                     if 'ab' in feature_level:
                        my_feature = np.c_[feature_a[:,1:],feature_b[:,1:]]
                  else:
                     if 'a' in feature_level:
                        my_feature = np.c_[my_feature,feature_a[:,1:]]
                     if 'b' in feature_level:
                        my_feature = np.c_[my_feature,feature_b[:,1:]]
  #print("this is done!")

  #print(my_feature.shape)

####################################################### 2018-12-5
  
  # replace NaN of samples
  my_feature = feature_NaN(my_feature,my_feature.shape[1])
  my_feature = feature_inf(my_feature,my_feature.shape[1])
  # mean centralization & standalization
  
  if i_standardization>0:
     for i_feature in range(my_feature.shape[1]):  
      yy=my_feature[:,i_feature]
      yy= yy-sum(yy)/len(yy)  # mean centralization
      if sum(yy*yy)>0:
         yy= yy/np.sqrt(sum(yy*yy)/len(yy))  # normalization
      my_feature[:,i_feature]=yy

######################################################

  my_feature3=my_feature.copy()  # 
  my_label3=my_label.copy()  # 

  test_feature, test_label, train_feature, train_label  = feature_train_test_split(my_feature3,my_label3, patient_index, segment_length_minutes, split_time, n_train)

  #train_feature, train_label, test_feature, test_label, valid_feature, valid_label  = feature_train_valid_test_split_index(my_feature3,my_label3,pat)
  #print(test_feature.shape, train_feature.shape)

  n_1=sum(train_label)
  n_0=len(train_label)-n_1


  start = timeit.default_timer()


###
# mlp-auc
###
  #print('# mean_auc_train std_auc_train mean_auc_valid std_auc_valid mean_auc_test std_auc_test C cw_1')

  auc_valid_best=0
  auc_test_best=0

  pr_auc_valid_best=0
  pr_auc_test_best=0

  feature_used='_'+uvar_feature_select+'_'+mvar_feature_select+'_'

  #print(n_0,n_1)
  class_1_weight=n_0/n_1

  print('class_1_weight=',class_1_weight)

  rocpr=rocpr2=0
  mlp=None
  for kk in range(n_model_ensemble):
      reset_keras() 
      roc_auc, roc_auc3, pr_auc, pr_auc3, model = keras_mlp_10m_prb(test_feature, test_label, train_feature, train_label, mydir, pat, segment_length_minutes, class_1_weight, batch_size, epochs, kernel_constraint_weight, verbose, hidden_layer_size, i_feature_selection)
      print(roc_auc, pr_auc, roc_auc3, pr_auc3)
      if roc_auc3*pr_auc3>rocpr and roc_auc*pr_auc>0.7:
         rocpr=roc_auc3*pr_auc3
         mlp=model
      if roc_auc3*pr_auc3>rocpr2:
         rocpr2=roc_auc3*pr_auc3
         mlp2=model
      f = open(os.path.join(mydir,'auc_best_coeff_linear.dat'), 'a')
      f.write("%s %f %f %f %f\n" % (pat, roc_auc, pr_auc, roc_auc3, pr_auc3))
      f.close()

  model_file_name= pat+'_best.hd5'
  if mlp is None:
     mlp2.save(model_file_name)   # HDF5 file, you have to pip3 install h5py if don't have it
  else:
     mlp.save(model_file_name)   # HDF5 file, you have to pip3 install h5py if don't have it

  roc_auc3, pr_auc3, test_label_10, probas2_10, model_file_name = keras_mlp_10m_prb_oldmodel_test(test_feature, test_label, model_file_name)

  #print(roc_auc, pr_auc, roc_auc3, pr_auc3)

  stop = timeit.default_timer()


