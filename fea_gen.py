import argparse
import sys
import os, datetime
import errno
import scipy.io as sio  #for mat file

import numpy as np
import timeit
import pandas as pd

from AR_v160_svm import *

############################################################
# make a directory according to time
############################################################

mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d'))
try:
    os.makedirs(mydir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error..

############################################################
parser = argparse.ArgumentParser()

# Evaluation settings
parser.add_argument('-csv', '--path', help='path to the csv that includes the files', default='/CSVs')
parser.add_argument('-m', '--mode', help='Mode. 0: feature generation, Mode. 1: training, 2: validation, 3: test', type=int, default=0,
                    choices=[0, 1, 2, 3])
parser.add_argument('-p', '--patient', help='Patient number, 1 to 15 is available', type=int, default=1)
parser.add_argument('-l', '--file_segment_length', help='Segment length in minutes, 1 or 10', type=int, default=10)
parser.add_argument('-sm', '--subtract_mean', help='Subtract channelwise mean of each file', type=int, default=1,
                    choices=[0, 1])

parser.add_argument('-ns', '--n_start', help='Starting index of segments to calculate features for', type=int, default=-1)
parser.add_argument('-ne', '--n_end', help='End index of segments to calculate features for', type=int, default=-1)

############################################################
args = parser.parse_args()
print(args)

patient_index=args.patient
 
segment_length_minutes=args.file_segment_length

mode=args.mode
n_start=args.n_start
n_end=args.n_end
n_start=0
n_end=2

pat='patient'+str(patient_index)


############################################################
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

i_standardization = -1 # no normalization

mvar_feature_lib=['mpcH','mpcR','lincoh','corrfAmp','corrfPha','corr','granger']
mvar_feature_function_lib=['mpc_H(pat_gData2, lowcut, highcut, fs=400)',
'mpc_R(pat_gData2, lowcut, highcut, fs=400)',
'coherence_f(pat_gData2, lowcut, highcut, fs=400)', 
"corr_f(pat_gData2, lowcut, highcut, phase_or_amp='amp', fs=400)",
"corr_f(pat_gData2, lowcut, highcut, phase_or_amp='phase', fs=400)",
'corr_y(pat_gData2)',
'ar_granger(ar_poly_input)']

feature_range =['_mpcH_mpcR_lincoh_corrfAmp_corrfPha_corr_']
mvar_feature_select= feature_range[0]


lowcut = 0.0
highcut = 200.0

############################################################

if mode <= 1: # train & feature_generation
   segment_file_name_lables='/CSVs/train_filenames_labels_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'
if mode == 2: # valid
   segment_file_name_lables='/CSVs/validation+_filenames_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'
if mode == 3: # test
   segment_file_name_lables='/CSVs/test_filenames_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'

df = pd.read_csv(segment_file_name_lables)
n_files=len(df)
segment_fnames=df['image']
if n_start <0:
   n_start=0
if n_end > n_files or n_end < 0:
   n_end = n_files
i_file_range=range(n_start,n_end+1)

if mode == 1: # train
   labels=df['class']
else:
   labels=np.zeros(n_files)

for i_mat_segment in i_file_range:
    channel_range=range(0,num_channel) # univariate
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

        my_feature_all_channel, my_label_all_channel = ar_ps(ar_poly_input)
 
    if i_mat_segment == i_file_range[0]:
       my_feature = my_feature_all_channel # add samples for each file portions, train& test or blocs
       my_label =my_label_all_channel
    else:
       my_feature = np.r_[my_feature,my_feature_all_channel]  # add samples for each file portions, train& test or blocs
       my_label = np.r_[my_label,my_label_all_channel]

############################################################
# get mvar
####
    pat_gData=pat_gData*1.0  

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
      
#####
# calculation is done
#####

############################################################
# collect and write feature 
    print('feature collection')

    if i_mat_segment == i_file_range[0]:
       #my_label = pat_gLabel
       for feature in mvar_feature_lib:
           feature_ = '_'+feature+'_'
           if feature_ in mvar_feature_select:
                  my_feature_name_a='featurea'+'_'+feature
                  my_feature_name_b='featureb'+'_'+feature
                  exec(my_feature_name_a + "="+feature+'a')
                  exec(my_feature_name_b + "="+feature+'b')
              
        
    else:
       #my_label = np.r_[my_label,pat_gLabel]
       for feature in mvar_feature_lib:
           feature_ = '_'+feature+'_'
           if feature_ in mvar_feature_select:
                  my_feature_name_a='featurea'+'_'+feature
                  my_feature_name_b='featureb'+'_'+feature
                  exec(my_feature_name_a + "=np.r_[" +my_feature_name_a +',' +feature+'a'+"]")
                  exec(my_feature_name_b + "=np.r_[" +my_feature_name_b +',' +feature+'b'+"]")


# replace NaN of samples
my_feature = feature_NaN(my_feature,my_feature.shape[1])
my_feature = feature_inf(my_feature,my_feature.shape[1])

# feature standalization
if i_standardization == 1:
     for i_feature in range(my_feature.shape[1]):  
      yy=my_feature[:,i_feature]
      yy= yy-sum(yy)/len(yy)  # mean centralization
      if sum(yy*yy)>0:
         my_feature[:,i_feature]= yy/np.sqrt(sum(yy*yy)/len(yy))  # normalization

f = open(os.path.join(mydir,pat+'_label_feature.dat'), 'w')  
print(my_label.shape,my_feature.shape)
np.savetxt(f, np.c_[my_label,my_feature])
f.close()

# save features

for feature in mvar_feature_lib:
       feature_ = '_'+feature+'_'
       if feature_ in mvar_feature_select:
                  my_feature_name_a='featurea'+'_'+feature
                  my_feature_name_b='featureb'+'_'+feature

                  f = open(os.path.join(mydir,pat+'_'+my_feature_name_a[7:]+'_label_feature.dat'), 'w')  # write label & feature before shuffle
                  exec("np.savetxt(f, np.c_[my_label,"+my_feature_name_a+"])")
                  f.close()
                  f = open(os.path.join(mydir,pat+'_'+my_feature_name_b[7:]+'_label_feature.dat'), 'w')  # write label & feature before shuffle
                  exec("np.savetxt(f, np.c_[my_label,"+my_feature_name_b+"])")
                  f.close()

