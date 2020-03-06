import numpy as np

import sys
import os, datetime
import h5py
import scipy.io as sio  #for mat file
import gc 

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#import matplotlib.pyplot as plt


from scipy import signal
from sklearn.utils import shuffle
from scipy import interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

 
#from Input import *


from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow as tf

from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras import optimizers
from keras.models import load_model
from keras import regularizers
from keras.constraints import Constraint
from keras import backend as K

from scipy.signal import butter, lfilter
import pandas as pd

num_channel = 16
##########################################################

# Reset Keras Session, to deal with memory leak when train model in loop
# possible problem with Keras opening tf section
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    #try:
    #    del classifier # this is from global space - change this as you need
    #except:
    #    pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    #set_session(tensorflow.Session(config=config))

    session_config = tf.ConfigProto(
      log_device_placement=True,
      inter_op_parallelism_threads=0,
      intra_op_parallelism_threads=0,
      allow_soft_placement=True)

    sess = tf.Session(config=session_config)

def vec_2_matrix(fea,dim_u, n_channel):

    n_sample = fea.shape[0]
    dim_m=fea.shape[1]-dim_u

    if dim_m==int(n_channel*n_channel):   # full matrix 
       n_col = int(dim_u/n_channel)+n_channel

    if dim_m==int(3*n_channel):   # b-feature
       n_col = int(dim_u/n_channel)+3

    if dim_m==int(n_channel*(n_channel-1)/2):  # a-feature
       n_col = int(dim_u/n_channel)+n_channel

    if dim_m==int(n_channel*(n_channel-1)/2+3*n_channel):  # ab-feature
       n_col = int(dim_u/n_channel)+n_channel+3

    fea_o=np.ones((n_sample,n_channel,n_col))
    for i in range(n_sample):
        vector=fea[i,:dim_u] 
        fea_o[i,:,:int(dim_u/n_channel)]=vector.reshape(n_channel,int(dim_u/n_channel))

    if dim_m==int(n_channel*n_channel):   # full matrix 
       for i in range(n_sample):
           vector=fea[i,dim_u:] 
           fea_o[i,:,int(dim_u/n_channel):]=vector.reshape(n_channel,n_channel)

    if dim_m==int(3*n_channel):   # b-feature
       for i in range(n_sample):
           vector=fea[i,dim_u:] 
           fea_o[i,:,int(dim_u/n_channel):]=vector.reshape(3,n_channel).T

    if dim_m==int(n_channel*(n_channel-1)/2):  # a-feature
       for i in range(n_sample):
           vector=fea[i,dim_u:] 
           matrix=fea_o[i,:,int(dim_u/n_channel):]
           #print(matrix.shape)
           k=0
           for ii in range(n_channel):        
               for jj in range(ii+1, n_channel):   # without the diagonal elements
                   matrix[ii, jj]=vector[k]
                   k=k+1
           fea_o[i,:,int(dim_u/n_channel):]=matrix

    if dim_m==int(n_channel*(n_channel-1)/2+3*n_channel):  # ab-feature
       for i in range(n_sample):
           vector=fea[i,dim_u:int(-3*n_channel)] 
           matrix=fea_o[i,:,int(dim_u/n_channel):-3]
           k=0
           for ii in range(n_channel):        
               for jj in range(ii+1, n_channel):   # without the diagonal elements
                   matrix[ii, jj]=vector[k]
                   k=k+1
           fea_o[i,:,int(dim_u/n_channel):-3]=matrix

           vector=fea[i,int(-3*n_channel):] 
           fea_o[i,:,-3:]=vector.reshape(3,n_channel).T
    return fea_o


def upper_right_triangle_2_orginal(fea,n_channel):

    fea_o=np.ones((fea.shape[0],n_channel,n_channel))
    for i in range(fea.shape[0]):
        vector=fea[i] 
        matrix=fea_o[i]
        k=0
        for i in range(n_channel):        
            for j in range(i+1, n_channel):   # without the diagonal elements
                matrix[i, j]=vector[k]
                k=k+1
        fea_o[i]=matrix
    return fea_o

def feature_b_reshape(fea,n_channel):

    fea_o=np.ones((fea.shape[0],n_channel,3))
    for i in range(fea.shape[0]):
        vector=fea[i]
        vector=vector.reshape(3,n_channel)
        fea_o[i]=vector.T            
    return fea_o


def feature_train_test_split(my_feature, my_label, patient_index=1, segment_length_minutes=10, split_time=None, n_train=None):
  #print(n_train)
  #my_feature, my_label = shuffle(my_feature, my_label, random_state=0)
  n10=int(4*segment_length_minutes)
  if split_time is None and n_train is None:
     n_sample=len(my_label)
     n_train=int(int(n_sample*2/3)/n10)*n10

  train_feature=my_feature[:n_train]
  train_label=my_label[:n_train]
  test_feature=my_feature[n_train:]
  test_label=my_label[n_train:]

  if split_time is not None:     
     segment_file_name_lables='/CSVs/train_filenames_labels_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv' 
     df = pd.read_csv(segment_file_name_lables)
     n_files=len(df)
     segment_fnames=df['image']
     d2=df['image'].sort_values()
     d3=d2[d2<split_time]

     train_mask=np.zeros(n_files)
     train_mask[d3.keys()]=1
     mask=np.repeat(train_mask, n10)

     train_feature=my_feature[mask>0.5]
     train_label=my_label[mask>0.5]
     test_feature=my_feature[mask<0.5]
     test_label=my_label[mask<0.5]

  return test_feature, test_label, train_feature, train_label

def feature_train_valid_test_split_index(my_feature,my_label,PA):  # for kaggle2016

  pat_PA={'patient1': '1', 'patient2': '2', 'patient3':'3'}
  n_ii={'1': 570, '2': 1836, '3': 1908}
  n_pi={'1': 256, '2': 222, '3': 255}
  n_test_public={'1': 61, '2': 297, '3': 206}
  n_test_private={'1': 144, '2': 697, '3': 483}

  pat=pat_PA[PA]

  n_train=(n_ii[pat]+n_pi[pat])*40
  n_valid=n_test_public[pat]*40

  train_feature=my_feature[:n_train]
  train_label=my_label[:n_train]
  valid_feature=my_feature[n_train:(n_valid+n_train)]
  valid_label=my_label[n_train:(n_valid+n_train)]
  test_feature=my_feature[(n_valid+n_train):]
  test_label=my_label[(n_valid+n_train):]
 
  return train_feature, train_label, valid_feature, valid_label, test_feature, test_label


def feature_NaN(my_feature,dim_feature):

  from sklearn.preprocessing.imputation import Imputer
  dim_feature=my_feature.shape[1]
  imp = Imputer(missing_values=np.nan, strategy='mean')
  correction_array=[0]*2*dim_feature
  correction_array=np.asarray(correction_array).reshape(2,dim_feature)
  imp.fit(correction_array) 
  my_feature=imp.transform(my_feature) # preprocessing to get rid of NaN, infinity, etc.

  return my_feature


def feature_inf(my_feature,dim_feature):

  from sklearn.preprocessing.imputation import Imputer
  dim_feature=my_feature.shape[1]
  imp = Imputer(missing_values=np.inf, strategy='mean')
  correction_array=[0]*2*dim_feature
  correction_array=np.asarray(correction_array).reshape(2,dim_feature)
  imp.fit(correction_array) 
  my_feature=imp.transform(my_feature) # preprocessing to get rid of NaN, infinity, etc.

  return my_feature


##########################################################


def eeg_pca_ica_mat_segment_Melborne(segment_fname,pat_gLabel,segment_length_minutes=10):

   if True:
      mat_dic=sio.loadmat(segment_fname)  # read file contents as dictionary
      pat_gData=mat_dic['dataStruct'][0][0][0]
   
      L_segment=pat_gData.shape[0]
      n_channel=pat_gData.shape[1]

      n_sample_new=int(4*segment_length_minutes)
      L_segment_new=int(L_segment/n_sample_new)

      L_omit=L_segment_new*n_sample_new-L_segment
      if L_omit <0:
         pat_gData2=pat_gData[:L_omit,:]
      else:
         pat_gData2=pat_gData

      del pat_gData
      gc.collect()

      pat_gData=pat_gData2.reshape(n_sample_new,L_segment_new,n_channel)
      del pat_gData2
      gc.collect()
      n_sample=pat_gData.shape[0]
      L_segment=pat_gData.shape[1]
      n_channel=pat_gData.shape[2]
      
      b40=np.ones(n_sample_new).reshape(n_sample_new,1)
      pat_gLabel=b40*pat_gLabel
      pat_gLabel=pat_gLabel.T.reshape(n_sample)

      
   print('iEEG_min iEEG_max',np.max(pat_gData),np.min(pat_gData))

   return pat_gData,pat_gLabel

##########################################################



def ar_map(xm,p,k,f_threshold=0.0,i_low_pass=1,noise_level=0.0,id_sorted_term_power=None):
   
       L_segment=xm.shape[0]
       l=xm.shape[1]

       l2=int((l-1)/2)

       xm2=xm.copy()  # raw to filtered
       xm2=xm         # filtered to filtered

       x=np.ones((L_segment-k, 1))  # container for the constructed x_variable
   
       for i in range(0,l):
           for j in range(0,k):
               xx= xm[j:-(k-j),i]
               if p==1:
                  x=np.c_[ x, xx] 
       x2= np.delete(x, np.s_[0], 1) # without z´using constant term, as Senger et al.
       y= xm2[k:,l2]

       return x2,y
##########################################################

def band_ps(ar_poly_input):


   patient=ar_poly_input['patient']
   channel=ar_poly_input['channel']
   ii_index=ar_poly_input['ii_index']
   pi_index= ar_poly_input['pi_index']
   output_dir= ar_poly_input['mydir']
   data_type= ar_poly_input['data_type']
   data_portion =ar_poly_input['data_portion'] 
   i_PCA =ar_poly_input['i_PCA']
   i_ICA =ar_poly_input['i_ICA']
   i_PCA_standardization =ar_poly_input['i_PCA_standardization']
   pat_gData  =   ar_poly_input['pat_gData']
   pat_gLabel  =   ar_poly_input['pat_gLabel']
       

   
   #print(pat_gData.shape) # (sample_num,sample_size,channel_num)
   
   n_sample=pat_gData.shape[0]
   L_segment=pat_gData.shape[1]
   n_channel=pat_gData.shape[2]
   
   ps_delta=np.zeros(n_sample)  # 
   ps_theta=np.zeros(n_sample)  # 
   ps_alpha=np.zeros(n_sample)  # 
   ps_beta=np.zeros(n_sample)  # 
   ps_gamma=np.zeros(n_sample)  # 
   mean=np.zeros(n_sample)  # 
   sigma=np.zeros(n_sample)  # 
   skew=np.zeros(n_sample)  # 
   kurt=np.zeros(n_sample)  # 
   
   
   CH=str(channel)

   
   for m in range(0,n_sample):

       y= pat_gData[m,:,channel]  # mean centralization
       y_mean=sum(y)/len(y)
       y= y-y_mean  # mean centralization
       y_sigma=0 
       if sum(y*y)>0:
          y_sigma=np.sqrt(sum(y*y)/len(y)) 
          y= y/y_sigma  # normalization
       
       fs=200 
       f_y, Sf_y = signal.periodogram(y, fs)   #fftw?
       
       ps_delta[m]=sum(Sf_y[(f_y>0.0) * (f_y<4.0)])/sum(Sf_y[f_y>0.0])
       ps_theta[m]=sum(Sf_y[(f_y>4.0) * (f_y<8.0)])/sum(Sf_y[f_y>0.0])
       ps_alpha[m]=sum(Sf_y[(f_y>8.0) * (f_y<12.0)])/sum(Sf_y[f_y>0.0])
       ps_beta[m]=sum(Sf_y[(f_y>12.0) * (f_y<30.0)])/sum(Sf_y[f_y>0.0])
       ps_gamma[m]=sum(Sf_y[f_y>30.0])/sum(Sf_y[f_y>0.0])
       
   
       y2_mean_s=sum(y*y)/len(y)
       y3_mean=sum(y*y*y)/len(y)
       y4_mean=sum(y*y*y*y)/len(y)

       mean[m]=y_mean
       sigma[m]=y_sigma
       if y2_mean_s>0:
          skew[m]=y3_mean/y2_mean_s/np.sqrt(y2_mean_s)
          kurt[m]=y4_mean/y2_mean_s/y2_mean_s
       else:
          skew[m]=y3_mean
          kurt[m]=y4_mean
    
       #if not(m%200): print(m,mean[m],sigma[m],skew[m],kurt[m],ps_delta[m],ps_theta[m],ps_alpha[m],ps_beta[m],ps_gamma[m],ps_delta[m]+ps_theta[m]+ps_alpha[m]+ps_beta[m]+ps_gamma[m])


   dict_return={'ps_delta': ps_delta,
       'ps_theta': ps_theta,
       'ps_alpha': ps_alpha,
       'ps_beta': ps_beta,
       'ps_gamma': ps_gamma,
       'mean': mean,
       'sigma': sigma,
       'skew': skew, 
       'kurt': kurt, 
       'pat_Label': pat_gLabel
   }


   return dict_return



##########################################################
##########################################################################
 
def ar_ps(ar_poly_input):


   poly_order=ar_poly_input['poly_order']
   time_delay=ar_poly_input['time_delay']
   regression_mode= ar_poly_input['regression_mode']
   num_neigh= ar_poly_input['num_neigh']
   f_threshold=ar_poly_input['f_threshold']
   n_noisy=ar_poly_input['n_noisy']
   noise_level=ar_poly_input['noise_level']
   i_low_pass=ar_poly_input['i_low_pass']
   pat_gData  =   ar_poly_input['pat_gData']
   pat_gLabel  =   ar_poly_input['pat_gLabel']
 

   #print('start of AR.')
   
   
   #print(pat_gData.shape) # (sample_num,sample_size,channel_num)
   
   n_sample=pat_gData.shape[0]
   L_segment=pat_gData.shape[1]
   n_channel=pat_gData.shape[2]
   #print(n_channel)
   
   
   k=time_delay # time delay
   p=poly_order # polynomial order
   l=num_neigh # number of neighbour (include target channel)
   
   
   if p>0 and p<11: 
      n_variable = k*l*p # no cross terms
   
   # compose the matrix by hand
   

   id_sorted_term_power=None
   
   for channel in range(n_channel):           

     err2=np.zeros(n_sample) # power of prediction error
     err3=np.zeros(n_sample) # power of constant term
     err6=np.zeros(n_sample) # power of original signal

     ps_delta=np.zeros(n_sample)  # 
     ps_theta=np.zeros(n_sample)  # 
     ps_alpha=np.zeros(n_sample)  # 
     ps_beta=np.zeros(n_sample)  # 
     ps_gamma=np.zeros(n_sample)  # 
     mean=np.zeros(n_sample)  # 
     sigma=np.zeros(n_sample)  # 
     skew=np.zeros(n_sample)  # 
     kurt=np.zeros(n_sample)  # 

     for m in range(0,n_sample):
       y= pat_gData[m,:,channel]  # mean centralization
       y_mean=sum(y)/len(y)
       y= y-y_mean  # mean centralization
       y_sigma=0 
       if sum(y*y)>0:
          y_sigma=np.sqrt(sum(y*y)/len(y)) 
          y= y/y_sigma  # normalization

       fs=200 
       f_y, Sf_y = signal.periodogram(y, fs)   #fftw?
       
       ps_delta[m]=sum(Sf_y[(f_y>0.0) * (f_y<4.0)])/sum(Sf_y[f_y>0.0])
       ps_theta[m]=sum(Sf_y[(f_y>4.0) * (f_y<8.0)])/sum(Sf_y[f_y>0.0])
       ps_alpha[m]=sum(Sf_y[(f_y>8.0) * (f_y<12.0)])/sum(Sf_y[f_y>0.0])
       ps_beta[m]=sum(Sf_y[(f_y>12.0) * (f_y<30.0)])/sum(Sf_y[f_y>0.0])
       ps_gamma[m]=sum(Sf_y[f_y>30.0])/sum(Sf_y[f_y>0.0])
          
       y2_mean_s=sum(y*y)/len(y)
       y3_mean=sum(y*y*y)/len(y)
       y4_mean=sum(y*y*y*y)/len(y)

       mean[m]=y_mean
       sigma[m]=y_sigma
       if y2_mean_s>0:
          skew[m]=y3_mean/y2_mean_s/np.sqrt(y2_mean_s)
          kurt[m]=y4_mean/y2_mean_s/y2_mean_s
       else:
          skew[m]=y3_mean
          kurt[m]=y4_mean

       xm=[0]
       xm[0]=y
       xm=np.transpose(xm)       
       x2,y = ar_map(xm,p,k,0.0,1,0.0,id_sorted_term_power) # noise_level must be 0.0 here !!!      
       x3=x2
       y3=y
       model2 = LinearRegression(fit_intercept = False)
       model2.fit(x3,y3)
   
       ####################################################
       # record the coefficients ->
       ####################################################
       if m==0:
          coeffs=model2.coef_         
       else:
          coeffs=np.c_[coeffs,model2.coef_]          

####################################################
   # get term power ->
####################################################
      
       err2[m]=sum((y-model2.predict(x2))*(y-model2.predict(x2))) # power of error
       err3[m]=sum((y-model2.predict(x2))*(y-model2.predict(x2))*(y-model2.predict(x2))*(y-model2.predict(x2))) # power of error
       err6[m]=sum(y*y) # power of original signal
   
     current_feature=np.c_[err2/err6,np.sqrt(len(y)*err3/err2-1.0),coeffs.T,ps_delta,ps_theta,ps_alpha,ps_beta,ps_gamma,mean,sigma,skew,kurt]  # compose the feature vector, err & coeffs        
     #coeffs= np.delete(coeffs, 0, 1) # delete the first column of coeffs, seed
     if channel == 0:
           my_feature_all_channel=current_feature.copy()
           my_label_all_channel=pat_gLabel.copy()
     else:
           my_feature_all_channel=np.c_[my_feature_all_channel,current_feature]

   return my_feature_all_channel, my_label_all_channel


##########################################################################
def get_feature_mask(feature_select,time_delay,dim_feature_channelwise):
    
    mask=[]
    for channel in range(num_channel):
      if ('E' in feature_select):
         mask.append(1)  # err
         mask.append(1)  # err
      elif ('e' in feature_select):
         mask.append(1)  # err
         mask.append(0)  # err
      else:
         mask.append(0)  # err
         mask.append(0)  # err
      if ('c' in feature_select):
         for dd in range(2,2+time_delay):
             mask.append(1)  # coeffs
      else:
         for dd in range(2,2+time_delay):
             mask.append(0)  # coeffs
      if ('p' in feature_select):
         for dd in range(2+time_delay,7+time_delay):
             mask.append(1)  # ps
      else:
         for dd in range(2+time_delay,7+time_delay):
             mask.append(0)  # ps
      if ('Y' in feature_select):
         for dd in range(7+time_delay,dim_feature_channelwise):
             mask.append(1)  # y-statistics
      elif ('y' in feature_select):
         mask.append(0)  # err
         mask.append(0)  #
         mask.append(1)  #
         mask.append(1)  # err
      else:
         for dd in range(7+time_delay,dim_feature_channelwise):
             mask.append(0)  # y-statistics
    return mask


###############################################################################################
# multilayer perceptron
###############################################################################################
# creat function for multilayer perceptron with feature selection, small network

# Create function returning a compiled network
def create_network(input_dim, Dropout_rate, kernel_regularizer_weight, activity_regularizer_weight, hidden_layer_size):
          
  n_hidden_layer=len(hidden_layer_size)

  mlp = Sequential()

  if True:
       if n_hidden_layer >=1:
         l=hidden_layer_size[0]
         mlp.add(
             Dense(l, input_shape=(input_dim,), activation ='relu', kernel_regularizer=regularizers.l1_l2(l1=kernel_regularizer_weight, l2=kernel_regularizer_weight), activity_regularizer=regularizers.l1(activity_regularizer_weight))
         )
         mlp.add(Dropout(Dropout_rate))
         mlp.add(BatchNormalization())
         i0=1
  if n_hidden_layer >=1:
     for k in range(i0,n_hidden_layer-1):
         l=hidden_layer_size[k]

         mlp.add(
             Dense(l, activation ='relu', kernel_regularizer=regularizers.l1_l2(l1=kernel_regularizer_weight, l2=kernel_regularizer_weight), activity_regularizer=regularizers.l1(activity_regularizer_weight))
         )

         mlp.add(Dropout(Dropout_rate))
         mlp.add(BatchNormalization())

  if n_hidden_layer >=2:
       l=hidden_layer_size[n_hidden_layer-1]
       mlp.add(
           Dense(l, activation ='relu')
       )

       mlp.add(Dropout(Dropout_rate))

  if n_hidden_layer >=1:
         mlp.add(
              Dense(1, activation ='sigmoid')
         )
  else:
         print(n_hidden_layer)
         mlp.add(
             Dense(1, input_shape=(input_dim,), activation ='sigmoid')
         )
 

  # for large minibatch, the learnign rate should be smaller
  adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

  mlp.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])


  # Return compiled network
  return mlp

###############################################################################################

# all you need to create a mask matrix M, which is a NxN identity matrix
# and you can write a contraint like below

class DiagonalWeight(Constraint):
    """Constrains the weights to be diagonal.
    """
    def __call__(self, w):
        N = K.int_shape(w)[-1]
        m = K.eye(N)
        w *= m
        return w

###############################################################################################
###############################################################################################

###############################################################################################
# multilayer perceptron with feature selection, small network

def keras_mlp_10m_prb(test_feature, test_label, train_feature, train_label, mydir='./', pat='patient',segment_length_minutes=10, class_1_weight=1, batch_size=800, epochs=100, kernel_constraint_weight=1e-7, verbose= 1, hidden_layer_size=[16,8,4], i_feature_selection=0, sample_weight_train=None, Dropout_rate=0, kernel_regularizer_weight=1e-7, activity_regularizer_weight=0):

  random_state = np.random.RandomState()
  input_dim=train_feature.shape[1]
  
  mlp = create_network(input_dim, Dropout_rate, kernel_regularizer_weight, activity_regularizer_weight, hidden_layer_size)

  my_class_weight = {0:1 , 1:class_1_weight}

  train_feature2, train_label2 = shuffle(train_feature, train_label, random_state=0)

  if np.sum(sample_weight_train) == None:
#  mlp.fit(train_feature, train_label, epochs=epochs, batch_size=batch_size, class_weight = 'auto', shuffle=True, verbose= verbose, validation_split=0.33)
     mlp.fit(train_feature, train_label, epochs=epochs, batch_size=batch_size, class_weight = 'auto', shuffle=True, verbose= verbose)
  else:
     mlp.fit(train_feature, train_label, epochs=epochs, batch_size=batch_size, sample_weight = sample_weight_train, shuffle=True, verbose= verbose)

#auc for train set
  probas_ = mlp.predict_proba(train_feature)
  probas = probas_[:, 0]

  n_sample_new=int(4*segment_length_minutes)

  n_seg=int(len(probas)/n_sample_new)
  if n_seg*n_sample_new != len(probas):
     print('rearange error')

  probas_10_ = probas.reshape(n_seg,n_sample_new)
  probas_10 = probas_10_.mean(1)

  train_label_10_ = train_label.reshape(n_seg,n_sample_new)
  train_label_10 = train_label_10_[:,0]

  # Compute ROC curve and area the curve for test set
  fpr, tpr, thresholds = roc_curve(train_label_10, probas_10)
  roc_auc = auc(fpr, tpr)
  # calculate precision-recall curve for test set
  precision, recall, thresholds2= precision_recall_curve(train_label_10, probas_10)	
  # calculate precision-recall AUC
  pr_auc = auc(recall, precision)

#auc for test set
  probas2_ = mlp.predict_proba(test_feature)
  probas2 = probas2_[:, 0]
  n_seg=int(len(probas2)/n_sample_new)
  if n_seg*n_sample_new != len(probas2):
     print('rearange error')

  probas2_10_ = probas2.reshape(n_seg,n_sample_new)
  probas2_10 = probas2_10_.mean(1)

  test_label_10_ = test_label.reshape(n_seg,n_sample_new)
  test_label_10 = test_label_10_[:,0]

  # Compute ROC curve and area the curve for test set
  fpr, tpr, thresholds = roc_curve(test_label_10, probas2_10)
  roc_auc2 = auc(fpr, tpr)
  # calculate precision-recall curve for test set
  precision, recall, thresholds2= precision_recall_curve(test_label_10, probas2_10)	
  # calculate precision-recall AUC
  pr_auc2 = auc(recall, precision)


# save model
  #model_file_name= 'model_mlp_'+pat+'_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_.hd5'
  #mlp.save(model_file_name)   # HDF5 file, you have to pip3 install h5py if don't have it 
  #print('model is saved') 
  #return roc_auc, roc_auc2, pr_auc, pr_auc2, model_file_name #, svm_weights, n_sv
  return roc_auc, roc_auc2, pr_auc, pr_auc2, mlp #, svm_weights, n_sv

###############################################################################################

###############################################################################################
def keras_mlp_10m_prb_oldmodel_test(test_feature, test_label, model_file_name, segment_length_minutes=10):

# reload model
  mlp = load_model(model_file_name)

#auc for test set
  probas2_ = mlp.predict_proba(test_feature)
  probas2 = probas2_[:, 0]

  n_sample_new=int(4*segment_length_minutes)
  n_seg=int(len(probas2)/n_sample_new)
  if n_seg*n_sample_new != len(probas2):
     print('rearange error')

  probas2_10_ = probas2.reshape(n_seg,n_sample_new)
  probas2_10 = probas2_10_.mean(1)

  test_label_10_ = test_label.reshape(n_seg,n_sample_new)
  test_label_10 = test_label_10_[:,0]

  # Compute ROC curve and area the curve for test set
  fpr, tpr, thresholds = roc_curve(test_label_10, probas2_10)
  roc_auc2 = auc(fpr, tpr)
  # calculate precision-recall curve for test set
  precision, recall, thresholds2= precision_recall_curve(test_label_10, probas2_10)	
  # calculate precision-recall AUC
  pr_auc2 = auc(recall, precision)
  
  return roc_auc2, pr_auc2, test_label_10, probas2_10, model_file_name

###############################################################################################


def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):   # without the diagonal elements
            accum.append(matrix[i, j])

    return np.array(accum)

def col_row_max(matrix):
    accum = []

    for i in range(matrix.shape[0]):
        accum.append(max(np.delete(matrix[i],i)))

    return np.array(accum)



from scipy.signal import hilbert


# change in pat_gData can be passed back ?
def mpc_H(pat_gData, lowcut, highcut, fs=400, band_pass_method='square_wave'):

   nyq = 0.5 * fs
   if lowcut < 0:
      lowcut=0
   if highcut > nyq:
      highcut = nyq
   
   n_sample=pat_gData.shape[0] #40 = 600s/15s
   L_segment=pat_gData.shape[1]
   n_channel=pat_gData.shape[2]
   print(n_channel)

   for m in range(0,n_sample):
       for channel in range(0,n_channel):
           y= pat_gData[m,:,channel]
           y_mean=sum(y)/len(y)
           y= y-y_mean  # mean centralization, important for correlation matrix !!!
           y_sigma=0 
           if sum(y*y)>0:
              y_sigma=np.sqrt(sum(y*y)/len(y)) 
              y= y/y_sigma  # normalization
           if band_pass_method=='butter':
              y = butter_bandpass_filter(y, lowcut, highcut, fs, order=6)
           if band_pass_method=='square_wave':
#################
# band pass
              freq = np.fft.rfftfreq(len(y), d=1./fs)
              fl=np.fft.rfft(y)
              fl2=fl*(freq>=lowcut)*(freq<=highcut)
              y=np.fft.irfft(fl2)
#################
           pat_gData[m,:,channel]=y


       X=pat_gData[m,:,:]

#       print(np.real(hilbert(X))-X)
#       h = X + (1j * hilbert(X)) # bug in Michael Hills code, 2018.11.28
       h = hilbert(X)
       phase = np.angle(h).T

       num_bins = int(np.exp(0.626 + 0.4 * np.log(X.shape[1] - 1)))
       Hmax = np.log(num_bins)

       num_channels=n_channel

       XX = np.ones((num_channels, num_channels), dtype=np.float64)

       for i in range(num_channels):
           for j in range(i, num_channels):
                ch1_phase = phase[i]
                ch2_phase = phase[j]

                phase_diff = np.mod(np.abs(ch1_phase - ch2_phase), np.pi * 2.0)

                # convert phase_diff into a pdf of num_bins
                hist = np.histogram(phase_diff, bins=num_bins)[0]
                pdf = hist.astype(np.float64) / np.sum(hist)

                H = np.sum(pdf * np.log(pdf + 1e-12))

                p = (H + Hmax) / Hmax

                XX[i][j] = p
                XX[j][i] = p
       XX[np.isnan(XX)] = 0
       w, v = np.linalg.eig(XX)
       #w = np.absolute(w)
       idx = w.argsort()[::-1]   
       w = w[idx]
       v = v[:,idx]
       u = col_row_max(XX)
       XX = upper_right_triangle(XX)
       if m==0:
          c=XX
          W=w
          V=v[0]
          U=u
       else:
          c=np.c_[c,XX]
          W=np.c_[W,w]
          V=np.c_[V,v[0]]
          U=np.c_[U,u]

   dict_return={'corr_y': c.T,
       'corr_max': U.T,
       'e_values': W.T,
       'e_vector': V.T
   }


   return dict_return


def mpc_R(pat_gData, lowcut, highcut, fs=400, band_pass_method='square_wave'):

   n_sample=pat_gData.shape[0] #40 = 600s/15s
   L_segment=pat_gData.shape[1]
   n_channel=pat_gData.shape[2]
   pat_gData=pat_gData*1.0
   for m in range(0,n_sample):
       for channel in range(0,n_channel):
           y= pat_gData[m,:,channel]
           y_mean=sum(y)/len(y)
           y= y-y_mean  # mean centralization
           y_sigma=0 
           if sum(y*y)>0:
              y_sigma=np.sqrt(sum(y*y)/len(y)) 
              y= y/y_sigma  # normalization
           if band_pass_method=='butter':
              y = butter_bandpass_filter(y, lowcut, highcut, fs, order=6)
           if band_pass_method=='square_wave':
#################
# band pass
              freq = np.fft.rfftfreq(len(y), d=1./fs)
              fl=np.fft.rfft(y)
              fl2=fl*(freq>=lowcut)*(freq<=highcut)
              y=np.fft.irfft(fl2)
#################
           pat_gData[m,:,channel]=y
#           print(pat_gData[m,:,channel])

       X=pat_gData[m,:,:]

#       h = X + (1j * hilbert(X)) # bug in Michael Hills code, 2018.11.28
       h = hilbert(X)
       phase = np.angle(h).T

#       print(X)

       num_channels=n_channel

       XX = np.ones((num_channels, num_channels), dtype=np.float64)

       for i in range(num_channels):
           for j in range(i, num_channels):
                ch1_phase = phase[i]
                ch2_phase = phase[j]

                plv = np.exp(1j*(ch1_phase - ch2_phase))
                p= np.abs(sum(plv)/len(plv))

                XX[i][j] = p
                XX[j][i] = p

       XX[np.isnan(XX)] = 0
       w, v = np.linalg.eig(XX)
       #w = np.absolute(w)
       idx = w.argsort()[::-1]   
       w = w[idx]
       v = v[:,idx]

       u = col_row_max(XX)
       XX = upper_right_triangle(XX)
       if m==0:
          c=XX
          W=w
          V=v[0]
          U=u
       else:
          c=np.c_[c,XX]
          W=np.c_[W,w]
          V=np.c_[V,v[0]]
          U=np.c_[U,u]

   dict_return={'corr_y': c.T,
       'corr_max': U.T,
       'e_values': W.T,
       'e_vector': V.T
   }


   return dict_return



def coherence_f(pat_gData, lowcut, highcut, fs=400):

   n_sample=pat_gData.shape[0] #40 = 600s/15s
   L_segment=pat_gData.shape[1]
   n_channel=pat_gData.shape[2]


   n_segment=10
   l_segment=np.int(L_segment/n_segment)

   for m in range(0,n_sample):
       for channel in range(0,n_channel):
           y= pat_gData[m,:,channel]
           y_mean=sum(y)/len(y)
           y= y-y_mean  # mean centralization
           y_sigma=0 
           if sum(y*y)>0:
              y_sigma=np.sqrt(sum(y*y)/len(y)) 
              y= y/y_sigma  # normalization
           pat_gData[m,:,channel]=y

   freq = np.fft.rfftfreq(L_segment, d=1./fs)

#   f_y, Sf_y = signal.periodogram(y, fs)
#   plt.semilogy(f_y,Sf_y)

   fl=np.fft.rfft(y)
#   yy=np.fft.irfft(fl)
#   print(yy-y)

#   f_y, Sf_y = signal.periodogram(yy, fs)
#   plt.semilogy(f_y,Sf_y)

   fl2=fl*(freq>=lowcut)*(freq<=highcut)
#   print(len(fl2))

#   print(fl)
#   print(fl2)
   yy2=np.fft.irfft(fl2)
#   print(yy2)

#   f_y, Sf_y = signal.periodogram(yy2, fs)
#   plt.semilogy(f_y,Sf_y)
#   plt.savefig('bandpass_sf.png')
#   plt.close()

   freq = np.fft.rfftfreq(l_segment, d=1./fs)
   s_range= np.arange(len(freq))

   freq_ = freq[(freq>=lowcut)*(freq<=highcut)]
   s_range_ = s_range[(freq>=lowcut)*(freq<=highcut)]

   for m in range(n_sample):
       for i in range(n_segment):   
             pat_f=np.fft.rfft(pat_gData[m,i*l_segment:(i+1)*l_segment,:], axis=0)  # fourier for one segement of a sample
             for s in s_range_:
                 if s==s_range_[0]:
                    x=np.outer(np.conjugate(pat_f[s,:]),pat_f[s,:]).flatten()
                 else:
                    x=np.c_[x,np.outer(np.conjugate(pat_f[s,:]),pat_f[s,:]).flatten()]
             if i==0:
                X=x
             else:
                X=X+x
       X=np.absolute(X/n_segment)

       for s in range(len(freq_)):
             y_=X[:,s].reshape(n_channel,n_channel)
             y=np.diagonal(y_)
             Y=np.outer(np.sqrt(y),np.sqrt(y))
             z=np.divide(y_,Y)
             if s==0:
                XX=z
             else:
                XX=XX+z                     
       XX=XX/len(freq_)
       XX[np.isnan(XX)] = 0
       w, v = np.linalg.eig(XX)
       #w = np.absolute(w)
       idx = w.argsort()[::-1]   
       w = w[idx]
       v = v[:,idx]

       u = col_row_max(XX)
       XX = upper_right_triangle(XX)
       if m==0:
          c=XX
          W=w
          V=v[0]
          U=u
       else:
          c=np.c_[c,XX]
          W=np.c_[W,w]
          V=np.c_[V,v[0]]
          U=np.c_[U,u]

   dict_return={'corr_y': c.T,
       'corr_max': U.T,
       'e_values': W.T,
       'e_vector': V.T
   }


   return dict_return


# modified from Michael Hills, for amplitude or phase
# instead of use log10(amp), use normalized c_xy/sqrt(c_xx*c_yy) 
# use squre-function for band-pass, order=1 in butter-band-pass

def corr_f(pat_gData, lowcut, highcut, phase_or_amp='amp', fs=400):

   n_sample=pat_gData.shape[0] #40 = 600s/15s
   L_segment=pat_gData.shape[1]
   n_channel=pat_gData.shape[2]

   for m in range(n_sample):
    for channel in range(n_channel):
       y= pat_gData[m,:,channel]
       y_mean=sum(y)/len(y)
       y= y-y_mean  # mean centralization
       y_sigma=0 
       if sum(y*y)>0:
          y_sigma=np.sqrt(sum(y*y)/len(y)) 
          y= y/y_sigma  # normalization
       pat_gData[m,:,channel]=y

   axis = pat_gData.ndim - 2
   pat_f=np.fft.rfft(pat_gData, axis=axis)

   freq = np.fft.rfftfreq(L_segment, d=1./fs)

   for i in range(n_sample):
       X=pat_f[i,:,:]
       X=X[(freq>=lowcut)*(freq<=highcut)]
       if phase_or_amp=='amp':
          X=np.absolute(X)
       if phase_or_amp=='phase':
          X=np.angle(X)
       x_=np.dot(X.T,X)

       y=np.diagonal(x_)
       Y=np.outer(np.sqrt(y),np.sqrt(y))
       if sum(abs(y))>0:
          XX=np.divide(x_,Y)
       else:
          XX=x_

       XX[np.isnan(XX)] = 0
       w, v = np.linalg.eig(XX)
       #w = np.absolute(w)
       idx = w.argsort()[::-1]   
       w = w[idx]
       v = v[:,idx]

       u = col_row_max(XX)
       XX = upper_right_triangle(XX)
       if i==0:
          c=XX
          W=w
          V=v[0]
          U=u
       else:
          c=np.c_[c,XX]
          W=np.c_[W,w]
          V=np.c_[V,v[0]]
          U=np.c_[U,u]

   dict_return={'corr_y': c.T,
       'corr_max': U.T,
       'e_values': W.T,
       'e_vector': V.T
   }


   return dict_return


def corr_y(pat_gData):

   n_sample=pat_gData.shape[0] #40 = 600s/15s
   L_segment=pat_gData.shape[1]
   n_channel=pat_gData.shape[2]

   for m in range(n_sample):
    for channel in range(0,n_channel):
       y= pat_gData[m,:,channel]
       y_mean=sum(y)/len(y)
       y= y-y_mean  # mean centralization
       y_sigma=0 
       if sum(y*y)>0:
          y_sigma=np.sqrt(sum(y*y)/len(y)) 
          y= y/y_sigma  # normalization
       pat_gData[m,:,channel]=y

# pat_gData is read as integer ?

   for i in range(n_sample):
       X=pat_gData[i,:,:]
       x_=np.dot(X.T,X)/L_segment

       y=np.diagonal(x_)
       Y=np.outer(np.sqrt(y),np.sqrt(y))

#       print(y)
       if sum(abs(y))>0:
          XX=np.divide(x_,Y)
       else:
          XX=x_

       XX[np.isnan(XX)] = 0
       w, v = np.linalg.eig(XX)
       #w = np.absolute(w)
       idx = w.argsort()[::-1]   
       w = w[idx]
       v = v[:,idx]

       u = col_row_max(XX)
       XX = upper_right_triangle(XX)
       if i==0:
          c=XX
          W=w
          V=v[0]
          U=u
       else:
          c=np.c_[c,XX]
          W=np.c_[W,w]
          V=np.c_[V,v[0]]
          U=np.c_[U,u]

   dict_return={'corr_y': c.T,
       'corr_max': U.T,
       'e_values': W.T,
       'e_vector': V.T
   }


   return dict_return

   
