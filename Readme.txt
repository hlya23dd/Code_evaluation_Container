####################################################################
To build the local docker image:
docker build . -t feature_based

For setting the env variables:
export DATA_DIR=""
export CSV_DIR=""

####################################################################
Our feature-based seizure prediction method runs in 3 basic steps:
Step 1: generate univariate and mutivariate features for train segments and store them in files. 
Step 2: grid search of optimal feature combination for each patient, find the best model via validation on a subset of train segments 
Step 3: deliver predictions for train/valid/test sets

Corresponding docker commands:
Step 1: docker run -v $PWD:/code -v $DATA_DIR:/segments -v $CSV_DIR:/CSVs/ feature_based  python ./fea_gen.py --patient $patient_index --mode 0 --file_segment_length 10 --n_start $j --n_end $k
Step 2: docker run -v $PWD:/code -v $DATA_DIR:/segments -v $CSV_DIR:/CSVs/ feature_based  python ./grid_search.py --patient $patient_index -uf $uvari -bf $mvari -bv $fli -mns 100
Step 3: docker run -v $PWD:/code -v $DATA_DIR:/segments -v $CSV_DIR:/CSVs/ feature_based  python ./get_solutions.py  --patient $patient_index --mode 1 

For grid search of best feature combination the train set need to be split into train and valid subsets. The number of train subset needs to be given in the input arguments, (--n_train). Otherwise, roughly the first 2/3 of the train set will be used for train and the remaining 1/3 will be used for validation. As output, information regarding the best feature combination and best model is written to the file "Best_model.py".

After grid search a prediction can be delivered for train/valid/test set, either with the stored best model from grid search or via training on the complete train set. This is controlled by the switch "--which_model".

Poor man's parallelism is recommandede for speeding up the calculation in the first two steps. It is realized with the help of the following shell/python scripts. 
1. feature_gen.sh      --> partial features
2. gather_feature.sh   --> complete features  
3. grid_search.sh      --> AUC of each feature combination
4. gather_auc.sh       --> AUC of all feature combinations
5. get_best_model.py   --> Best feature combination and best model

The following parameters need to be fed by hand:
 - n_runs and n_segment_per_run in gen_feature.sh,  n_runs*n_segment_per_run >= # of train segments
 - n_runs in gather_feature.sh
 - n_train for grid_search.py, patient specific
 - model_ensemble_size for grid_search.py, number of models trained for each feature combination, I used the value of 100, the default value is 30 for a fast run 

Directory structure:
./feature_generation/: temporary directory used for feature generation
./CSVs/: directory for .csv input files
./geberate_solutions/: codes for generating solution files 
./features/: directory used to store features of train segments
./solutions/: place for solution files



 
