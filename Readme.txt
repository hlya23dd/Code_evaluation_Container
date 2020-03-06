Our feature-based seizure prediction method runs in 3 basic steps:
Step 1: generate univariate and mutivariate features for train segments and store them in files. 
Step 2: grid search of optimal feature combination for each patient, find the best model via validation on a subset of train segments 
Step 3: use the best model to deliver predictions for train/valid/test sets

Poor man's parallelism is utilized to speed up the first two steps. It is realized with the help of the following shell/python scripts. 
1. gen_feature_uvar.sh 
2. gen_feature_mvar.sh
3. gather_feature_mvar.sh     
4. gather_feature_uvar.sh  
5. grid_search.sh
6. gather_auc.sh
7. get_best_model.py 

For the paralleled feature calculation it is necessary to specify the number of parallel runs (n_uvar_runs or n_uvar_run) and the number of train segments for each run (n_segment_per_run). The same values are applied to the merging scripts.

For grid search of best feature combination the train set need to be split into train and valid subsets. A split time or the number of train subset needs to be given in the corresponding input file (Input.py). Otherwise, roughly 2/3 of the train set will be used for train and the remaining 1/3 will be used for validation.

After grid search a prediction can be delivered by easily for either set with the stored best model.

To summarize, the following parameters need to be fed by hand:
 - n_uvar_runs and n_segment_per_run in gen_feature_uvar.sh,  n_uvar_runs*n_segment_per_run >= # of train segments
 - n_mvar_runs and n_segment_per_run in gen_feature_mvar.sh,  n_mvar_runs*n_segment_per_run >= # of train segments
 - n_uvar_runs in gather_feature_uvar.sh
 - n_mvar_runs in gather_feature_mvar.sh
 - split_time or n_train in Input.py for grid search, patient specific
 - n_model_ensemble in Input.py for grid search, number of models trained for each feature combination, I used the value of 100, can be set to 20 or 30 for a fast run 

Directory structure:
./templates/: templates for parallel runs of feature generation and grid search
./feature_generation/: temporary directory used for feature generation
./CSVs/: directory for .csv input files
./geberate_solutions/: codes for generating solution files 
./features/: directory used to store features of train segments
./solutions/: place for solution files

 


