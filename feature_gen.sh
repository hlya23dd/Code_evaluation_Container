#!/bin/bash

mkdir -p ./feature_generation


patient_index=1
segment_length_minutes=10
n_runs=10
n_segment_per_run=500

	for i in $(seq 0 $n_runs)
	do

           cp -r ./templates/feature_generation_tmp ./feature_generation/feature_generation_Patient"$patient_index"_part"$i"
           cd ./feature_generation/feature_generation_Patient"$patient_index"_part"$i"

           j=$((i*$n_segment_per_run))
           k=$((j+n_segment_per_run-1)) 

           # docker run -v $PWD:/code -v $DATA_DIR:/segments -v $CSV_DIR:/CSVs/ feature_based  python ./fea_gen.py --patient $patient_index --mode 0 --file_segment_length 10 --n_start $j --n_end $k
           cd ../../

	done


