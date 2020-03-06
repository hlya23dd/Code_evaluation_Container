#!/bin/bash
mkdir -p grid_search

flevel=(a b ab)
mvar=(mpcH mpcR lincoh corrfAmp corrfPha corr)
uvar=(c e E p y cE ce cp cy ey Ey ep Ep py cpe cpE epy Epy cey cEy cpy cepy cEpy)

for patient_index in {1..2}
do
	for i in {0..5}
	do
	    for ilevel in {0..2}
	    do
	        for j in {0..22}
	        do
                                mvari=${mvar[$i]}
                                uvari=${uvar[$j]}
                                fli=${flevel[$ilevel]}
				mkdir grid_search/patient"$patient_index"_"$uvari"_"$mvari"_"$fli"_mlp
				cp AR_v160_svm.py grid_search.py  grid_search/patient"$patient_index"_"$uvari"_"$mvari"_"$fli"_mlp
				cd grid_search/patient"$patient_index"_"$uvari"_"$mvari"_"$fli"_mlp
                                # docker run -v $PWD:/code -v $DATA_DIR:/segments -v $CSV_DIR:/CSVs/ feature_based  python ./grid_search.py --patient $patient_index -uf $uvari -bf $mvari -bv $fli -mns 100
				cd ../..
	       done
           done
        done
done

