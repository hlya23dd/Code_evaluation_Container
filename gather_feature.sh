#!/bin/bash
mkdir -p ./features


patient_index=1
n_runs=10

mvar=(mpcH mpcR lincoh corrfAmp corrfPha corr)

for i in $(seq 0 $n_runs)
do

        cat ./feature_generation/feature_generation_Patient"$patient_index"_part"$i"/2020*/patient"$patient_index"_label_feature.dat >> ./features/patient"$patient_index"_label_feature.dat

	for j in {0..5}
	do
           mvari=${mvar[$j]}
           cat ./feature_generation/feature_generation_Patient"$patient_index"_part"$i"/2020*/patient"$patient_index"_a_"$mvari"_label_feature.dat >> ./features/patient"$patient_index"_a_"$mvari"_label_feature.dat
           cat ./feature_generation/feature_generation_Patient"$patient_index"_part"$i"/2020*/patient"$patient_index"_b_"$mvari"_label_feature.dat >> ./features/patient"$patient_index"_b_"$mvari"_label_feature.dat

	done
done


