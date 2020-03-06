#!/bin/bash

rm auc_best_patient"$patient_index"_uvar_mvar_mlp_2019.dat
flevel=(a b ab)
mvar=(mpcH mpcR lincoh corrfAmp corrfPha corr)
uvar=(c e E p y cE ce cp cy ey Ey ep Ep py cpe cpE epy Epy cey cEy cpy cepy cEpy)

for patient_index in {1..2}
do
	for ilevel in {0..2}
	do
	    for i in {0..5}
	    do
	        for j in {0..22}
	        do
                                mvari=${mvar[$i]}
                                uvari=${uvar[$j]}
                                fli=${flevel[$ilevel]}
                                cat grid_search/patient"$patient_index"_"$uvari"_"$mvari"_"$fli"_mlp/2020*/auc_best_coeff_linear.dat >> grid_search/auc_best_patient"$patient_index".dat
	       done
           done
        done
done

