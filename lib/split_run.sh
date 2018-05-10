#!/usr/bin/env bash
rootpath=/home/alzoghba/citeulike_data/terms_keywords_based
for dataFolder in 2k_20_P5_reduced 2k_20_P10_reduced 2k_30_P3_reduced 2k_30_P5_reduced 2k_30_P10_reduced 2k_40_P3_reduced 2k_40_P5_reduced    
do
    screen -S splitting_${dataFolder} -dm bash -c " source /home/alzoghba/HyPRec/py3.5_hyprec/bin/activate;  \
    python3 /home/alzoghba/IFUP2018/Recommender_evaluator/lib/split.py -d ${rootpath}/${dataFolder} -s in-matrix-item; exec sh;"      
done
