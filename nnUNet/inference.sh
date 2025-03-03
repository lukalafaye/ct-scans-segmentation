#!/bin/bash

# Set environment variables;
export nnUNet_raw="nnUNet_raw"
export nnUNet_preprocessed="nnUNet_preprocessed"
export nnUNet_results="nnUNet_results"

# Run inference

nnUNetv2_predict -d Dataset001_CTSCAN -i nnUNet_raw/Dataset001_CTSCAN/imagesTs -o test-results -f 0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetResEncUNetMPlans

#Run postprocessing

nnUNetv2_apply_postprocessing -i test-results -o test-results-pp   -pp_pkl_file nnUNet_results/Dataset001_CTSCAN/nnUNetTrainer__nnUNetResEncUNetMPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl   -np 8   -plans_json nnUNet_results/Dataset001_CTSCAN/nnUNetTrainer__nnUNetResEncUNetMPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json
