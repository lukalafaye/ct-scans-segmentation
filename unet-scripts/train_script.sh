#!/bin/bash

# Set environment variables1;5D
export nnUNet_raw="nnUNet_raw"
export nnUNet_preprocessed="nnUNet_preprocessed"
export nnUNet_results="nnUNet_results"

# Create directories if they do not exist
mkdir -p "$nnUNet_raw"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

echo "Environment variables set correctly and directories created."

nnUNetv2_plan_and_preprocess -d 001 -pl nnUNetPlannerResEncM --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=1 nnUNet_compile=False nnUNetv2_train Dataset001_CTSCAN 2d 0 -p nnUNetResEncUNetMPlans --npz --val
CUDA_VISIBLE_DEVICES=1 nnUNet_compile=False nnUNetv2_train Dataset001_CTSCAN 2d 1 -p nnUNetResEncUNetMPlans --npz --val
CUDA_VISIBLE_DEVICES=1 nnUNet_compile=False nnUNetv2_train Dataset001_CTSCAN 2d 2 -p nnUNetResEncUNetMPlans --npz --val
CUDA_VISIBLE_DEVICES=1 nnUNet_compile=False nnUNetv2_train Dataset001_CTSCAN 2d 3 -p nnUNetResEncUNetMPlans --npz --val
CUDA_VISIBLE_DEVICES=1 nnUNet_compile=False nnUNetv2_train Dataset001_CTSCAN 2d 4 -p nnUNetResEncUNetMPlans --npz --val
