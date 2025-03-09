
# Steps to train nnUnet

Change working directory to `nnUNet/`.

Requirements:
- Python3
- GPU

1. Create and activate python3 env: `python3 -m venv venv && source venv/bin/activate`
2. Prepare environment: `./prepare_env.sh` # MODIFY LINE UNDER: fetch exact torch command using pytorch install website
3. Prepare data for nnUnet: `python3 prepare_data.py
4. Make sure GPUs are available: `python3 list-gpus.py`
5. Train UNet model: `./train_script.sh`, tip: Use screen sessions to run folds in parallel. Stop when you see on progress.png in each fold folder that the model stops learning.
6. Rename all best `checkpoint_best.pth` files to `checkpoint_final.pth` inside each fold folder (0-4)
7. Find best UNet config: `CUDA_VISIBLE_DEVICES=1 nnUNetv2_find_best_configuration 001 -p nnUNetResEncUNetMPlans -c 2d`
8. Prepare test data: `python3 prepare_test_data.py`
9. Run inference: `./inference.sh`
10. Generate submission: `python3 generate-submission.py`

Final output `predictions.csv` contains all predictions.


# Steps to train Segformer

Change working directory to `segformer/`.

Requirements:
- Python3
- GPU

1. Create and activate python3 env: `python3 -m venv venv && source venv/bin/activate`
2. Prepare environment: `./prepare_env.sh` # MODIFY LINE UNDER: fetch exact torch command using pytorch install website
3. Make sure GPUs are available: `python3 list-gpus.py`
4. In `train/` folder, you can train the segformer model: `python3 train-segformer-dice-viz.py` or `train-segformer-dice-ce-viz.py` (for dice loss only or mix of dice + cross entropy)
5. In `inference/` folder, you can run trained models: `python3 inference-segformer.py`, generate some images with their originals: `python3 visualize_side_to_side.py` and verify the generated test submission.csv : `python3 verify.py`

