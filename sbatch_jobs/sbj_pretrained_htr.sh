#!/bin/bash
#SBATCH -A Berzelius-2024-316
#SBATCH --gpus 1

#SBATCH -t 01-12:00:00
#SBATCH --output=logs/htr_w_ms_base_pretrained_training%j.text
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=salum.nassor.2008@student.uu.se

cd /proj/berzelius-2024-46/users/x_salom/TrOCR
conda activate ocr
python main.py --train --pretrained --dataset_paths ../datasets/handwritten/augmented_v2/image_labels_dataset.csv ../datasets/handwritten/normall/image_labels_dataset.csv --save_dir ../custom_models/trocr-handwritten/ms/base --train_config_path ../configs/train/trOCR/handwritten/pretrained/ms_base.json
