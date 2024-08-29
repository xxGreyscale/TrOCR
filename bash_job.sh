#!/bin/bash
#SBATCH -A Berzelius-2024-46
#SBATCH --gpus 4

#SBATCH -t 01-12:00:00
#SBATCH --output=logs/htr_training%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=salum.nassor.2008@student.uu.se

cd /proj/berzelius-2024-46/users/x_salom/TrOCR
conda activate ocr
python main.py --train --custom --dataset_paths="datasets/handwritten/demoktati-n-labours/image_labels_dataset.csv" --save_dir="./custom_models/trocr-handwritten/small" --train_config_path configs/train/htr.json


