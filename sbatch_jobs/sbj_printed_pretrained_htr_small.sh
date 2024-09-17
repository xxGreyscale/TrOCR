#!/bin/bash
#SBATCH -A Berzelius-2024-316
#SBATCH --gpus 1
#SBATCH -C "fat"

#SBATCH -t 02-12:00:00
#SBATCH --output=/home/x_salom/logs/custom_pretrained_small%j.text
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=salum.nassor.2008@student.uu.se

module load Anaconda/2023.09-0-hpc1-bdist
module load buildenv-gcccuda/11.8.0-gcc11.3.0
cd /proj/berzelius-2024-46/users/x_salom/TrOCR
conda activate myocr
python main.py --train --pretrained --dataset_paths datasets/printed/synthetic/130k/image_labels_dataset.csv --save_dir custom_models/trocr-printed/pretrained/small --config_path configs/train/trOCR/printed/custom_pretrained_htr_small.json
