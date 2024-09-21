#!/bin/bash
#SBATCH -A Berzelius-2024-316
#SBATCH --gpus 1
#SBATCH -C "fat"

#SBATCH -t 02-12:00:00
#SBATCH --output=/home/x_salom/logs/printed_medium%j.text
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=salum.nassor.2008@student.uu.se

module load Anaconda/2023.09-0-hpc1-bdist
module load buildenv-gcccuda/11.8.0-gcc11.3.0
cd ~/TrOCR
conda activate myocr
python main.py --train --custom --dataset_paths datasets/printed/synthetic/beta_v3/image_labels_dataset.csv --save_dir custom_models/trocr-printed/scratch/medium --config_path configs/train/trOCR/printed/custom_printed.json