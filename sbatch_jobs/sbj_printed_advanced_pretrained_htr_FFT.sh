#!/bin/bash
#SBATCH -A Berzelius-2024-316
#SBATCH --gpus 1
#SBATCH -C "fat"

#SBATCH -t 02-12:00:00
#SBATCH --output=/home/x_salom/logs/advanced_printed_fft%j.text
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=salum.nassor.2008@student.uu.se

module load Anaconda/2023.09-0-hpc1-bdist
module load buildenv-gcccuda/11.8.0-gcc11.3.0
cd ~/TrOCR
conda activate ml-ocr
python main.py --train --pretrained --dataset_paths datasets/printed/Histrorical_News_Paper/image_labels_dataset.csv datasets/printed/synthetic/beta_v1/image_labels_dataset.csv datasets/printed/synthetic/beta_v2/image_labels_dataset.csv datasets/printed/synthetic/beta_v3/image_labels_dataset.csv --save_dir custom_models/trocr-printed/pretrained/ms-base-fft --config_path configs/train/trOCR/printed/advanced_pretrained_htr_fft.json