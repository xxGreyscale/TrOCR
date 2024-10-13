#!/bin/bash
#SBATCH -A Berzelius-2024-316
#SBATCH --gpus 1
#SBATCH -C "fat"

#SBATCH -t 02-16:00:00
#SBATCH --output=/home/x_salom/logs/evaluate_all_models%j.text
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=salum.nassor.2008@student.uu.se

module load Anaconda/2023.09-0-hpc1-bdist
module load buildenv-gcccuda/11.8.0-gcc11.3.0
cd ~/TrOCR
conda activate ml-ocr
python evaluate.py --dataset datasets/printed/Histrorical_News_Paper --model custom_models/trocr-printed/pretrained/ms-base-fft/2.0/vision_model --processor custom_models/trocr-printed/pretrained/ms-base-fft/2.0/processor
python evaluate.py --dataset datasets/printed/Histrorical_News_Paper --model custom_models/trocr-printed/pretrained/ms-base-ful/2.0/vision_model --processor custom_models/trocr-printed/pretrained/ms-base-ful/2.0/processor
python evaluate.py --dataset datasets/printed/Histrorical_News_Paper --model custom_models/trocr-printed/pretrained/ms-base-fll/2.0/vision_model --processor custom_models/trocr-printed/pretrained/ms-base-fll/2.0/processor