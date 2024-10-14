#!/bin/bash
#SBATCH -A Berzelius-2024-316
#SBATCH --gpus 1
#SBATCH -C "fat"

#SBATCH -t 00-04:00:00
#SBATCH --output=/home/x_salom/logs/evaluate_custom_FFT_on_Kubhist_n_synthetic%j.text
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=salum.nassor.2008@student.uu.se

module load Anaconda/2023.09-0-hpc1-bdist
module load buildenv-gcccuda/11.8.0-gcc11.3.0
cd ~/TrOCR
conda activate ml-ocr
python model_evaluation.py --name FFT --batch_size 128 --dataset_name synthetic --dataset /home/x_salom/TrOCR/datasets/printed/synthetic/test/image_labels_dataset.csv --model custom_models/duplicated-3/trocr-printed/pretrained/fft/1.3/vision_model --processor custom_models/duplicated-3/trocr-printed/pretrained/fft/1.3/processor
python model_evaluation.py --name FFT --batch_size 128 --dataset_name kubhist --dataset /home/x_salom/TrOCR/datasets/printed/Histrorical_News_Paper/test.csv  --model custom_models/duplicated-3/trocr-printed/pretrained/fft/1.3/vision_model --processor custom_models/duplicated-3/trocr-printed/pretrained/fft/1.3/processor