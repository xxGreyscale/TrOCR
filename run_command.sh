#!/usr/bin/env bash

# Define an array of commands
#commands=(
#  "TrOCR-SWE --train --custom --lr=6.5e-5 --epochs=3 --eval_every=3 --batch_size=32 --max_target_length=108 --encode=\"WinKawaks/vit-small-patch16-224\" --decoder=\"KBLab/bert-base-swedish-cased\" --tokenizer_type=\"BERT\" --dataset_paths=\"datasets/printed/150k/augmented/image_labels_dataset.csv\" --save_dir=\"./custom_models/trocr-printed/w_augmentation/small\" --model_version=1.1"
#  "TrOCR-SWE --train --custom --lr=6.5e-5 --epochs=3 --eval_every=3 --batch_size=32 --max_target_length=108 --encode=\"WinKawaks/vit-small-patch16-224\" --decoder=\"KBLab/bert-base-swedish-cased\" --tokenizer_type=\"BERT\" --dataset_paths=\"datasets/handwritten/augmented_v2/image_labels_dataset.csv\" --save_dir=\"./custom_models/trocr-handwritten/w_augmentation/small\" --model_version=1.3"
#  # Add more commands here
#)
commands=(
#"TrOCR-SWE --generate_dataset --lang=\"sv\" --augment --printed --num_images 150000 --save_dir \"datasets\\printed\\250k\\augmented\""
"TrOCR-SWE --train --custom --lr=6.5e-5 --epochs=3 --eval_every=3 --batch_size=32 --max_target_length=108 --encode=\"WinKawaks/vit-small-patch16-224\" --decoder=\"KBLab/bert-base-swedish-cased\" --tokenizer_type=\"BERT\" --dataset_paths \"datasets/printed/150k/augmented/image_labels_dataset.csv\" \"datasets/printed/250k/augmented/image_labels_dataset.csv\" --save_dir=\"./custom_models/trocr-printed/w_augmentation/small_medium\" --model_version=1.0"
)

# Loop through each command and run it
for cmd in "${commands[@]}"; do
  echo "Running command: $cmd"
  eval "$cmd"
  if [ $? -ne 0 ]; then
    echo "Command failed: $cmd"
    exit 1
  fi
done

echo "All commands completed successfully"