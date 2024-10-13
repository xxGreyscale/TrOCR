import argparse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from jiwer import cer
from jiwer import wer
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from evaluate import load

from data.loader.custom_loader import CustomLoader
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, _df, _processor, _max_target_length=100):
        self.df = _df
        self.processor = _processor
        self.max_target_length = _max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['file_name'][idx]
        text = self.df['text'][idx]
        img = Image.open(img_path).convert('RGB')
        pixel_values = self.processor(img, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        _encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return _encoding


def prepare_dataset(args, processor):
    """
    Prepare the dataset for training
    :param args:
    :param processor: Processor for the model
    :return: train_dataset and eval_dataset
    """
    test_df = args
    # Reset the indices to start from zero
    test_df.reset_index(drop=True, inplace=True)

    # Create the datasets
    eval_dataset = CustomDataset(test_df, processor, 108)
    print(f"Eval dataset: {len(eval_dataset)}")
    return eval_dataset


def set_data_loader(eval_dataset, batch_size=16):
    """
    Set the data loader for training and evaluation
    :param batch_size:
    :param eval_dataset:
    :return: train_dataloader, eval_dataloader
    """
    if len(eval_dataset) == 0:
        raise ValueError("Eval dataset size must be greater than 0 after splitting")
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return eval_dataloader


def evaluate_wer_and_cer(_model, _processor, data_loader):
    avg_cer, avg_wer = evaluate(_model, _processor, data_loader)
    return avg_cer, avg_wer


def compute_cer(processor, pred_ids, label_ids):
    """
    Compute the Character Error Rate
    :param pred_ids:
    :param label_ids:
    :return: float value of the CER
    """
    cer_metric = load("cer", trust_remote_code=True)
    wer_metric = load("wer")
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    _cer = cer_metric.compute(predictions=pred_str, references=label_str)
    _wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return _cer, _wer


def evaluate(model, processor, eval_dataloader):
    """
    Evaluate the model
    :return: None
    """
    if torch.cuda.is_available():
        model = model.to('cuda')  # Move model to GPU
    model = model.eval()
    valid_cer = 0.0
    valid_wer = 0.0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            # run batch generation
            outputs = model.generate(batch["pixel_values"].to("cuda"))
            # compute metrics
            cer, wer = compute_cer(processor=processor, pred_ids=outputs, label_ids=batch["labels"])
            valid_cer += cer
            valid_wer += wer

    valid_cer /= len(eval_dataloader)
    valid_wer /= len(eval_dataloader)
    return valid_cer, valid_wer


def run(args):
    # Load the pre-trained TrOCR model and processor
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    # "../custom_models/trocr-printed/pretrained/ms-base-fft/2.0/vision_model"
    processor = TrOCRProcessor.from_pretrained(args.processor)
    # "../custom_models/trocr-printed/pretrained/ms-base-fft/2.0/processor"

    # dataset_paths = ["../datasets/printed/synthetic/test/image_labels_dataset.csv" ]
    # dataset_paths = ["../datasets/printed/Histrorical_News_Paper/test.csv"]
    dataset_paths = args.dataset
    dataset_cl = CustomLoader(dataset_paths)
    dataset_cl.generate_dataframe(['image', 'text'])
    # put ../ in every file name in the dataframe
    df = dataset_cl.get_dataframe()
    df["file_name"] = df["file_name"].apply(lambda x: x.replace("Histrorical_News_Paper/test.csv", "").strip())

    eval_dataloader = (set_data_loader(
        prepare_dataset(
            df if dataset_cl is not None else None, processor), args.batch_size))

    fft_model_cer, fft_model_wer = evaluate_wer_and_cer(model, processor, eval_dataloader)
    print(f"OCR full fine tuned on synthetic dataset average CER: {fft_model_cer}")
    print(f"OCR full fine tuned on synthetic dataset average WER: {fft_model_wer}")


#  main
if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, nargs='+', help='Path to the dataset')
    parser.add_argument('--model', type=str, help='Path to the model')
    parser.add_argument('--processor', type=str, help='Path to the processor')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    run(args)
