import csv

import wandb
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer, DeiTImageProcessor, ViTImageProcessor, RobertaTokenizer, TrOCRProcessor
from evaluate import load
from transformers import VisionEncoderDecoderModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class TrOCR:
    def __init__(self, config, save_dir):
        self.config = config
        self.save_dir = save_dir
        self.cer_metric = load("cer", trust_remote_code=True)
        logger.info(f"GPU available: {torch.cuda.is_available()}. Now Using device: {torch.cuda.get_device_name(0)}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def compute_cer(self, pred_ids, label_ids):
        """
        Compute the Character Error Rate
        :param pred_ids:
        :param label_ids:
        :return: float value of the CER
        """
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        _cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        return _cer

    def prepare_dataset(self, args):
        """
        Prepare the dataset for training
        :param dataframe:
        :param test_dataframe
        :return: train_dataset and eval_dataset
        """
        train_df, test_df = args
        # reset the indices to start from zero
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        # create the datasets
        train_dataset = CustomDataset(train_df, self.processor, self.config.max_target_length)
        eval_dataset = CustomDataset(test_df, self.processor, self.config.max_target_length)
        print(f"Train dataset: {len(train_dataset)}")
        print(f"Eval dataset: {len(eval_dataset)}")
        return train_dataset, eval_dataset

    def build_model(self):
        """
        Build the model within the instance
        with the specified configuration
        encoder: for feature extraction
        decoder: for text generation, e.g. BERT or RoBERTa
        tokenizer_type: type of tokenizer to use, e.g. BERT or RoBERTa
        """
        if self.config.tokenizer_type == "BERT":
            tokenizer = BertTokenizer.from_pretrained(self.config.decoder)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(self.config.decoder)

        if self.config.image_processor_type == "DeiT":
            feature_extractor = ViTImageProcessor.from_pretrained(self.config.encoder)
        elif self.config.image_processor_type == "ViT":
            feature_extractor = DeiTImageProcessor.from_pretrained(self.config.encoder)
        else:
            logger.error("Unknown Image processor type")
            return
        processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(self.config.encoder, self.config.decoder)

        # Check if multiple GPUs are available and wrap the model
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

            model.to(self.device)
            model, processor = self.setup_model_config(processor, model.module)
            self.model = model
            self.processor = processor
        else:
            model.to(self.device)
            model, processor = self.setup_model_config(processor, model)
            self.model = model
            self.processor = processor

    def setup_model_config(self, processor, model):
        # set special tokens used for creating the decoder_input_ids from the labels
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        # make sure vocab size is set correctly
        model.config.vocab_size = processor.tokenizer.vocab_size
        # set beam search parameters
        model.config.eos_token_id = processor.tokenizer.sep_token_id
        model.config.max_length = self.config.max_target_length
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4
        return model, processor

    def build_model_with_pretrained(self, processor_path, vision_encoder_decoder_model_path, use_fast=False):
        """
        Build the model with a pre-trained model
        Technically, this is used for fine-tuning. And the model path is the same with the processor path
        :param vision_encoder_decoder_model_path:
        :param processor_path:
        :param use_fast:
        :return: nothing
        """
        processor = TrOCRProcessor.from_pretrained(processor_path, use_fast=use_fast)
        model = VisionEncoderDecoderModel.from_pretrained(vision_encoder_decoder_model_path)
        # Check if multiple GPUs are available and wrap the model
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
            model.to(self.device)
            model, processor = self.setup_model_config(processor, model.module)
        else:
            model.to(self.device)
            model, processor = self.setup_model_config(processor, model)
        self.model = model
        self.processor = processor

    def set_data_loader(self, train_dataset, eval_dataset):
        """
        Set the data loader for training and evaluation
        :param train_dataset:
        :param eval_dataset:
        :return: None
        """
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False)
        return train_dataloader, eval_dataloader

    def summary(self):
        """
        Print the model summary
        :return: None
        """
        print(self.model)

    def evaluate(self, model, eval_dataloader):
        """
        Evaluate the model
        :return: None
        """
        # evaluate"
        model.to(self.device)
        model.eval()
        valid_cer = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                # run batch generation
                outputs = self.model.generate(batch["pixel_values"].to(self.device))
                # compute metrics
                cer = self.compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                valid_cer += cer

        valid_cer /= len(eval_dataloader)
        logger.info(f"Validation CER: {valid_cer}")
        logger.info(f"learning rate: {self.config.learning_rate}")
        return valid_cer

    def train(self, train_dataloader, eval_dataloader, eval_every=1):
        """
        Train the model with the specified configuration
        eval_every: evaluate the model every n epochs
        :return: None
        """

        wandb.init(project=self.config.wandb_project, config=self.config.__dict__, dir=self.config.log_dir)
        best_cer = float('inf')  # start with a high CER
        learning_rate = self.config.learning_rate
        best_train_loss = float('inf')  # start with a high loss

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        for epoch in range(self.config.epochs):
            # train
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Epoch: {epoch}", leave=True):
                # get the inputs
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                # forward + backward + optimize
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                logger.info(f"New best train loss found: {best_train_loss / len(train_dataloader)}")
            logger.info(f"Loss after epoch {epoch}: {train_loss / len(train_dataloader)}")

            # Log training loss to wandb
            wandb.log({"train_loss": train_loss / len(train_dataloader), "epoch": epoch})

            # evaluate
            if (epoch + 1) % eval_every != 0:
                continue
            valid_cer = self.evaluate(self.model, eval_dataloader)
            if valid_cer < best_cer:
                best_cer = valid_cer
                # Log validation CER to wandb
                wandb.log({"valid_cer": valid_cer, "epoch": epoch})
                logger.info(f"New best CER found: {best_cer}")
                logger.info(f"Saving the best model...")
                self.model.save_pretrained(f"{self.save_dir}/{self.config.model_version}/vision_model/")
                self.processor.save_pretrained(f"{self.save_dir}/{self.config.model_version}/processor/")
                # save the best model

            # record the best CER, loss and learning rate in csv
            # make sure the file is available first
            with open(f"{self.save_dir}/{self.config.model_version}/metrics.csv", mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "CER", "Loss", "Learning Rate"])
                writer.writerow([epoch, best_cer, best_train_loss, learning_rate])

            if isinstance(self.model, nn.DataParallel):
                logger.info("Saving the model trained in multiple GPU...")
                self.model.module.save_pretrained(f"{self.save_dir}/{self.config.model_version}/vision_model/")
            else:
                logger.info("Saving the model...")
                self.model.save_pretrained(f"{self.save_dir}/{self.config.model_version}/vision_model/")
            logger.info("Saving the processor")
            self.processor.save_pretrained(f"{self.save_dir}/{self.config.model_version}/processor/")

        logger.info('Finished Training')
        logger.info(f"Best CER: {best_cer}")
        wandb.finish()
