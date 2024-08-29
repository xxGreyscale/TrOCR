import csv
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer, DeiTImageProcessor, ViTImageProcessor, RobertaTokenizer, TrOCRProcessor
from evaluate import load
from transformers import VisionEncoderDecoderModel
import logging
import torch.distributed as dist

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
    def __init__(self, config, save_dir, rank, world_size):
        self.config = config
        self.save_dir = save_dir
        self.cer_metric = load("cer", trust_remote_code=True)
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.model = None
        self.processor = None

    def compute_cer(self, pred_ids, label_ids):
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        return cer

    def average_across_gpus(self, tensor):
        """
        Average a tensor across all GPUs.
        """
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            tensor /= dist.get_world_size()
        return tensor

    def prepare_dataset(self, dataframe):
        train_df, test_df = train_test_split(dataframe, test_size=0.2)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        train_dataset = CustomDataset(train_df, self.processor, self.config.max_target_length)
        eval_dataset = CustomDataset(test_df, self.processor, self.config.max_target_length)
        return train_dataset, eval_dataset

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
        processor = TrOCRProcessor.from_pretrained(processor_path, use_fast=use_fast)
        model = VisionEncoderDecoderModel.from_pretrained(vision_encoder_decoder_model_path)
        model.to(self.device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
        self.model, self.processor = self.setup_model_config(processor, model)

    def build_model(self):
        if self.config.tokenizer_type == "BERT":
            tokenizer = BertTokenizer.from_pretrained(self.config.decoder)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(self.config.decoder)

        if self.config.image_processor_type == "DeiT":
            feature_extractor = DeiTImageProcessor.from_pretrained(self.config.encoder)
        elif self.config.image_processor_type == "ViT":
            feature_extractor = ViTImageProcessor.from_pretrained(self.config.encoder)
        else:
            logger.error("Unknown Image processor type")
            return
        processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(self.config.encoder, self.config.decoder)
        model.to(self.device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
        self.model, self.processor = self.setup_model_config(processor, model.module)

    def set_data_loader(self, train_dataset, eval_dataset):
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank)
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=self.world_size, rank=self.rank)
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, sampler=train_sampler)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.config.batch_size, sampler=eval_sampler)
        return train_dataloader, eval_dataloader

    def evaluate(self, model, eval_dataloader):
        model.to(self.device)
        model.eval()
        total_cer = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                outputs = self.model.module.generate(batch["pixel_values"].to(self.device))
                cer = self.compute_cer(pred_ids=outputs, label_ids=batch["labels"].to(self.device))
                batch_size = batch["pixel_values"].size(0)
                total_cer += cer * batch_size
                total_samples += batch_size

        total_cer_tensor = torch.tensor(total_cer, device=self.device)
        total_samples_tensor = torch.tensor(total_samples, device=self.device)

        dist.reduce(total_cer_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_samples_tensor, dst=0, op=dist.ReduceOp.SUM)

        if self.rank == 0:
            aggregated_cer = total_cer_tensor.item() / total_samples_tensor.item()
            logger.info(f"Validation CER: {aggregated_cer}")
            return aggregated_cer
        else:
            return None

    def train(self, train_dataloader, eval_dataloader, eval_every=1):
        best_cer = float('inf')
        learning_rate = self.config.learning_rate
        best_train_loss = float('inf')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Epoch: {epoch}", leave=True):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                # Sum loss across GPUs for reporting
                avg_loss_tensor = torch.tensor(loss.item(), device=self.device)
                avg_loss_tensor = self.average_across_gpus(avg_loss_tensor)
                train_loss += avg_loss_tensor.item()

                optimizer.step()
                optimizer.zero_grad()

            train_loss /= len(train_dataloader)
            if self.rank == 0 and train_loss < best_train_loss:
                best_train_loss = train_loss
                logger.info(f"New best train loss found: {best_train_loss}")

            logger.info(f"Loss after epoch {epoch}: {train_loss}")

            if (epoch + 1) % eval_every == 0:
                aggregated_cer = self.evaluate(self.model, eval_dataloader)
                if self.rank == 0 and aggregated_cer is not None and aggregated_cer < best_cer:
                    best_cer = aggregated_cer
                    logger.info(f"New best CER found: {best_cer}")
                    self.model.module.save_pretrained(f"{self.save_dir}/{self.config.model_version}/vision_model/")
                    self.processor.save_pretrained(f"{self.save_dir}/{self.config.model_version}/processor/")

                if self.rank == 0:
                    with open(f"{self.save_dir}/{self.config.model_version}/metrics.csv", mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Epoch", "CER", "Loss", "Learning Rate"])
                        writer.writerow([epoch, best_cer, best_train_loss, learning_rate])

        logger.info('Finished Training')
        logger.info(f"Best CER: {best_cer}")
