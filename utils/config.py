import json


class Config:
    def __init__(
            self,
            learning_rate,
            epochs,
            batch_size,
            max_target_length,
            model_version,
            decoder=None,
            tokenizer_type=None,
            image_processor_type=None,
            processor=None,
            vision_encoder_decoder_model=None,
            encoder=None,
            eval_frequency=1,
            test_dataset=None,
            wandb_project="TrOCR - Thesis Results",
            log_dir=None  # logs for wandb
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_target_length = max_target_length
        self.wandb_project = wandb_project
        if (tokenizer_type is not None
                and image_processor_type is not None
                and decoder is not None
                and encoder is not None):
            print(f"encoder: {encoder}, decoder: {decoder}, tokenizer_type: {tokenizer_type}")
            self.tokenizer_type = tokenizer_type
            self.image_processor_type = image_processor_type
            self.decoder = decoder
            self.encoder = encoder
        if processor is not None and vision_encoder_decoder_model is not None:
            self.processor = processor
            self.vision_encoder_decoder_model = vision_encoder_decoder_model
        self.model_version = model_version
        self.eval_frequency = eval_frequency
        self.test_dataset = test_dataset
        self.log_dir = log_dir

    def __str__(self):
        if hasattr(self, "processor"):
            return f"Config(learning_rate={self.learning_rate}, epochs={self.epochs}, batch_size={self.batch_size}, max_target_length={self.max_target_length}, model_version={self.model_version}, processor={self.processor}, vision_encoder_decoder_model={self.vision_encoder_decoder_model}), eval_frequency={self.eval_frequency}, test_dataset={self.test_dataset}, log_dir={self.log_dir}, wandb_project={self.wandb_project})"
        if hasattr(self, "decoder"):
            return f"Config(learning_rate={self.learning_rate}, epochs={self.epochs}, batch_size={self.batch_size}, max_target_length={self.max_target_length}, model_version={self.model_version}, decoder={self.decoder}, tokenizer_type={self.tokenizer_type}, image processor={self.image_processor_type}, encoder={self.encoder}), eval_frequency={self.eval_frequency}, test_dataset={self.test_dataset}, log_dir={self.log_dir}, wandb_project={self.wandb_project})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        if hasattr(self, "processor") and hasattr(self, "vision_encoder_decoder_model"):
            return {
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "max_target_length": self.max_target_length,
                "model_version": self.model_version,
                "processor": self.processor,
                "vision_encoder_decoder_model": self.vision_encoder_decoder_model,
                "test_dataset": self.test_dataset,
                "wandb_project": self.wandb_project,
                "eval_frequency": self.eval_frequency
            }
        if (hasattr(self, "decoder")
                and hasattr(self, "tokenizer_type")
                and hasattr(self, "image_processor_type")
                and hasattr(self, "encoder")):
            return {
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "max_target_length": self.max_target_length,
                "model_version": self.model_version,
                "decoder": self.decoder,
                "tokenizer_type": self.tokenizer_type,
                "image_processor_type": self.image_processor_type,
                "encoder": self.encoder,
                "test_dataset": self.test_dataset,
                "wandb_project": self.wandb_project,
                "eval_frequency": self.eval_frequency
            }

    @staticmethod
    def from_dict(data):
        if "processor" in data:
            return Config(
                learning_rate=data["learning_rate"],
                epochs=data["epochs"],
                batch_size=data["batch_size"],
                max_target_length=data["max_target_length"],
                model_version=data["model_version"],
                processor=data["processor"],
                vision_encoder_decoder_model=data["vision_encoder_decoder_model"],
                test_dataset= data["test_dataset"] if len(data["test_dataset"]) > 0 else None,
                eval_frequency=data["eval_frequency"],
                wandb_project=data["wandb_project"]
            )
        if "decoder" in data:
            return Config(
                learning_rate=data["learning_rate"],
                epochs=data["epochs"],
                batch_size=data["batch_size"],
                max_target_length=data["max_target_length"],
                model_version=data["model_version"],
                decoder=data["decoder"],
                tokenizer_type=data["tokenizer_type"],
                image_processor_type=data["image_processor_type"],
                encoder=data["encoder"],
                test_dataset= data["test_dataset"] if len(data["test_dataset"]) > 0 else None,
                eval_frequency=data["eval_frequency"],
                wandb_project=data["wandb_project"]
            )

    @staticmethod
    def from_json(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return Config.from_dict(data)

    def to_json(self, file_path):
        with open(file_path, 'w') as json_file:
            json.dump(self.to_dict(), json_file, indent=4)
