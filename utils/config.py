class Config:
    def __init__(
            self,
            learning_rate,
            epochs,
            batch_size,
            max_target_length,
            model_version,
            num_epochs,
            decoder=None,
            tokenizer_type=None,
            image_processor_type=None,
            processor=None,
            vision_encoder_decoder_model=None,
            encoder=None,
            eval_frequency=1,
            log_dir=None  # logs for wandb

    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_target_length = max_target_length
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
        self.num_epochs = num_epochs
        self.eval_frequency = eval_frequency
        self.log_dir = log_dir

    def __str__(self):
        if hasattr(self, "processor"):
            return f"Config(learning_rate={self.learning_rate}, epochs={self.epochs}, batch_size={self.batch_size}, max_target_length={self.max_target_length}, model_version={self.model_version}, num_epochs={self.num_epochs}, processor={self.processor}, vision_encoder_decoder_model={self.vision_encoder_decoder_model}), eval_frequency={self.eval_frequency})"
        if hasattr(self, "decoder"):
            return f"Config(learning_rate={self.learning_rate}, epochs={self.epochs}, batch_size={self.batch_size}, max_target_length={self.max_target_length}, model_version={self.model_version}, num_epochs={self.num_epochs}, decoder={self.decoder}, tokenizer_type={self.tokenizer_type}, image processor={self.image_processor_type}, encoder={self.encoder}), eval_frequency={self.eval_frequency})"
