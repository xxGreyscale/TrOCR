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
            processor=None,
            vision_encoder_decoder_model=None,
            encoder=None
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_target_length = max_target_length
        if tokenizer_type is not None and decoder is not None and encoder is not None:
            print(f"encoder: {encoder}, decoder: {decoder}, tokenizer_type: {tokenizer_type}")
            self.tokenizer_type = tokenizer_type
            self.decoder = decoder
            self.encoder = encoder
        if processor is not None and vision_encoder_decoder_model is not None:
            self.processor = processor
            self.vision_encoder_decoder_model = vision_encoder_decoder_model
        self.model_version = model_version
        self.num_epochs = num_epochs
