class Config:
    def __init__(
            self, learning_rate, epochs, batch_size, max_target_length,
            decoder, tokenizer_type, encoder, model_version, num_epochs
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_target_length = max_target_length
        self.decoder = decoder
        self.tokenizer_type = tokenizer_type
        self.encoder = encoder
        self.model_version = model_version
        self.num_epochs = num_epochs
