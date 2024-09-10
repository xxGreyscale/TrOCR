import argparse
import json
import logging

# import wandb

from data.datasets.handwritten.generate_dataset import GenerateDemoktatiLabourerDataset
from data.datasets.printed.generate import GenerateSyntheticPrintedDataset
from data.loader.custom_loader import CustomLoader
from models.trocr.TrOCR import TrOCR
from utils.config import Config
import warnings


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Vision Transformers don't support Flash Attention yet
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", message=".*We strongly recommend passing in an `attention_mask`.*")


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--generate_dataset', action=argparse.BooleanOptionalAction,
                        help='Generate dataset for training the model')
    parser.add_argument('--train', action=argparse.BooleanOptionalAction,
                        help='Train the model')
    parser.add_argument('--augment_data', action=argparse.BooleanOptionalAction,
                        help='Generate augmented dataset for training the model')
    parser.add_argument('--htr', action=argparse.BooleanOptionalAction,
                        help='Generate handwritten dataset for training HTR model')
    parser.add_argument('--printed', action=argparse.BooleanOptionalAction,
                        help='Generate printed dataset for training printed text recognition model')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction,
                        help='Train a model with pretrained weights, specify the model')
    parser.add_argument('--files_path', nargs="+", help='Path to XML and images files')
    parser.add_argument('--save_dir', type=str, help='Directory to save the dataset')
    parser.add_argument('--lang', type=str, help='Language to use')
    parser.add_argument('--num_images', type=int, help='Number of images to generate')
    parser.add_argument('--custom', action=argparse.BooleanOptionalAction, help='Model type')
    parser.add_argument('--dataset_paths', nargs="+", help='Path to the dataset')
    parser.add_argument('--with_half_data', action=argparse.BooleanOptionalAction,
                        help='Get the first half of the dataset')
    parser.add_argument('--train_config_path', type=str, help='Path to the training configuration file')

    def read_json_file(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        _config = Config(
            learning_rate=data.get('lr'),
            epochs=data.get('epochs'),
            batch_size=data.get('batch_size'),
            max_target_length=data.get('max_target_length'),
            model_version=data.get('model_version'),
            num_epochs=data.get('epochs'),  # Assuming num_epochs is same as epochs
            decoder=data.get('decoder'),
            tokenizer_type=data.get('tokenizer_type'),
            encoder=data.get('encoder'),
            image_processor_type=data.get('image_processor_type'),
            eval_frequency=data.get('eval_frequency'), # Evaluate the model every n epochs
            processor=data.get('processor'),
            vision_encoder_decoder_model=data.get('vision_encoder_decoder_model'),
            log_dir=data.get('log_dir')
        )
        return _config

    args = parser.parse_args()

    if args.generate_dataset:
        if args.htr:
            try:
                generator = GenerateDemoktatiLabourerDataset()
                generator.generate_dataset(args.files_path, args.save_dir, augment_data=args.augment_data)
            except Exception as e:
                logger.info(f"Error generating dataset: {e}")
        elif args.printed:
            pages = ["Facebook", "Liverpool FC", "Kungliga Tekniska högskolan", "Malmö FF",
                     "Twitter", "Students' IP", "Uppsala", "Eleda Stadion", "Youtube", "Chelsea FC",
                     "Uppsala universitet", "IK Sirius FK", "Google", "Svenska", "Hjärtdjur", "Storängen"]
            generator = GenerateSyntheticPrintedDataset(pages=pages, target_dir=args.save_dir)
            generator.generate_dataset(args.num_images, augment_data=args.augment_data)
        else:
            logger.info("Invalid dataset type. Please specify the dataset type")

    if args.train:
        # wandb.login(key="MY_KEY")
        config = read_json_file(args.train_config_path)
        if args.dataset_paths is None:
            raise ValueError("Please specify the dataset path")
        if config.learning_rate is None:
            raise ValueError("Please specify the learning rate")
        if config.epochs is None:
            raise ValueError("Please specify the number of epochs")
        if config.batch_size is None:
            raise ValueError("Please specify the batch size")
        if config.max_target_length is None:
            raise ValueError("Please specify the maximum target length")
        if config.model_version is None:
            raise ValueError("Please specify the model version")
        # Check if we are using a pretrained model or a custom model
        if args.pretrained is not None:
            logger.info("Using pretrained model...")
            transfer_learning(config, args)
        elif args.custom is not None:
            train(config, args)
        else:
            logger.error("Invalid model type. Please specify the model type")


def predict():
    pass


def evaluate():
    pass


def transfer_learning(config: Config, args):
    if config.processor is None:
        raise ValueError("Please specify the trocr model")
    if config.vision_encoder_decoder_model is None:
        raise ValueError("Please specify the Vision Encoder Decoder model")
    try:
        # get the dataset
        logger.info("Getting the dataset...")
        data = CustomLoader(args.dataset_paths)
        data.generate_dataframe()
        # create the model
        logger.info("Creating the model...")
        model = TrOCR(config, args.save_dir)
        model.build_model_with_pretrained(config.processor, config.vision_encoder_decoder_model)
        # prepare the dataset and loader
        if args.with_half_data:
            train_dataloader, eval_dataloader = model.set_data_loader(*model.prepare_dataset(data.get_half_dataframe()))
        else:
            train_dataloader, eval_dataloader = model.set_data_loader(*model.prepare_dataset(data.get_dataframe()))
        # train the model
        model.train(train_dataloader, eval_dataloader, eval_every=config.eval_frequency)
    except Exception as e:
        logger.error(f"Error training model: {e}")


def train(config: Config, args):
    if config.decoder is None:
        raise ValueError("Please specify the decoder type")
    if config.tokenizer_type is None:
        raise ValueError("Please specify the tokenizer type")
    if config.encoder is None:
        raise ValueError("Please specify the encoder type")
    try:
        # get the dataset
        # data = CustomLoader("datasets/handwritten/augmented/image_labels_dataset.csv")
        logger.info("Getting the dataset...")
        data = CustomLoader(args.dataset_paths)
        data.generate_dataframe()
        # create the model
        logger.info("Creating the model...")
        model = TrOCR(config, args.save_dir)
        model.build_model()
        # prepare the dataset and loader
        train_dataloader, eval_dataloader = model.set_data_loader(*model.prepare_dataset(data.get_dataframe()))
        # train the model
        model.train(train_dataloader, eval_dataloader, eval_every=config.eval_frequency)
    except Exception as e:
        logger.error(f"Error training model: {e}")


if __name__ == '__main__':
    main()
