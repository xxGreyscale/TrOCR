import argparse
import logging
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
    parser.add_argument('--augment_data', action=argparse.BooleanOptionalAction,
                        help='Generate augmented dataset for training the model')
    parser.add_argument('--htr', action=argparse.BooleanOptionalAction,
                        help='Generate handwritten dataset for training HTR model')
    parser.add_argument('--printed', action=argparse.BooleanOptionalAction,
                        help='Generate printed dataset for training printed text recognition model')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction,
                        help='Train a model with pretrained weights, specify the model')
    parser.add_argument('--custom', action=argparse.BooleanOptionalAction, help='Model type')
    parser.add_argument('--files_path', nargs="+", help='Path to XML and images files')
    parser.add_argument('--save_dir', type=str, help='Directory to save the dataset')
    parser.add_argument('--lang', type=str, help='Language to use')
    parser.add_argument('--num_images', type=int, help='Number of images to generate')
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, help='Train the model')
    parser.add_argument('--eval_every', type=int, help='Evaluate the model every n epochs')
    parser.add_argument('--dataset_paths', nargs="+", help='Path to the dataset')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--max_target_length', type=int, help='Maximum target length')
    parser.add_argument('--decoder', type=str, help='Decoder type')
    parser.add_argument('--tokenizer_type', type=str, help='Tokenizer type')
    parser.add_argument('--encoder', type=str, help='Encoder type')
    parser.add_argument('--model_version', type=str, help='Model version')
    parser.add_argument('--processor', type=str, help='Processor path for the model')
    parser.add_argument('--vision_encoder_decoder_model', type=str, help='Vision Encoder Decoder model')
    parser.add_argument('--with_half_data', action=argparse.BooleanOptionalAction,
                        help='Get the first half of the dataset')

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
                     "Twitter", "Students' IP", "Uppsala", "Eleda Stadion"]
            # pages = ["Youtube", "Chelsea FC", "Uppsala universitet", "IK Sirius FK",
            #          "Google", "Svenska", "Hjärtdjur", "Storängen"] this is the original list
            generator = GenerateSyntheticPrintedDataset(pages=pages, target_dir=args.save_dir)
            generator.generate_dataset(args.num_images, augment_data=args.augment_data)
        else:
            logger.info("Invalid dataset type. Please specify the dataset type")

    if args.train:
        if args.dataset_paths is None:
            raise ValueError("Please specify the dataset path")
        if args.lr is None:
            raise ValueError("Please specify the learning rate")
        if args.epochs is None:
            raise ValueError("Please specify the number of epochs")
        if args.batch_size is None:
            raise ValueError("Please specify the batch size")
        if args.max_target_length is None:
            raise ValueError("Please specify the maximum target length")
        if args.model_version is None:
            raise ValueError("Please specify the model version")
        # Check if we are using a pretrained model or a custom model
        if args.pretrained is not None:
            logger.info("Using pretrained model...")
            transfer_learning(args)
        elif args.custom is not None:
            train(args)
        else:
            logger.error("Invalid model type. Please specify the model type")


def predict():
    pass


def evaluate():
    pass


def transfer_learning(args):
    if args.processor is None:
        raise ValueError("Please specify the trocr model")
    if args.vision_encoder_decoder_model is None:
        raise ValueError("Please specify the Vision Encoder Decoder model")
    try:
        config = Config(
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_target_length=args.max_target_length,
            processor=args.processor,
            vision_encoder_decoder_model=args.vision_encoder_decoder_model,
            model_version=args.model_version,
            num_epochs=args.epochs
        )
        # get the dataset
        logger.info("Getting the dataset...")
        data = CustomLoader(args.dataset_paths)
        data.generate_dataframe()
        # create the model
        logger.info("Creating the model...")
        model = TrOCR(config, args.save_dir)
        model.build_model_with_pretrained(args.processor, args.vision_encoder_decoder_model)
        # prepare the dataset and loader
        if args.with_half_data:
            train_dataloader, eval_dataloader = model.set_data_loader(*model.prepare_dataset(data.get_half_dataframe()))
        else:
            train_dataloader, eval_dataloader = model.set_data_loader(*model.prepare_dataset(data.get_dataframe()))
        # train the model
        model.train(train_dataloader, eval_dataloader, eval_every=args.eval_every)
    except Exception as e:
        logger.error(f"Error training model: {e}")


def train(args):
    if args.decoder is None:
        raise ValueError("Please specify the decoder type")
    if args.tokenizer_type is None:
        raise ValueError("Please specify the tokenizer type")
    if args.encoder is None:
        raise ValueError("Please specify the encoder type")
    try:
        # create the configuration with wandb
        # Initiate wandb
        config = Config(
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_target_length=args.max_target_length,
            decoder=args.decoder,
            tokenizer_type=args.tokenizer_type,
            encoder=args.encoder,
            model_version=args.model_version,
            num_epochs=args.epochs
        )
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
        model.train(train_dataloader, eval_dataloader, eval_every=args.eval_every)
    except Exception as e:
        logger.error(f"Error training model: {e}")


if __name__ == '__main__':
    main()
