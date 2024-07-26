# This is a sample Python script.
import argparse
from data.datasets.handwritten.generate_dataset import GenerateDemoktatiLabourerDataset
from data.datasets.printed.generate import GenerateSyntheticPrintedDataset
from data.loader.custom_loader import CustomLoader
from models.TrOCR import TrOCR
from utils.config import Config

import warnings

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
    parser.add_argument('--htr', action=argparse.BooleanOptionalAction,
                        help='Generate handwritten dataset for training HTR model')
    parser.add_argument('--printed', action=argparse.BooleanOptionalAction,
                        help='Generate printed dataset for training printed text recognition model')
    parser.add_argument('--files_path', nargs="+", help='Path to XML and images files')
    parser.add_argument('--save_dir', type=str, help='Directory to save the dataset')
    parser.add_argument('--num_images', type=int, help='Number of images to generate')
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, help='Train the model')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--max_target_length', type=int, help='Maximum target length')
    parser.add_argument('--decoder', type=str, help='Decoder type')
    parser.add_argument('--tokenizer_type', type=str, help='Tokenizer type')
    parser.add_argument('--encode', type=str, help='Encoder type')
    parser.add_argument('--model_version', type=str, help='Model version')

    args = parser.parse_args()
    if args.generate_dataset:
        if args.htr:
            try:
                generator = GenerateDemoktatiLabourerDataset()
                generator.generate_dataset(args.files_path, args.save_dir)
            except Exception as e:
                print(f"Error generating dataset: {e}")
        elif args.printed:
            pages = ["Youtube", "Uppsala universitet", "Svenska", "Hjärtdjur", "Storängen"]
            generator = GenerateSyntheticPrintedDataset(pages=pages, target_dir=args.save_dir)
            generator.generate_dataset(args.num_images)
        else:
            print("Invalid dataset type. Please specify the dataset type")

    if args.train:

        if args.dataset_path is None:
            raise ValueError("Please specify the dataset path")
        if args.save_dir is None:
            raise ValueError("Please specify the save directory")
        if args.lr is None:
            raise ValueError("Please specify the learning rate")
        if args.epochs is None:
            raise ValueError("Please specify the number of epochs")
        if args.batch_size is None:
            raise ValueError("Please specify the batch size")
        if args.max_target_length is None:
            raise ValueError("Please specify the maximum target length")
        if args.decoder is None:
            raise ValueError("Please specify the decoder type")
        if args.tokenizer_type is None:
            raise ValueError("Please specify the tokenizer type")
        if args.encode is None:
            raise ValueError("Please specify the encoder type")
        if args.model_version is None:
            raise ValueError("Please specify the model version")
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
                encoder=args.encode,
                model_version=args.model_version,
                num_epochs=args.epochs
            )
            # get the dataset
            # data = CustomLoader("datasets/handwritten/augmented/image_labels_dataset.csv")
            print("Getting the dataset...")
            data = CustomLoader(args.dataset_path)
            data.generate_dataframe()
            # create the model
            print("Creating the model...")
            model = TrOCR(config, args.save_dir)
            model.build_model()
            # prepare the dataset and loader
            train_dataloader, eval_dataloader = model.set_data_loader(*model.prepare_dataset(data.get_dataframe()))
            # train the model
            model.train(train_dataloader, eval_dataloader)
        except Exception as e:
            print(f"Error training model: {e}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
