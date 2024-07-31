import csv
import os
import wikipedia
from wikipedia.exceptions import PageError
import numpy as np
import random
import re
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from data.preprocessing.augmentation.transforms import CustomTransformation
from PIL import ImageOps
import time
from requests.exceptions import ConnectTimeout
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONSTANTS
API_KEY = "AIzaSyBb9yWCtp_L2sKJ-_qBEDeEZdGrZ4odvgA"


class GenerateSyntheticPrintedDataset:
    def __init__(self, pages=None, lang="sv", target_dir=None):
        """
        Generate a synthetic printed dataset
        :param pages: Number of pages to get sentences from
        :param lang: Language to use, Default is Swedish
        :param target_dir: Directory to save the dataset
        """
        self.num_images = 0
        self.num_of_sentences = 0
        self.wikipedia = wikipedia
        self.pages = pages
        self.target_dir = target_dir
        self.lang = lang
        self.custom_transforms = CustomTransformation(
            min_noise_factor=0.1, max_noise_factor=0.2, sigma=5.0,
            random_noise_p=0.7, random_rotation_p=0.5,
            invert_p=0.2, elastic_grid_p=0.8, resize_p=0.5
        )

    def read_wikipedia_page(self, page_name):
        """
        Read a Wikipedia page
        :param page_name:
        :return:
        """
        self.wikipedia.set_lang(self.lang)
        search = self.wikipedia.search(page_name)
        page = self.wikipedia.page(auto_suggest=False, title=search[0])
        links = page.links
        return page.content, links

    def get_sentences(self, page_name):
        content, links = self.read_wikipedia_page(page_name)
        for link in tqdm(links, desc=f"Reading Wikipedia pages links from {page_name}"):
            try:
                _content, _links = self.read_wikipedia_page(link)
                content += ". " + _content
            except ConnectTimeout:
                time.sleep(5)
            except PageError:
                logger.info(f"Failed to get content from {link} because it's not a Wikipedia page.")
            except Exception as e:
                logger.error(f"Failed to get content from {link} with error: {e}")
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
        return sentences

    def get_sentences_with_retry(self, page_name, retries=5, delay=5):
        for _ in range(retries):
            try:
                return self.get_sentences(page_name)
            except ConnectTimeout:
                time.sleep(delay)
        raise Exception(f"Failed to get sentences from {page_name} after {retries} attempts")

    def generate_sentence_from_pages(self, pages):
        # pages to search from
        self.pages = pages
        sentences = []
        for i, page_name in enumerate(pages):
            try:
                if len(sentences) >= self.num_images:
                    self.num_of_sentences = len(sentences)
                    break
                logger.info(f"Getting sentences from {page_name}: {i + 1}/{len(pages)}")
                _sentences = self.get_sentences_with_retry(page_name)
                # filter text with more than 110 characters
                _sentences = [sentence for sentence in _sentences if len(sentence) < 110]
                sentences += _sentences
            except Exception as e:
                logger.error(f"Failed to get sentences from {page_name} with error: {e}")
        logger.info(f"Total sentences: {len(sentences)}")
        return sentences

    @staticmethod
    def get_google_fonts():
        """
        Get Google fonts
        :return: Array list of (font_name, font_path)
        """
        missed_fonts = 0
        fonts = []
        if not os.path.exists("datasets/fonts"):
            raise IsADirectoryError("Fonts directory not found")

        # Get fonts info .csv from the fonts directory
        # open font-info.csv
        if not os.path.exists("datasets/fonts/font-info.csv"):
            raise FileNotFoundError("Font info file not found")

        # open the file
        with open("datasets/fonts/font-info.csv", "r") as file:
            reader = csv.reader(file)
            fonts_meta = [row for row in reader]

        # filter handwritten, monospace, medieval, serif and grotesque fonts
        fonts_filter = ["handwritten", "monospace", "medieval", "serif", "grotesque"]
        filtered_fonts = [font_meta for font_meta in fonts_meta if font_meta[1] not in fonts_filter]

        # the fonts name matches the files in directory datasets/fonts
        for font in filtered_fonts[1:]:
            if not os.path.exists(f"datasets/fonts/{font[0].replace(' ','').lower().strip()}"):
                # logger.warn(f"Skipping since, font file for {font[0]} is not found")
                missed_fonts += 1
                continue

            # get file in the directory with .tff extension
            font_files = [file for file in os.listdir(f"datasets/fonts/{font[0].replace(' ','').lower().strip()}")
                          if file.endswith("Regular.ttf")]
            if len(font_files) == 0:
                missed_fonts += 1
                # logger.warn(f"Font file for {font[0]} regular not found")
                continue
            font = (font[0], f"datasets/fonts/{font[0].replace(' ','').lower().strip()}/{font_files[0]}")
            fonts.append(font)
        logger.info(f"Missed fonts: {missed_fonts} out of {len(filtered_fonts)}")
        return fonts

    def get_random_fonts(self):
        """
        Get random fonts
        :return:
        """
        return random.choices(self.get_google_fonts(), k=10)

    def pair_fonts_with_sentences(self):
        """
        Pair fonts with sentences
        :return:
        """
        # pair at least 2 fonts with a sentence
        sentences = self.generate_sentence_from_pages(self.pages)
        # shuffle the sentences
        random.shuffle(sentences)
        fonts = self.get_random_fonts()
        paired_fonts = []
        for i, sentence in tqdm(enumerate(sentences), total=len(sentences), desc="Pairing fonts with sentences"):
            font = random.choice(fonts)
            font_name, font_path = font
            font_size = random.randint(20, 40)
            # Create the directory if it doesn't exist
            paired_fonts.append((sentence, f"image_{str(i).zfill(7)}", (font_name, font_path, font_size)))
        # get random font
        return paired_fonts

    # Create a function that generates an image with a random font and a random swedish sentence
    def generate_image(self, sentence, img_name, font, augment_data=False):
        """
        Generate an image with a random font and a random Swedish sentence
        :param sentence:
        :param img_name:
        :param font:
        :param augment_data:
        :return:
        """
        font_name, font_path, font_size = font
        try:
            img = Image.new("RGB", (1, 1), "white")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, size=font_size)

            # Get text bounding box
            bbox = draw.textbbox((0, 0), sentence, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            img = Image.new("RGB", (text_width, text_height), "white")
            # create image
            padding = tuple(random.randint(5, 20) for _ in range(4))
            img = ImageOps.expand(img, padding, fill="white")
            draw = ImageDraw.Draw(img)
            # create one line sentence
            # place image in the center
            draw.text((padding[0], padding[1]), sentence,
                      font=font, fill=tuple([np.random.randint(0, 100)] * 3), )
            if augment_data:
                img = self.custom_transforms.data_transformer(img)
            # save image
            # Check if the directory exists
            if not os.path.exists(f"{self.target_dir}/images"):
                os.makedirs(os.path.join(self.target_dir, "images"))
            img.save(os.path.join(self.target_dir, "images", f"{img_name}.jpeg"))
            return font_name, sentence
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            return None, None

    def generate_dataset(self, num_images, augment_data=False):
        """
        Generate the synthetic printed dataset
        :param num_images:
        :param augment_data:
        :return:
        """
        """
        @todo: Could be a recursive function since number of images is not always guaranteed to a specified number
        """
        try:
            failed = 0
            self.num_images = num_images
            paired_fonts = self.pair_fonts_with_sentences()
            # create a csv file to image paths to label pair
            # check if available first
            if not os.path.exists(self.target_dir):
                os.makedirs(self.target_dir)
            # check if the csv file exists
            file_exists = os.path.isfile(f"{self.target_dir}/image_labels_dataset.csv")
            with open(f"{self.target_dir}/image_labels_dataset.csv",
                      mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["image_path", "label"])
                for i, (sentence, img_name, font) in tqdm(enumerate(paired_fonts),
                                                          total=self.num_images, desc="Generating dataset"):
                    font_name, sentence = self.generate_image(sentence, img_name, font, augment_data)
                    if font_name is None or sentence is None:
                        failed += 1
                        continue
                    writer.writerow([f"{self.target_dir}/images/{img_name}.jpeg", sentence])
                    if i == self.num_images:
                        break
            if failed > 0:
                logger.info(f"Failed to generate {failed} images")
            logger.info(f"Dataset generation completed!")
            return None
        except Exception as e:
            print(f"Failed to generate dataset: {e}")
