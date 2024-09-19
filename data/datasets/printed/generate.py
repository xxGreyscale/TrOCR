import csv
import logging
import os
import traceback

import wikipedia
from wikipedia.exceptions import PageError
import numpy as np
import random
import re
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from data.datasets.printed.fonts import Fonts
from data.preprocessing.augmentation.transforms import CustomTransformation
from PIL import ImageOps
import time
from requests.exceptions import ConnectTimeout
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logger import setup_logger

logger = setup_logger("GenerateSyntheticPrintedDataset", "logs")

# CONSTANTS
API_KEY = "AIzaSyBb9yWCtp_L2sKJ-_qBEDeEZdGrZ4odvgA"


class GenerateSyntheticPrintedDataset:
    def __init__(self, pages=None, target_dir=None):
        """
        Generate a synthetic dataset for project printed dataset
        :param pages: Number of pages to get sentences from
        :param target_dir: Directory to save the dataset
        """
        self.fonts = Fonts()
        self.num_images = 0
        self.num_of_sentences = 0
        self.pages = pages
        self.target_dir = target_dir
        self.custom_transforms = CustomTransformation(
            min_noise_factor=0.1, max_noise_factor=0.2, sigma=5.0,
            random_noise_p=0.7, random_rotation_p=0.5,
            invert_p=0.2, elastic_grid_p=0.8, resize_p=0.5
        )

    @staticmethod
    def read_wikipedia_page(page_name, lang="sv"):
        """
        Read a Wikipedia page
        :param lang:
        :param page_name:
        :return:
        """
        wikipedia.set_lang(lang)
        search = wikipedia.search(page_name)
        page = wikipedia.page(auto_suggest=False, title=search[0])
        links = page.links
        return page.content, links

    @staticmethod
    def get_sentences(page_name, lang="sv"):
        content, links = GenerateSyntheticPrintedDataset.read_wikipedia_page(page_name, lang)
        for link in tqdm(links, desc=f"Reading Wikipedia pages links from {page_name}"):
            try:
                _content, _links = GenerateSyntheticPrintedDataset.read_wikipedia_page(link, lang)
                content += ". " + _content
            except ConnectTimeout:
                continue
            except PageError:
                continue
            except Exception:
                continue
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
        sentences = [GenerateSyntheticPrintedDataset.sanitize_sentence(str(sentence)) for sentence in sentences]
        return sentences

    @staticmethod
    def sanitize_sentence(sentence):
        """
        Sanitize a sentence
        :param sentence:
        :return:
        """
        if not isinstance(sentence, str):
            raise TypeError("Expected sentence to be a string!")
        sentence = re.sub(r'={2,}', '', sentence)
        sentence = re.sub(r'\[\d+]', '', sentence)
        sentence = re.sub(r'\([^)]*\)', '', sentence)
        sentence = re.sub(r'\s+\.', '.', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = sentence.strip()
        return sentence

    @staticmethod
    def get_sentences_with_retry(page_name, retries=5, delay=2):
        for _ in range(retries):
            try:
                sentences = GenerateSyntheticPrintedDataset.get_sentences(page_name)
                if sentences is not None and len(sentences) > 0:
                    return sentences
            except ConnectTimeout:
                time.sleep(delay)
            except Exception:
                time.sleep(delay)
        raise Exception(f"Failed to get sentences from {page_name} after {retries} attempts")

    def generate_sentence_from_pages(self, pages):
        self.pages = pages
        sentences = []

        # check if .csv file exists
        if os.path.exists(f"{self.target_dir}/sentences.csv"):
            with open(f"{self.target_dir}/sentences.csv", mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    # sanitize the sentence
                    sentence = GenerateSyntheticPrintedDataset.sanitize_sentence(str(row[0]))
                    sentences.append(sentence)
            return sentences
        else:
            def fetch_sentences(page_name):
                try:
                    __sentences = [_sentence for _sentence in
                                   GenerateSyntheticPrintedDataset.get_sentences_with_retry(page_name)
                                   if len(_sentence) < 110]
                    return __sentences
                except Exception as _e:
                    time.sleep(.5)

            with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                future_to_page = {executor.submit(fetch_sentences, page_name): page_name for page_name in pages}
                for future in as_completed(future_to_page):
                    page_name = future_to_page[future]
                    try:
                        _sentences = future.result()
                        sentences += _sentences
                    except Exception as e:
                        logger.log(msg=f"Error: Failed to get sentences from {page_name} with error: {e}",
                                   level=logging.ERROR)

            logger.info(f"Total sentences: {len(sentences)}")
            # save sentences to a .csv file
            # Check if the file exists
            # Ensure the target directory exists
            os.makedirs(self.target_dir, exist_ok=True)

            file_exists = os.path.isfile(f"{self.target_dir}/sentences.csv")
            with open(f"{self.target_dir}/sentences.csv", mode='a' if file_exists else 'w', newline='',
                      encoding='utf-8') as file:
                writer = csv.writer(file)
                # Write the header row if the file is being created
                if not file_exists:
                    writer.writerow(["sentence"])

                # Write the sentences
                for sentence in sentences:
                    writer.writerow([sentence])
            return sentences

    def pair_fonts_with_sentences(self, max_sentences=500000):
        """
        Pair fonts with sentences
        :return:
        """
        # pair at least 2 fonts with a sentence
        sentences = self.generate_sentence_from_pages(self.pages)

        # shuffle the sentences
        random.shuffle(sentences)
        fonts = self.fonts.get_random_fonts()
        paired_fonts = []

        def check_text_fits(__sentence, __font):
            _img = Image.new("RGB", (1, 1), "white")
            draw = ImageDraw.Draw(_img)
            # Ensure font is correctly created
            ___font = None
            if isinstance(__font, tuple):
                _font_name, _font_path, _font_size = __font
                try:
                    ___font = ImageFont.truetype(font_path, size=font_size)
                except IOError:
                    ___font = ImageFont.load_default()

            # Get text bounding box
            if ___font is None:
                raise ValueError("Font is not defined")
            bbox = draw.textbbox((0, 0), sentence, font=___font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            return text_width > 0 and text_height > 0

        def split_sentence(__sentence, max_length=110):
            words = __sentence.split()
            __lines = []
            current_line = []

            for word in words:
                if len(' '.join(current_line + [word])) <= max_length:
                    current_line.append(word)
                else:
                    __lines.append(' '.join(current_line))
                    current_line = [word]

            if current_line:
                __lines.append(' '.join(current_line))

            return __lines

        try:
            for i, sentence in tqdm(enumerate(sentences), total=len(sentences), desc="Pairing fonts with sentences"):
                if i >= max_sentences:
                    break
                lines = split_sentence(sentence)
                for line in lines:
                    while True:
                        font = random.choice(fonts)
                        font_size = random.randint(20, 40)
                        font_name, font_path = font
                        # check if the font works with the sentence
                        try:
                            if check_text_fits(line, (font_name, font_path, font_size)):
                                paired_fonts.append(
                                    (sentence, f"image_{str(i).zfill(7)}", (font_name, font_path, font_size)))
                                break
                        except Exception as e:
                            logger.error(f"Failed to setup image: {e}")
                            continue

            return paired_fonts
        except Exception as e:
            logger.error(f"Failed to pair fonts with sentences: {e}")

    @staticmethod
    def setup_image(sentence, __font):
        font_name, font_path, font_size = __font
        img = Image.new("RGB", (1, 1), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, size=font_size)
        except IOError:
            # logger.error(f"Failed to load font: {font_path}")
            font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox((0, 0), sentence, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        img = Image.new("RGB", (text_width, text_height), "white")
        padding = tuple(random.randint(5, 20) for _ in range(4))
        img = ImageOps.expand(img, padding, fill="white")
        draw = ImageDraw.Draw(img)

        fill_color = tuple(np.random.randint(0, 100) for _ in range(3))
        draw.text((padding[0], padding[1]), sentence, font=font, fill=fill_color)

        np_img = np.array(img)

        if np.isnan(np_img).any() or np.isinf(np_img).any():
            np_img = np.nan_to_num(np_img, nan=0, posinf=255, neginf=0)
            print("Warning: Invalid values encountered and replaced.")

        img = Image.fromarray(np_img)
        return img

    def generate_image(self, sentence, img_name, font, augment_data=False):
        """
        Generate an image with a random font and a random Swedish sentence
        :param sentence:
        :param img_name:
        :param font:
        :param augment_data:
        :return:
        """
        # font_name, font_path, font_size = font
        try:
            img = self.setup_image(sentence, font)
            if augment_data:
                try:
                    img = self.custom_transforms.data_transformer(img)
                except Exception as e:
                    logger.error(f"Data augmentation failed: {e}")
                    return None, None

            images_dir = os.path.join(self.target_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            save_path = os.path.join(images_dir, f"{img_name}.jpeg")
            img.save(save_path, format='JPEG')

            return img_name, sentence
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return None, None
        except IsADirectoryError as e:
            logger.error(f"Directory error: {e}")
            return None, None
        except OSError as e:
            logger.error(f"OS error: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Failed to generate image '{img_name}': {e}")
            return None, None

    def generate_dataset(self, num_images, augment_data=False):
        try:
            failed = 0
            generated_images = 0
            logger.info(f"Generating {num_images} images...")
            paired_fonts = self.pair_fonts_with_sentences(max_sentences=num_images)

            if not os.path.exists(self.target_dir):
                os.makedirs(self.target_dir)

            # Create an iterable of arguments with img_name properly defined
            if augment_data:
                args = [(sentence, img_name, font, augment_data)
                        for i, (sentence, img_name, font) in enumerate(paired_fonts)]
            else:
                args = [(sentence, img_name, font)
                        for i, (sentence, img_name, font) in enumerate(paired_fonts)]

            results = []
            with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                futures = {executor.submit(self.generate_image_wrapper, arg) for arg in args}
                for future in tqdm(as_completed(futures), total=len(paired_fonts), desc="Processing images"):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                            generated_images += 1
                    except Exception as e:
                        logger.error(f"Error processing result: {e}")
                        failed += 1

            # Open CSV file once for writing
            logger.info("Writing image labels to CSV file...")
            file_exists = os.path.isfile(f"{self.target_dir}/image_labels_dataset.csv")
            with open(f"{self.target_dir}/image_labels_dataset.csv",
                      mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["image_path", "label"])
                for img_name, sentence in results:
                    writer.writerow([f"images/{img_name}.jpeg", sentence])

            if failed > 0:
                logger.info(f"Failed to generate {failed} images")
            logger.info(f"Successfully generated {generated_images} images")
            return None

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
        except IsADirectoryError as e:
            logger.error(f"Directory error: {e}")
        except csv.Error as e:
            logger.error(f"CSV error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}\n{traceback.format_exc()}")

    def generate_image_wrapper(self, args):
        sentence, img_name, font, augment = args
        return self.generate_image(sentence, img_name, font, augment)
