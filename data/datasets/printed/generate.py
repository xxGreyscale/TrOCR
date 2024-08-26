import csv
import functools
import logging
import os
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
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logger import setup_logger

logger = setup_logger("GenerateSyntheticPrintedDataset", "logs")

# CONSTANTS
API_KEY = "AIzaSyBb9yWCtp_L2sKJ-_qBEDeEZdGrZ4odvgA"


class GenerateSyntheticPrintedDataset:
    def __init__(self, pages=None, target_dir=None):
        """
        Generate a synthetic printed dataset
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
            #     runtime error keep going for now
            except PageError:
                continue
                # runtime error keep going for now
                # logger.error(f"Failed to get content from {link} because it's not a Wikipedia page.")
            except:
                continue
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
        sentences = [sentence.replace('\n', ' ').replace('\t', ' ') for sentence in sentences]
        return sentences

    @staticmethod
    def get_sentences_with_retry(page_name, retries=5, delay=2):
        for _ in range(retries):
            try:
                sentences = GenerateSyntheticPrintedDataset.get_sentences(page_name)
                if sentences is not None and len(sentences) > 0:
                    return sentences
            except ConnectTimeout:
                time.sleep(delay)
            except:
                time.sleep(delay)
        raise Exception(f"Failed to get sentences from {page_name} after {retries} attempts")

    def generate_sentence_from_pages(self, pages):
        self.pages = pages
        sentences = []

        def fetch_sentences(page_name):
            try:
                __sentences = [sentence for sentence in
                               GenerateSyntheticPrintedDataset.get_sentences_with_retry(page_name)
                               if len(sentence) < 110]
                return __sentences
            except Exception as _e:
                time.sleep(.5)
                # runtime error keep going for now
                # logger.error(f"Failed to get sentences from {page_name} with error: {_e}")

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
        return sentences

    def pair_fonts_with_sentences(self):
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
        try:
            for i, sentence in tqdm(enumerate(sentences), total=len(sentences), desc="Pairing fonts with sentences"):
                font = random.choice(fonts)
                font_name, font_path = font
                font_size = random.randint(20, 40)
                # Create the directory if it doesn't exist
                paired_fonts.append((sentence, f"image_{str(i).zfill(7)}", (font_name, font_path, font_size)))
            # get random font
            return paired_fonts
        except Exception as e:
            logger.error(f"Failed to pair fonts with sentences: {e}")

    @staticmethod
    def setup_image(sentence, font):
        font_name, font_path, font_size = font
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
        return img

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
            img = self.setup_image(sentence, font)
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
            generated_images = 0
            logger.info(f"Generating {num_images} images...")
            paired_fonts = self.pair_fonts_with_sentences()
            if not os.path.exists(self.target_dir):
                os.makedirs(self.target_dir)
            file_exists = os.path.isfile(f"{self.target_dir}/image_labels_dataset.csv")
            with open(f"{self.target_dir}/image_labels_dataset.csv",
                      mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["image_path", "label"])
                with Pool(cpu_count()) as p:
                    for i, (sentence, img_name, font) in enumerate(paired_fonts):
                        generate_image_func = functools.partial(self.generate_image, img_name=img_name, font=font,
                                                                augment_data=augment_data)
                        result = p.apply_async(generate_image_func, (sentence,))
                        font_name, sentence = result.get()
                        if font_name is None or sentence is None:
                            failed += 1
                            continue
                        writer.writerow([f"{self.target_dir}/images/{img_name}.jpeg", sentence])
                        generated_images += 1
                        if generated_images == num_images:
                            break
            if failed > 0:
                logger.info(f"Failed to generate {failed} images")
            logger.info(f"Dataset generation completed!")
            return None
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
        except IsADirectoryError as e:
            logger.error(f"Directory error: {e}")
        except csv.Error as e:
            logger.error(f"CSV error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
