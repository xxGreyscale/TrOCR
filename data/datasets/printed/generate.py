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
        self.language = "sv"
        self.target_dir = target_dir
        self.custom_transforms = CustomTransformation(
            min_noise_factor=0.1, max_noise_factor=0.2, sigma=5.0,
            random_noise_p=0.3, random_rotation_p=0.2,
            invert_p=0.1, elastic_grid_p=0.4, resize_p=0.3
        )

    @staticmethod
    def fetch_wikipedia_page(page_name, lang="sv"):
        wikipedia.set_lang(lang)
        search = wikipedia.search(page_name)
        if not search:
            logger.info(f"Failed to find search results for '{page_name}'")
            return [], []
        page = wikipedia.page(search[0])
        links = page.links if page.links else []
        return page.content, links

    @staticmethod
    def read_wikipedia_random_pages(lang="sv", retries=5, delay=2):
        """
        Read a random Wikipedia page with retry mechanism
        :param lang: Language of the Wikipedia page
        :param retries: Number of retries for network errors
        :param delay: Delay between retries
        :return: List of sanitized sentences
        """
        wikipedia.set_lang(lang)
        for attempt in range(retries):
            try:
                random_page = wikipedia.random(1)
                page = wikipedia.page(random_page)
                content = page.content
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
                sentences = [GenerateSyntheticPrintedDataset.sanitize_sentence(str(sentence)) for sentence in sentences]
                return sentences
            except (ConnectTimeout, ConnectionError) as e:
                time.sleep(delay)
            except wikipedia.exceptions.DisambiguationError as e:
                time.sleep(delay)
            except Exception as e:
                time.sleep(delay)

    def get_sentences(page_name, lang="sv"):
        content, links = GenerateSyntheticPrintedDataset.fetch_wikipedia_page(page_name, lang)
        for link in tqdm(links, desc=f"Reading Wikipedia pages links from {page_name}"):
            try:
                _content, _links = GenerateSyntheticPrintedDataset.fetch_wikipedia_page(link, lang)
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
            except wikipedia.exceptions.DisambiguationError as e:
                time.sleep(delay)
            except wikipedia.exceptions.PageError as e:
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
                    __sentences = GenerateSyntheticPrintedDataset.get_sentences_with_retry(page_name)
                    if __sentences is None:
                        logger.error(f"Failed to get sentences from {page_name}: returned None")
                        return []
                    return [_sentence for _sentence in __sentences if len(_sentence) < 110]
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

            progress_bar = tqdm(total=(self.num_of_sentences - len(sentences)), desc="Getting sentences from random pages....")
            while len(sentences) < self.num_of_sentences:
                try:
                    _sentences = GenerateSyntheticPrintedDataset.read_wikipedia_random_pages()
                    sentences += _sentences
                    progress_bar.update(len(_sentences))
                except Exception as e:
                    continue
            progress_bar.close()


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
        self.num_of_sentences = max_sentences
        sentences = self.generate_sentence_from_pages(self.pages)
        random.shuffle(sentences)
        fonts = self.fonts.get_random_fonts()
        paired_fonts = []
        def pair_sentence_with_font(args):
            i, sentence = args
            font = random.choice(fonts)
            font_size = random.randint(18, 40)
            font_name, font_path = font
            return sentence, f"image_{str(i).zfill(7)}", (font_name, font_path, font_size)

        try:
            with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                futures = {executor.submit(pair_sentence_with_font, (i, sentence)): sentence for i, sentence in enumerate(sentences[:max_sentences])}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Pairing fonts with sentences"):
                    result = future.result()
                    if result:
                        paired_fonts.append(result)
            return paired_fonts
        except Exception as e:
            logger.error(f"Failed to pair fonts with sentences: {e}")
            return []

    @staticmethod
    def setup_image(sentence, __font):
        def text_fits(text_width, text_height):
            return text_width > 0 and text_height > 0

        def get_bounding_box(sentence, font, draw):
            # Get text bounding box
            bbox = draw.textbbox((0, 0), sentence, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            return text_width, text_height

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

        def draw_text(text_width, text_height, sentence, font):
            #  TODO: Implement the split_sentence function, for each line in lines, draw the text
            img = Image.new("RGB", (text_width, text_height), "white")
            max_padding_x = max(2, (text_width - 1) // 2)
            max_padding_y = max(2, (text_height - 1) // 2)
            # Generate random padding values within the allowable range
            padding = (
                random.randint(2, max_padding_x),
                random.randint(2, max_padding_y),
                random.randint(2, max_padding_x),
                random.randint(2, max_padding_y)
            )
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

        font_name, font_path, font_size = __font
        img = Image.new("RGB", (1, 1), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, size=font_size)
        except IOError:
            font = ImageFont.load_default()

        text_width, text_height = get_bounding_box(sentence, font, draw)
        images = []
        if not text_fits(text_width, text_height):
        #     split the sentence
            lines = split_sentence(sentence)
            for line in lines:
                text_width, text_height = get_bounding_box(line, font, draw)
                if text_fits(text_width, text_height):
                    img = draw_text(text_width, text_height, line, font)
                    images.append(img)
        else:
            img = draw_text(text_width, text_height, sentence, font)
            images.append(img)
        return images

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
            imgs = self.setup_image(sentence, font)
            results = []
            for i, img in enumerate(imgs):
                if augment_data:
                    try:
                        processed_img = self.custom_transforms.data_transformer(img)
                    except Exception as e:
                        # logger.error(f"Data augmentation failed: {e}")
                        # Just use the original image
                        processed_img = img # Use the original image
                else:
                    processed_img = img
                img_name = f"{img_name}_{str(i).zfill(3)}"
                images_dir = os.path.join(self.target_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                save_path = os.path.join(images_dir, f"{img_name}.jpeg")
                processed_img.save(save_path, format='JPEG')
                results.append((img_name, sentence))

            return results
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
                args = [(sentence, img_name, font, augment_data)
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
                for result in tqdm(results, desc="Writing image labels to CSV file"):
                    if result != (None, None):
                        for img_name, sentence in result:
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
