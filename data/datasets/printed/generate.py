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
import requests
import json
import time
from requests.exceptions import ConnectTimeout

# CONSTANTS
API_KEY = "AIzaSyBb9yWCtp_L2sKJ-_qBEDeEZdGrZ4odvgA"


class GenerateSyntheticPrintedDataset:
    def __init__(self, pages=None, target_dir=None):
        self.num_images = 0
        self.num_of_sentences = 0
        self.wikipedia = wikipedia
        self.pages = pages
        self.target_dir = target_dir
        self.custom_transforms = CustomTransformation(
            min_noise_factor=0.1, max_noise_factor=0.2, sigma=0.5,
            random_noise_p=0.7, random_rotation_p=0.5,
            invert_p=0.5, elastic_grid_p=0.8, resize_p=0.5
        )

    def read_wikipedia_page(self, page_name):
        self.wikipedia.set_lang("sv")
        search = self.wikipedia.search(page_name)
        page = self.wikipedia.page(auto_suggest=False, title=search[0])
        links = page.links
        return page.content, links

    def get_swedish_sentences(self, page_name):
        content, links = self.read_wikipedia_page(page_name)
        for link in tqdm(links, desc=f"Reading Wikipedia pages links from {page_name}"):
            try:
                _content, _links = self.read_wikipedia_page(link)
                content += ". " + _content
            except ConnectTimeout:
                time.sleep(5)
            except PageError:
                print(f"Failed to get content from {link} because it's not a Wikipedia page.")
            except Exception as e:
                print(f"Failed to get content from {link} with error: {e}")
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
        return sentences

    def get_swedish_sentences_with_retry(self, page_name, retries=5, delay=5):
        for _ in range(retries):
            try:
                return self.get_swedish_sentences(page_name)
            except ConnectTimeout:
                time.sleep(delay)
        raise Exception(f"Failed to get sentences from {page_name} after {retries} attempts")

    def generate_swedish_sentence_from_pages(self, pages):
        # pages to search from
        self.pages = pages
        sentences = []
        for i, page_name in enumerate(pages):
            try:
                # filter text with more than 110 characters
                if len(sentences) >= self.num_images:
                    self.num_of_sentences = len(sentences)
                    break
                print(f"Getting sentences from {page_name}: {i + 1}/{len(pages)}")
                _sentences = self.get_swedish_sentences_with_retry(page_name)
                _sentences = [sentence for sentence in _sentences if len(sentence) < 110]
                sentences += _sentences
            except Exception as e:
                print(f"Failed to get sentences from {page_name} with error: {e}")
        print(f"Total sentences: {len(sentences)}")
        return sentences

    @staticmethod
    def download_google_fonts():
        # get fonts from google fonts api
        try:
            url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={API_KEY}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # fetch every font and save it in the fonts directory
                fonts = json.loads(response.text)
                if not os.path.exists("datasets/fonts"):
                    os.makedirs("datasets/fonts")
                for font in tqdm(fonts["items"], desc="Downloading fonts"):
                    font_name = font["family"]
                    font_weight = random.choice(font["variants"])
                    font_url = font["files"][font_weight]
                    response = requests.get(font_url, timeout=10)
                    if response.status_code == 200:
                        with open(f"datasets/fonts/{font_name}_{font_weight}.ttf", "wb") as file:
                            file.write(response.content)
                    else:
                        raise Exception(f"Failed to download font {font_name}")
            else:
                raise Exception(f"Failed to download fonts: {response.text}")
        except Exception as e:
            time.sleep(1)
            raise Exception(f"Failed to download fonts: {e}")

    def get_google_fonts(self):
        if not os.path.exists("datasets/fonts"):
            print("Downloading fonts from Google Fonts API...")
            self.download_google_fonts()
        # Get locally stored fonts
        # get all fonts available in the file and return font name to path pair
        fonts = [(font.split(".")[0], os.path.join("datasets/fonts", font))
                 for font in os.listdir("datasets/fonts")]
        return fonts

    def get_random_fonts(self):
        fonts = self.get_google_fonts()
        # return random fonts
        random_fonts = random.choices(fonts, k=5)
        return random_fonts

    def pair_fonts_with_sentences(self):
        # pair at least 2 fonts with a sentence
        sentences = self.generate_swedish_sentence_from_pages(self.pages)
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
    def generate_image(self, sentence, img_name, font):
        font_name, font_path, font_size = font
        try:
            img = Image.new("RGB", (1, 1), "white")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, size=font_size)
            # get text size
            text_width = int(draw.textlength(sentence))
            text_height = font_size
            img = Image.new("RGB", (text_width, text_height), "white")
            # create image
            padding = tuple(random.randint(5, 20) for _ in range(4))
            img = ImageOps.expand(img, padding, fill="white")
            draw = ImageDraw.Draw(img)
            # create one line sentence
            # place image in the center
            draw.text((padding[0], padding[1]), sentence,
                      font=font, fill=tuple([np.random.randint(0, 100)] * 3), )
            img = self.custom_transforms.data_transformer(img)
            # save image
            # Check if the directory exists
            if not os.path.exists(f"{self.target_dir}/images"):
                os.makedirs(os.path.join(self.target_dir, "images"))
            img.save(os.path.join(self.target_dir, "images", f"{img_name}.jpeg"))
            return font_name, sentence
        except Exception as e:
            return None, None

    def generate_dataset(self, num_images):
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
                    font_name, sentence = self.generate_image(sentence, img_name, font)
                    if font_name is None or sentence is None:
                        failed += 1
                        continue
                    writer.writerow([f"{self.target_dir}/images/{img_name}.jpeg", sentence])
                    if i == self.num_images:
                        break
            if failed > 0:
                print(f"Failed to generate {failed} images")
            print(f"Dataset generation completed!")
            return None
        except Exception as e:
            print(f"Failed to generate dataset: {e}")
