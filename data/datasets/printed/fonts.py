import os
import random
import csv

from utils.logger import setup_logger

logger = setup_logger("Fonts", "logs")


class Fonts:

    @staticmethod
    def get_google_fonts():
        """
        Get Google fonts
        :return: Array list of (font_name, font_path)
        """
        missed_fonts = 0
        fonts = []
        if not os.path.exists("raw/fonts"):
            raise IsADirectoryError("Fonts directory not found")

        # Get fonts info .csv from the fonts directory
        # open font-info.csv
        if not os.path.exists("raw/fonts/font-info.csv"):
            raise FileNotFoundError("Font info file not found")

        # open the file
        with open("raw/fonts/font-info.csv", "r") as file:
            reader = csv.reader(file)
            fonts_meta = [row for row in reader]

        # filter handwritten, monospace, medieval, serif and grotesque fonts
        fonts_filter = ["handwritten", "monospace", "medieval", "serif", "grotesque"]
        filtered_fonts = [font_meta for font_meta in fonts_meta if font_meta[1] not in fonts_filter]

        # the fonts name matches the files in directory raw/fonts
        for font in filtered_fonts[1:]:
            if not os.path.exists(f"raw/fonts/{font[0].replace(' ', '').lower().strip()}"):
                # logger.warn(f"Skipping since, font file for {font[0]} is not found")
                missed_fonts += 1
                continue

            # get file in the directory with .tff extension
            font_files = [file for file in os.listdir(f"raw/fonts/{font[0].replace(' ', '').lower().strip()}")
                          if file.endswith("Regular.ttf")]
            if len(font_files) == 0:
                missed_fonts += 1
                # logger.warn(f"Font file for {font[0]} regular not found")
                continue
            font = (font[0], f"raw/fonts/{font[0].replace(' ', '').lower().strip()}/{font_files[0]}")
            fonts.append(font)
        logger.info(f"Missed fonts: {missed_fonts} out of {len(filtered_fonts)}")
        return fonts

    def get_random_fonts(self):
        """
        Get random fonts
        :return:
        """
        return random.choices(self.get_google_fonts(), k=10)

    def default_font(self):
        """
        Get the default font
        :return:
        """
        return self.get_google_fonts()[0]
