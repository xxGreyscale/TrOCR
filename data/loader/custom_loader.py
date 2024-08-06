# Get the data
import os
import chardet
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

from data.loader.common.loader import LoaderInterface


class CustomLoader(LoaderInterface):
    def __init__(self, paths=None):
        super().__init__(paths)
        self.paths = paths if paths is not None else []
        self.dataframe: any = None
        self.maximum_char_length: int = 0

    def generate_dataframe(self, columns=None):
        """
        Generate a dataframe from the CSV files,
        This will generate a dataframe with the following columns:
        file_name: The path to the image file
        text: The text in the image
        """
        if columns is None:
            columns = ['image_path', 'label']
        combined_dataframe = pd.DataFrame()
        try:
            for path in self.paths:
                if os.path.isfile(path):
                    print(f'File exists: {path}')
                    with open(path, 'rb') as f:
                        result = chardet.detect(f.read())
                        _encoding = result['encoding']
                        print(f'Encoding: {_encoding}')
                    temp_dataframe = pd.read_csv(path, encoding="utf-8").dropna()
                    temp_dataframe.rename(columns={columns[0]: 'file_name', columns[1]: 'text'}, inplace=True)
                    combined_dataframe = pd.concat([combined_dataframe, temp_dataframe], ignore_index=True)
                else:
                    print(f"File does not exist: {path}")
            self.dataframe = combined_dataframe
            self.maximum_char_length = self.calculate_max_character_length()
        except FileNotFoundError as e:
            print(f'Error: {e}')

    def calculate_max_character_length(self):
        max_len = 0
        for _, row in tqdm(self.dataframe.iterrows(), total=self.dataframe.shape[0], desc='Calculating max length'):
            text = row['text']
            if len(text) > max_len:
                max_len = len(text)
        return max_len

    def get_dataframe(self):
        """
        Get the generated dataframe from dataset
        :return: DataFrame
        """
        return self.dataframe

    def get_half_dataframe(self):
        """
        Get the first half of the generated dataframe from dataset
        :return: DataFrame
        """
        return self.dataframe.iloc[:len(self.dataframe) // 2]

    def get_maximum_char_length(self):
        return self.maximum_char_length

    # Generate a histogram of the character length of each row
    def generate_histogram(self):
        self.dataframe['text_length'] = self.dataframe['text'].apply(lambda x: len(x))
        hist = self.dataframe['text_length'].hist()

        plt.title('Histogram of Text Lengths')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')

        # Add a key to each bar in the histogram
        for i in hist.patches:
            plt.text(i.get_x() + i.get_width() / 2, i.get_height(), str(int(i.get_height())), fontsize=11, ha='center')

        return hist

    def visualize_images(self, num_images=5, padding=None):
        """
        Visualize images from the dataframe.

        Parameters:
        dataframe (pd.DataFrame): DataFrame containing image file paths.
        num_images (int): Number of images to display.
        """
        cell_padding = {'x': 25, 'y': 5} if padding is None else {'x': padding[0], 'y': padding[1]}
        images_per_row = 3
        num_rows = (num_images + images_per_row - 1) // images_per_row  # Calculate the number of rows needed
        plt.figure(figsize=(cell_padding['x'], cell_padding['y'] * num_rows))
        # Adjust figure size based on the number of rows
        for i in range(num_images):
            img_path = self.dataframe.iloc[i]['file_name']
            # add ../ to the path to make it work
            img_path = "..\\" + img_path  # remove this line if the path in the dataframe is correct
            img = Image.open(img_path).convert('RGB')
            plt.subplot(num_rows, images_per_row, i + 1)
            plt.imshow(img)
            plt.title(self.dataframe.iloc[i]['text'], fontsize=16)
            plt.axis('off')
        plt.subplots_adjust(wspace=0.2, hspace=0.2)  # Adjust spacing between subplots
        plt.show()
