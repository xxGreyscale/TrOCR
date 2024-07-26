# Get the data
import os
import chardet
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from data.loader.common.loader import LoaderInterface


class CustomLoader(LoaderInterface):
    def __init__(self, path=None):
        super().__init__(path)
        self.path = path
        # We expect a dataframe
        self.dataframe: any = None
        self.maximum_char_length: int = 0

    def generate_dataframe(self):
        """
        Generate a dataframe from the CSV file,
        This will generate a dataframe with the following columns:
        file_name: The path to the image file
        text: The text in the image
        """
        try:
            if os.path.isfile(self.path):
                print('File exists')
                with open(self.path, 'rb') as f:
                    result = chardet.detect(f.read())
                    _encoding = result['encoding']
                    print(f'Encoding: {_encoding}')
                self.dataframe = pd.read_csv(self.path, encoding="utf-8").dropna()
                self.dataframe.rename(columns={'image_path': 'file_name', 'label': 'text'}, inplace=True)
                self.maximum_char_length = self.calculate_max_character_length()
            else:
                print("Make sure the data is already prepared")
        except FileNotFoundError:
            print(f'File does not exist at: {self.path}')

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
