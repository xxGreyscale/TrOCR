# Get the data
import os
import chardet
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from data.loader.common.loader import LoaderInterface


class CustomLoader(LoaderInterface):
    def __init__(self, paths=None):
        super().__init__(paths)
        self.paths = paths if paths is not None else []
        self.dataframe: any = None
        self.maximum_char_length: int = 0

    def generate_dataframe(self):
        """
        Generate a dataframe from the CSV files,
        This will generate a dataframe with the following columns:
        file_name: The path to the image file
        text: The text in the image
        """
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
                    temp_dataframe.rename(columns={'image_path': 'file_name', 'label': 'text'}, inplace=True)
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
