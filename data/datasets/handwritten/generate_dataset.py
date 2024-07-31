# Make a file that has an entry point to be accessed by console
# This will be the main file that will be run
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import gc
from tqdm import tqdm

from data.preprocessing.augmentation.transforms import CustomTransformation
from data.preprocessing.processing.data_processing import DataPreparation

# Constants
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
PADDING_TOKEN = 128
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 256


class GenerateDemoktatiLabourerDataset:
    # # look for all xml files that exists in all the subdirectories
    # of the given path, these xml files exist in the alto
    # folder. Only take xml files in the alto folder]
    def __init__(self):
        self.custom_transforms = CustomTransformation(
            min_noise_factor=0.05, max_noise_factor=0.15, sigma=5.0,
            random_noise_p=0.2, random_rotation_p=0.3,
            invert_p=0.2, elastic_grid_p=0.4, resize_p=0.3
        )

    @staticmethod
    def get_word_anno_xml_files(_path):
        for root, dirs, files in os.walk(_path):
            if 'page' not in root:
                continue
            for file in files:
                if file.endswith('.xml'):
                    yield os.path.join(root, file)

    @staticmethod
    def get_all_images(_path):
        _images = []
        for root, dirs, files in os.walk(_path):
            for file in files:
                if file.endswith('.jpg'):
                    yield {'path': os.path.join(root, file), 'name': file}

    @staticmethod
    def read_annotations(_xml_file, _images):
        ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        tree = ET.parse(_xml_file)
        root = tree.getroot()
        bound_boxes = []
        image_path = ''
        try:
            page_element = root.find('.//ns:Page', namespaces=ns)
            if page_element is not None:
                temp_image_path = page_element.get('imageFilename')
                for image in _images:
                    if temp_image_path in image['path']:
                        image_path = image['path']
                        break
            else:
                print(f"No Page element found in {_xml_file}")
                return image_path, bound_boxes
            for textLine in root.iter('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextLine'):
                # Extract the coordinates of the bounding box
                coords = textLine.find('ns:Coords', namespaces=ns).get('points')
                coords = [(int(x.split(',')[0]), int(x.split(',')[1])) for x in coords.split()]

                # Extract the text
                unicode = textLine.find('ns:TextEquiv/ns:Unicode', namespaces=ns)
                if unicode is not None and unicode.text is not None and unicode.text.strip():
                    text = unicode.text
                    box = {"text": text, "coords": coords}
                    bound_boxes.append(box)
                else:
                    # If no text is found, skip this box
                    continue
        except Exception as e:
            print(f"Error reading {_xml_file}: {e}")
        return image_path, bound_boxes

    def read_all_annotations(self, _xml_files, _images):
        _annotations = []
        for xml_file in tqdm(_xml_files, total=len(_xml_files), desc="Reading annotations"):
            # print(f"Reading annotations from {xml_file}")
            image_path, bound_box = self.read_annotations(xml_file, _images)
            _annotations.append({'image_path': image_path, 'boxes': bound_box})
        print(f"Successful annotated: {len(_annotations)} images")
        return _annotations

    def generate_dataset(self, files_path, save_dir, augment_data=False):
        # Get xml files and images
        xml_files = []
        images = []
        for path in tqdm(files_path, desc="Getting xml files and images"):
            if not os.path.exists(path):
                print(f"Path {path} does not exist")
                exit(1)
            xml_files.extend(list(self.get_word_anno_xml_files(path)))
            images.extend(list(self.get_all_images(path)))
        # Read annotations
        data_prep = DataPreparation(
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            BATCH_SIZE,
            PADDING_TOKEN,
            self.custom_transforms,
            save_dir=save_dir,
            augment_data=augment_data
        )
        # Prepare the dataset
        annotations = self.read_all_annotations(xml_files, images)
        annotations = [item for item in annotations if item['boxes']]
        data_prep.prepare_dataset(annotations, True, True)
        # Clear session and collect garbage
        del annotations
        gc.collect()
