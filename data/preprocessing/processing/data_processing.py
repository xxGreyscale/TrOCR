import tensorflow as tf
import os
import csv
from tqdm import tqdm


def __init__():
    print("Data Preparation module loaded!")


class DataPreparation:
    def __init__(
            self,
            _image_width,
            _image_height,
            _batch_size,
            _padding_token,
            custom_transforms=None,
            save_dir=None):
        self.image_width = _image_width
        self.image_height = _image_height
        self.save_dir = save_dir
        self.batch_size = _batch_size
        self.padding_token = _padding_token
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.characters = set()
        self.max_len = 0
        self.num_to_char = None
        self.char_to_num = None
        self.train_data = False
        self.custom_transforms = custom_transforms

    @staticmethod
    def distortion_free_resize(image, img_size):
        w, h = img_size
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        # Check tha amount of padding needed to be done.
        pad_height = abs(h - tf.shape(image)[0])
        pad_width = abs(w - tf.shape(image)[1])

        # Only necessary if you want to do same amount of padding on both sides.
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
            constant_values=255  # Change this to the value that corresponds to transparency in your images
        )

        image = tf.transpose(image, perm=[1, 0, 2])
        # rotate the image 270 degrees
        image = tf.image.rot90(image, k=3)
        image = tf.image.flip_left_right(image)
        return image

    @staticmethod
    def draw_sentence_bound_box(image, boxes, _batch_size=64):
        cropped_images = []
        unsuccessful_crops = 0
        for box in boxes:
            try:
                coords = box.get('coords', [])
                if all(isinstance(coord, (list, tuple)) and len(coord) == 2 for coord in coords):
                    x_values = [coord[0] for coord in coords]
                    y_values = [coord[1] for coord in coords]
                else:
                    print("Invalid coordinates")
                    continue
                bound_box = (min(y_values) - 70, min(x_values),
                             (max(y_values) - min(y_values)) + 60, max(x_values) - min(x_values))

                # Check the size of the image
                if isinstance(image, tf.Tensor) and hasattr(image, 'shape'):
                    image_height, image_width, _ = image.shape
                else:
                    print("Invalid image")
                    continue

                # Adjust the bounding box if it falls outside the image dimensions
                bound_box = (max(0, bound_box[0]), max(0, bound_box[1]),
                             min(image_height, bound_box[2]), min(image_width, bound_box[3]))

                # Crop the image
                cropped_img = tf.image.crop_to_bounding_box(image, *bound_box)
                cropped_images.append([box.get("text", ""), cropped_img])
            except Exception as e:
                unsuccessful_crops += 1
                print(f"Error cropping image: {e}, box: {box}")
        return cropped_images, unsuccessful_crops

    def crop_sentences(self, _image_name, _image, _boxes):
        __unsuccessful_crops = 0
        image_path = _image
        boxes = _boxes
        cropped_images, _unsuccessful_crops = self.draw_sentence_bound_box(image_path, boxes)
        __unsuccessful_crops += _unsuccessful_crops
        # Directory to save cropped images
        try:
            save_dir = self.save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # CSV file to save paths and labels
            csv_file = os.path.join(save_dir, "image_labels_dataset.csv")

            # Check if the CSV file already exists
            file_exists = os.path.isfile(csv_file)

            with open(csv_file, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # Write header only if the file does not exist
                if not file_exists:
                    writer.writerow(["image_path", "label"])  # Write header

                for i, cropped_image in enumerate(cropped_images):
                    _label = cropped_image[0]
                    _image = cropped_image[1]
                    _label = _label.strip()

                    # Save the cropped image
                    file_destinations_dir = os.path.join(save_dir, _image_name)
                    image_filename = f"cropped_image_{i}.png"
                    if not os.path.exists(file_destinations_dir):
                        os.makedirs(file_destinations_dir)
                    cropped_image_path = self.save_image(_image, image_filename, file_destinations_dir)
                    # Append the image path and label to the CSV file
                    writer.writerow([cropped_image_path, _label])
            tf.keras.backend.clear_session()
            return __unsuccessful_crops
        except Exception as e:
            print('Error creating directory:', e)
            return -1

    def save_image(self, _image, image_filename, file_destinations_dir):
        try:
            image_save_path = os.path.join(file_destinations_dir, image_filename)
            img = self.preprocess_image(_image)
            img.save(image_save_path)
            return image_save_path
        except Exception as e:
            print('Error saving image:', e)

    def preprocess_image(self, image):
        _img_size = (self.image_width, self.image_height)
        # augment data
        image = self.distortion_free_resize(image, _img_size)
        image = tf.keras.preprocessing.image.array_to_img(image)
        image = self.custom_transforms.data_transformer(image)
        return image

    def load_and_process_image(self, _image_name, image_path, boxes, _sentences=False):
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, 1)
            _unsuccessful_crops = self.crop_sentences(_image_name, image, boxes)
            return _unsuccessful_crops, "Processed successfully!"
        except Exception as e:
            print('Error loading and processing image:', e)
            return -1, "An error occurred while loading and processing image!"

    def prepare_dataset(self, _annotations, _train_data=False, _sentences=False):
        _unsuccessful_crops = 0
        _target_folder_dir = self.save_dir
        # check if csv file already exists
        csv_file = os.path.join(_target_folder_dir, "image_labels_dataset.csv")
        # if it exists delete it
        if os.path.exists(csv_file):
            os.remove(csv_file)
        try:
            for i, annotation in tqdm(enumerate(_annotations), total=len(_annotations), desc="Processing images"):
                image_path = annotation['image_path']
                image_name = image_path.split("\\")[-1].split(".")[0].strip()
                boxes = annotation['boxes']
                self.train_data = _train_data
                unsuccessful_crops, result = self.load_and_process_image(image_name, image_path, boxes, _sentences)
                _unsuccessful_crops += unsuccessful_crops
            print(f"Total unsuccessful crops: {_unsuccessful_crops}")
            print("Dataset preparation completed!")
            csv_file = os.path.join(_target_folder_dir, "image_labels_dataset.csv")
            return csv_file
        except Exception as e:
            print('Error preparing dataset:', e)
            return -1
