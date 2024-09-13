import json


class DatasetGenerationConfig:
    def __init__(self,
                 pages=None,
                 num_images=1000,
                 lang="se",
                 augment_data=False,
                 ):
        if pages is None:
            pages = [""]
        self.lang = lang
        self.pages = pages
        self.num_images = num_images
        self.augment_data = augment_data

    def __str__(self):
        return f"DatasetGenerationConfig(lang={self.lang}, num_images={self.num_images}, pages={self.pages}, augment_data={self.augment_data})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "lang": self.lang,
            "num_images": self.num_images,
            "pages": self.pages,
            "augment_data": self.augment_data
        }

    @staticmethod
    def from_dict(data):
        return DatasetGenerationConfig(
            lang=data["lang"],
            num_images=data["num_images"],
            pages=data["pages"],
            augment_data=data["augment_data"]
        )

    @staticmethod
    def from_json(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return DatasetGenerationConfig.from_dict(data)

    def to_json(self, file_path):
        with open(file_path, 'w') as json_file:
            json.dump(self.to_dict(), json_file, indent=4)
