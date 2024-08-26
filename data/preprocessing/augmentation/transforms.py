import numpy as np
import torch
from torchvision import transforms


class Augmentations:
    class RandomNoise(object):
        def __init__(self, min_noise_factor=0.1, max_noise_factor=0.2):
            self.min_noise_factor = min_noise_factor
            self.max_noise_factor = max_noise_factor

        def __call__(self, tensor):
            noise = torch.randn_like(tensor[0]) * np.random.uniform(self.min_noise_factor, self.max_noise_factor)
            tensor = torch.clamp(tensor + noise.unsqueeze(0), 0, 1)
            return self.validate_tensor(tensor)

        @staticmethod
        def validate_tensor(tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                # print("Invalid values detected in RandomNoise!")
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
            return torch.clamp(tensor, 0, 1)

    class ElasticGrid(object):
        def __init__(self, alpha=1, sigma=0.5):
            self.alpha = alpha
            self.sigma = sigma

        def __call__(self, tensor):
            tensor = transforms.ElasticTransform(
                alpha=max(2, 9 - (20 / 1000) * tensor.shape[0]),
                sigma=self.sigma,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=1
            )(tensor)
            return self.validate_tensor(tensor)

        @staticmethod
        def validate_tensor(tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                # print("Invalid values detected in ElasticGrid!")
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
            return torch.clamp(tensor, 0, 1)

    class Resize(object):
        def __init__(self, horizontal_ratio=(0.3, 1.5), vertical_ratio=(0.9, 1.1)):
            self.horizontal_ratio = horizontal_ratio
            self.vertical_ratio = vertical_ratio

        def __call__(self, tensor):
            _, h, w = tensor.shape
            tensor = transforms.Resize(
                size=(int(h * np.random.uniform(*self.vertical_ratio)),
                      int(w * np.random.uniform(*self.horizontal_ratio))),
                interpolation=transforms.InterpolationMode.BILINEAR
            )(tensor)
            return self.validate_tensor(tensor)

        @staticmethod
        def validate_tensor(tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                # print("Invalid values detected in Resize!")
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
            return torch.clamp(tensor, 0, 1)


class CustomTransformation:
    def __init__(self, min_noise_factor=0.01, max_noise_factor=0.05,
                 sigma=0.5, random_noise_p=0.5, random_rotation_p=0.5,
                 invert_p=0.5, elastic_grid_p=0.5, resize_p=0.5):
        self.min_noise_factor = min_noise_factor
        self.max_noise_factor = max_noise_factor
        self.sigma = sigma
        self.random_noise_p = random_noise_p
        self.random_rotation_p = random_rotation_p
        self.invert_p = invert_p
        self.elastic_grid_p = elastic_grid_p
        self.resize_p = resize_p
        self.data_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.RandomApply([transforms.RandomRotation(degrees=1, fill=1)], p=self.random_rotation_p),
            transforms.RandomApply([Augmentations.RandomNoise(
                min_noise_factor=self.min_noise_factor,
                max_noise_factor=self.max_noise_factor)],
                p=self.random_noise_p),
            transforms.RandomApply([transforms.functional.invert], p=self.invert_p),
            transforms.RandomApply([Augmentations.ElasticGrid(sigma=self.sigma)], p=self.elastic_grid_p),
            transforms.RandomApply([Augmentations.Resize()], p=self.resize_p),
            transforms.ToPILImage(),
        ])
