import os
import json
import torchvision

from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from .datasets import ImageDataset
from .samplers import SubSetSampler, RandomIdentitySampler
from .image import Market1501
from .transforms import RandomErasing
from base import BaseDataManger


class DataManger(BaseDataManger):
    def __init__(self, config, phase='train'):
        super().__init__()
        self.datasource = Market1501(
            root_dir=config['data_dir'],
            download=config['download'],
            extract=config['extract'])

        self.transform_train = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(padding=10, fill=0, padding_mode='constant'),
            transforms.RandomCrop(size=(256, 128), padding=None),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            RandomErasing(config['probability'], mean=[0.0, 0.0, 0.0])
        ])

        self.transfrom_val = transforms.Compose([
            transforms.Resize(size=(256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(size=(256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.training_set = ImageDataset(self.datasource.get_data('train'), transform=self.transform_train)

        self.test_set = {
            'query': ImageDataset(self.datasource.get_data('query'), transform=self.transform_test),
            'gallery': ImageDataset(self.datasource.get_data('gallery'), transform=self.transform_test)
        }

        if phase == 'train':
            self.train_sampler, self.val_sampler = RandomIdentitySampler(
                self.training_set,
                batch_size=config['batch_size'],
                num_instances=config['num_instances']).split(rate=1-config['validation_split'])

            self.train_loader = DataLoader(
                dataset=self.training_set,
                sampler=self.train_sampler,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory']
            )

            self.val_loader = DataLoader(
                dataset=self.training_set,
                sampler=self.val_sampler,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory']
            )
        elif phase == 'test':
            self.test_loader = {
                'query': DataLoader(self.test_set['query'], batch_size=32, shuffle=False, drop_last=False),
                'gallery': DataLoader(self.test_set['gallery'], batch_size=32, shuffle=False, drop_last=False),
            }
        else:
            raise ValueError("phase == train or phase == test")

    def get_dataloader(self, dataset):
        if dataset not in ['train', 'val', 'query', 'gallery']:
            raise ValueError("Error dataset paramaster, dataset in [train, query, gallery]")
        if dataset == 'train':
            return self.train_loader
        elif dataset == 'val':
            return self.val_loader
        elif dataset == 'query':
            return self.test_loader['query']
        elif dataset == 'gallery':
            return self.test_loader['gallery']

    def get_datasets(self, dataset: str):
        if dataset not in ['train', 'query', 'gallery']:
            raise ValueError("Error dataset paramaster, dataset in [train, query, gallery]")
        if dataset == 'train':
            return self.training_set
        elif dataset == 'query':
            return self.test_set['query']
        elif dataset == 'gallery':
            return self.test_set['gallery']
