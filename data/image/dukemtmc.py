import sys
sys.path.append('.')

from utils import download_file_from_google_drive
import os
import requests
import tarfile
import zipfile
import re
import glob
from tqdm import tqdm

class DukeMTMC_Reid(object):
    dataset_dir = 'dukemtmc_reid'
    dataset_id = '12nfb2yrdU3AuF3SKqLnJDyYeDqOflMVM'
    file_name = 'DukeMTMC-reID.zip'

    def __init__(self, root_dir='datasets', download=True, extract=True, re_label_on_train=True):
        self.root_dir = root_dir
        if download:
            print("Downloading!")
            self.file_name = self._download()
            print("Downloaded!")
        if extract:
            print("Extracting!")
            self._extract()
            print("Extracted!")

        self.data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')

        self.train_dir = os.path.join(self.data_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.data_dir, 'query')
        self.gallery_dir = os.path.join(self.data_dir, 'bounding_box_test')

        self.pid_container = dict()
        self.camid_containter = dict()
        self.frames_container = dict()

        print("Processing on train directory!")
        self.train, self.pid_container['train'], self.camid_containter['train'], self.frames_container['train'] = self.process_dir(
            self.train_dir, relabel=re_label_on_train)

        print("Processing on query directory!")
        self.query, self.pid_container['query'], self.camid_containter['query'], self.frames_container['query'] = self.process_dir(
            self.query_dir, relabel=False)

        print("Processing on gallery directory!")
        self.gallery, self.pid_container['gallery'], self.camid_containter['gallery'], self.frames_container['gallery'] = self.process_dir(
            self.gallery_dir, relabel=False)

    def get_data(self, mode='train'):
        if mode == 'train':
            return self.train
        elif mode == 'query':
            return self.query
        elif mode == 'gallery':
            return self.gallery
        else:
            raise ValueError('mode error')

    def process_dir(self, path, relabel):
        data = []
        pattern = re.compile(r'([-\d]+)_c(\d)_f([-\d]+)')

        with tqdm(total=len(os.listdir(path)*2)) as pbar:
            pid_container = set()
            camid_containter = set()
            frames_container = set()

            for img in os.listdir(path):
                name, ext = os.path.splitext(img)
                if ext == '.jpg':
                    img_path = os.path.join(path, img)
                    person_id, camera_id, frame = map(int, pattern.search(name).groups())
                    pid_container.add(person_id)
                    camid_containter.add(camera_id)
                    frames_container.add(frame)
                pbar.update(1)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            for img in os.listdir(path):
                name, ext = os.path.splitext(img)
                if ext == '.jpg':
                    img_path = os.path.join(path, img)
                    person_id, camera_id, frame = map(int, pattern.search(name).groups())
                    if relabel:
                        person_id = pid2label[person_id]
                    data.append((img_path, person_id, camera_id))
                pbar.update(1)
        return data, pid_container, camid_containter, frames_container

    def _download(self):
        os.makedirs(os.path.join(self.root_dir,self.dataset_dir, 'raw'), exist_ok=True)
        return download_file_from_google_drive(self.dataset_id, os.path.join(self.root_dir, self.dataset_dir, 'raw'))

    def _extract(self):
        file_path = os.path.join(self.root_dir, self.dataset_dir, 'raw', self.file_name)
        extract_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
        if self._exists(extract_dir):
            return
        try:
            tar = tarfile.open(file_path)
            os.makedirs(extract_dir, exist_ok=True)
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                tar.extract(member=member, path=extract_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(file_path, 'r')
            for member in tqdm(iterable=zip_ref.infolist(), total=len(zip_ref.infolist())):
                zip_ref.extract(member=member, path=extract_dir)
            zip_ref.close()

    def _exists(self, extract_dir):
        if os.path.exists(os.path.join(extract_dir, 'bounding_box_train')) \
            and os.path.exists(os.path.join(extract_dir, 'bounding_box_test')) \
            and os.path.exists(os.path.join(extract_dir, 'query')):
            return True
        return False

    def get_num_classes(self, dataset: str):
        if dataset not in ['train', 'query', 'gallery']:
            raise ValueError(
                "Error dataset paramaster, dataset in [train, query, gallery]")
        return len(self.pid_container[dataset])

    def get_num_camera(self, dataset: str):
        if dataset not in ['train', 'query', 'gallery']:
            raise ValueError(
                "Error dataset paramaster, dataset in [train, query, gallery]")
        return len(self.camid_containter[dataset])

    def get_name_dataset(self):
        return self.file_name.split('.zip')[0]

if __name__ == "__main__":
    dukemtmc = DukeMTMC_Reid(root_dir='/home/hien/Documents/datasets', download=True, extract=True)
    print('Train: len: {}, num_class: {}, num_camera: {}'.format(len(dukemtmc.train), dukemtmc.get_num_classes('train'), dukemtmc.get_num_camera('train')))
    print('Query: len: {}, num_class: {}, num_camera: {}'.format(len(dukemtmc.query), dukemtmc.get_num_classes('query'), dukemtmc.get_num_camera('query')))
    print('Gallery: len: {}, num_class: {}, num_camera: {}'.format(len(dukemtmc.gallery), dukemtmc.get_num_classes('gallery'), dukemtmc.get_num_camera('gallery')))

