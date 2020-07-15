import sys
sys.path.append('.')

import os
import requests
import tarfile
import zipfile
import re
import glob
from tqdm import tqdm

from utils import download_file_from_google_drive as down_gd, download_with_url

class Market1501(object):
    dataset_dir = 'market1501'
    dataset_id = '12pEaAd1pDVW0Rbpdr8wUwar0K6pGu9SV'
    file_name = 'Market-1501-v15.09.15.zip'
    google_drive_api = 'AIzaSyAVfS-7Dy34a3WjWgR509o-u_3Of59zizo'

    def __init__(self, root_dir='datasets', download=True, extract=True, re_label_on_train=True):
        self.root_dir = root_dir
        if download:
            print("Downloading!")
            self._download()
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
        pattern = re.compile(r'([-\d]+)_c(\d)s(\d)_([-\d]+)')

        with tqdm(total=len(os.listdir(path)*2)) as pbar:
            pid_container = set()
            camid_containter = set()
            frames_container = set()

            for img in os.listdir(path):
                name, ext = os.path.splitext(img)
                if ext == '.jpg':
                    img_path = os.path.join(path, img)
                    person_id, camera_id, seq, frame = map(int, pattern.search(name).groups())
                    if person_id == -1:
                        pbar.update(1)
                        continue
                    pid_container.add(person_id)
                    camid_containter.add(camera_id)
                    frames_container.add(self._re_frame(camera_id, seq, frame))
                pbar.update(1)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            for img in os.listdir(path):
                name, ext = os.path.splitext(img)
                if ext == '.jpg':
                    img_path = os.path.join(path, img)
                    person_id, camera_id, seq, frame = map(
                        int, pattern.search(name).groups())
                    if person_id == -1:
                        pbar.update(1)
                        continue
                    if relabel:
                        person_id = pid2label[person_id]
                    data.append((img_path, person_id, camera_id))
                pbar.update(1)
        return data, pid_container, camid_containter, frames_container

    def _download(self):
        os.makedirs(os.path.join(self.root_dir,
                                 self.dataset_dir, 'raw'), exist_ok=True)
        download_with_url(self.google_drive_api, self.dataset_id, os.path.join(self.root_dir, self.dataset_dir, 'raw'), self.file_name)

    def _extract(self):
        file_path = os.path.join(
            self.root_dir, self.dataset_dir, 'raw', self.file_name)
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

    def _re_frame(self, cam, seq, frame):
        """ Re frames on market1501.
            more info here: https://github.com/Wanggcong/Spatial-Temporal-Re-identification/issues/10
        """
        if seq == 1:
            return frame
        dict_cam_seq_max = {
            11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346, 17: 0, 18: 0,
            21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0,
            31: 161708, 32: 161769, 33: 104469, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0,
            41: 72107, 42: 72373, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0,
            51: 161095, 52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
            61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0, 67: 0, 68: 0}
        
        re_frame = 0
        for i in range(1, seq):
            re_frame += dict_cam_seq_max[int(str(cam) + str(i))]
        return re_frame + frame

    def get_name_dataset(self):
        return self.file_name.split('.zip')[0]

if __name__ == "__main__":
    market1501 = Market1501(root_dir='/home/hien/Documents/datasets', download=True, extract=True)
    print('Train: len: {}, num_class: {}, num_camera: {}'.format(len(market1501.train), market1501.get_num_classes('train'), market1501.get_num_camera('train')))
    print('Query: len: {}, num_class: {}, num_camera: {}'.format(len(market1501.query), market1501.get_num_classes('query'), market1501.get_num_camera('query')))
    print('Gallery: len: {}, num_class: {}, num_camera: {}'.format(len(market1501.gallery), market1501.get_num_classes('gallery'), market1501.get_num_camera('gallery')))
