import torch

torch.multiprocessing.set_sharing_strategy("file_system")

from PIL import Image


def imread(path):
    image = Image.open(path)
    return image


class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError(
                "Data caching is disabled and get funciton is unavailable! Check your config."
            )
        return self._dict[str(key)]

    def cache(self, key, value):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = value


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data=None, cache=None, transform=None):
        self.data = data
        self.cache = cache
        self.transform = transform
        self.transform_lib = "torchvision"

    def reset_memory(self):
        if self.cache != None:
            self.cache.reset()
        else:
            raise RuntimeError("not using memory dataset")

    def __getimage__(self, index):
        if self.cache != None:
            path, person_id, camera_id, *_ = self.data[index]

            if self.cache.is_cached(path):
                return self.cache.get(path), person_id, camera_id

            image = imread(path)
            self.cache.cache(path, image)
            return image, person_id, camera_id

        else:
            path, person_id, camera_id, *_ = self.data[index]
            image = imread(path)
            return image, person_id, camera_id

    def __getitem__(self, index):
        image, person_id, camera_id = self.__getimage__(index)

        if self.transform is not None:
            image = self.transform(image)
        return image, person_id, camera_id

    def get_img(self, index):
        img_path, person_id, camera_id = self.data[index]
        img = imread(img_path)
        return img

    def __len__(self):
        return len(self.data)
