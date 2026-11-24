import os
import os.path as osp
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as Data

__all__ = ['NUDTDataset', 'IRSTD1kDataset', 'SirstAugDataset', 'SirstDataset', 'augmentation', 'PadImg']


def _read_grayscale(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Unable to read image: {path}')
    return img


def _normalize_pair(img, mask, base_size):
    if img.shape != (base_size, base_size):
        img = cv2.resize(img, (base_size, base_size), interpolation=cv2.INTER_LINEAR)
    if mask.shape != (base_size, base_size):
        mask = cv2.resize(mask, (base_size, base_size), interpolation=cv2.INTER_NEAREST)
    img = img.reshape(1, base_size, base_size).astype(np.float32) / 255.0
    max_val = np.max(mask)
    if max_val > 0:
        mask = mask.reshape(1, base_size, base_size).astype(np.float32) / float(max_val)
    else:
        mask = mask.reshape(1, base_size, base_size).astype(np.float32)
    return torch.from_numpy(img), torch.from_numpy(mask)


class NUDTDataset(Data.Dataset):
    """NUDT-SIRST dataset loader supporting both img_idx split files and trainval/test folders."""

    def __init__(self, base_dir='./datasets/NUDT-SIRST', mode='train', base_size=256, split_dir=None):
        assert mode in ('train', 'test')
        root = Path(base_dir)
        self.base_size = base_size
        self.mode = mode

        structured = root / ('trainval' if mode == 'train' else 'test')
        if split_dir is None and (structured / 'images').is_dir():
            self.data_dir = structured
            self.names = sorted(p.name for p in (structured / 'images').glob('*.png'))
        else:
            self.data_dir = root if (root / 'images').is_dir() else structured
            candidates = [Path(split_dir)] if split_dir else [root / 'img_idx', root]
            filenames = (
                ('train_NUDT-SIRST.txt', 'trainval.txt', 'train.txt')
                if mode == 'train'
                else ('test_NUDT-SIRST.txt', 'test.txt')
            )
            split_file = None
            for folder in candidates:
                split_file = next((folder / name for name in filenames if (folder / name).is_file()), None)
                if split_file is not None:
                    break
            if split_file is None:
                raise FileNotFoundError(
                    f'NUDT-SIRST requires images/masks + img_idx/train_NUDT-SIRST.txt & test_NUDT-SIRST.txt '
                    f'(or trainval/test folders) under: {root}'
                )
            raw_names = split_file.read_text(encoding='utf-8-sig').split()
            self.names = sorted(n if n.lower().endswith('.png') else f'{n}.png' for n in raw_names)

        if not self.names:
            raise ValueError(f'Empty NUDT-SIRST {mode} split under {base_dir}')

    def __getitem__(self, i):
        name = self.names[i]
        img = _read_grayscale(self.data_dir / 'images' / name)
        mask = _read_grayscale(self.data_dir / 'masks' / name)
        return _normalize_pair(img, mask, self.base_size)

    def __len__(self):
        return len(self.names)


class IRSTD1kDataset(Data.Dataset):
    """IRSTD-1K dataset loader supporting img_idx split files, ISNet layout, and trainval/test folders."""

    def __init__(self, base_dir='./datasets/IRSTD-1K', mode='train', base_size=256):
        assert mode in ('train', 'test')
        root = Path(base_dir)
        self.base_size = base_size
        self.mode = mode

        structured = root / ('trainval' if mode == 'train' else 'test')
        if (structured / 'images').is_dir():
            self.img_dir = structured / 'images'
            self.mask_dir = structured / 'masks'
            self.names = sorted(p.name for p in self.img_dir.glob('*.png'))
        else:
            if (root / 'images').is_dir():
                self.img_dir = root / 'images'
                self.mask_dir = root / 'masks'
            elif (root / 'IRSTD1k_Img').is_dir():
                self.img_dir = root / 'IRSTD1k_Img'
                self.mask_dir = root / 'IRSTD1k_Label'
            else:
                raise FileNotFoundError(f'Could not find images directory under {root}')

            filenames = (
                ('train_IRSTD-1K.txt', 'trainval.txt', 'train.txt')
                if mode == 'train'
                else ('test_IRSTD-1K.txt', 'test.txt')
            )
            split_file = None
            for folder in (root / 'img_idx', root):
                split_file = next((folder / name for name in filenames if (folder / name).is_file()), None)
                if split_file is not None:
                    break
            if split_file is None:
                raise FileNotFoundError(f'Could not find IRSTD-1K split list under {root}')
            raw_names = split_file.read_text(encoding='utf-8-sig').split()
            self.names = [n if n.lower().endswith('.png') else f'{n}.png' for n in raw_names]

        if not self.names:
            raise ValueError(f'Empty IRSTD-1K {mode} split under {base_dir}')

    def __getitem__(self, i):
        name = self.names[i]
        img = _read_grayscale(self.img_dir / name)
        mask = _read_grayscale(self.mask_dir / name)
        return _normalize_pair(img, mask, self.base_size)

    def __len__(self):
        return len(self.names)


class SirstAugDataset(Data.Dataset):
    """SIRST-Aug dataset loader expecting trainval/{images,masks} and test/{images,masks}."""

    def __init__(self, base_dir='./datasets/sirst_aug', mode='train', base_size=256):
        assert mode in ('train', 'test')
        self.mode = mode
        self.base_size = base_size
        self.data_dir = Path(base_dir) / ('trainval' if mode == 'train' else 'test')
        if not (self.data_dir / 'images').is_dir():
            raise FileNotFoundError(f'SIRST-Aug directory not found: {self.data_dir / "images"}')
        self.names = sorted(p.name for p in (self.data_dir / 'images').glob('*.png'))
        self.transform = augmentation()

    def __getitem__(self, i):
        name = self.names[i]
        img = _read_grayscale(self.data_dir / 'images' / name)
        mask = _read_grayscale(self.data_dir / 'masks' / name)
        if self.mode == 'train':
            img, mask = self.transform(img, mask)
        return _normalize_pair(img, mask, self.base_size)

    def __len__(self):
        return len(self.names)


class SirstDataset(Data.Dataset):
    """NUAA-SIRST dataset loader."""

    def __init__(self, base_dir='./datasets/NUAA-SIRST', mode='train', base_size=256):
        assert mode in ('train', 'val', 'test')
        root = Path(base_dir)
        self.mode = mode
        self.base_size = base_size
        self.imgs_dir = root / 'images'
        self.label_dir = root / 'masks'
        candidates = (
            ('train_NUAA-SIRST.txt', 'trainval.txt')
            if mode == 'train'
            else ('test_NUAA-SIRST.txt', 'test1.txt', 'test.txt')
        )
        list_path = next((root / 'img_idx' / name for name in candidates if (root / 'img_idx' / name).is_file()), None)
        if list_path is None:
            raise FileNotFoundError(f'Could not find NUAA-SIRST split file under {root / "img_idx"}')
        self.names = [line.strip() for line in list_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        self.transform = augmentation()

    def __getitem__(self, i):
        name = self.names[i]
        fname = name if name.lower().endswith('.png') else f'{name}.png'
        img = _read_grayscale(self.imgs_dir / fname)
        mask = _read_grayscale(self.label_dir / fname)
        if self.mode == 'train':
            img, mask = self.transform(img, mask)
        return _normalize_pair(img, mask, self.base_size)

    def __len__(self):
        return len(self.names)


class augmentation(object):
    def __call__(self, input_img, target):
        if random.random() < 0.5:
            input_img = input_img[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            input_img = input_img[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            input_img = input_img.transpose(1, 0)
            target = target.transpose(1, 0)
        return input_img.copy(), target.copy()


def PadImg(img, times=32):
    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        _, h, w = img.shape
    else:
        raise ValueError('Unexpected number of dimensions in image')
    if h % times != 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if w % times != 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img
