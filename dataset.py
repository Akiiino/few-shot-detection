import os
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

import torch.utils.data as data


def loader(path, transform):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if transform is None:
            transform = lambda x: x
        return transform(img.convert('RGB'))

def make_dataset(dir_, class_to_idx):
    samples = [[] for class_ in class_to_idx]
    dir_ = os.path.expanduser(dir_)
    for target in sorted(os.listdir(dir_)):
        d = os.path.join(dir_, target)
        if not os.path.isdir(d) or target not in class_to_idx:
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                samples[class_to_idx[target]].append(path)

    return samples

class TaskDataset(data.Dataset):
    """A meta-learning data loader where the samples are arranged in this way: ::
        root/class_x/xxx.png
        root/class_x/xxy.png
        root/class_x/xxz.png
        root/class_y/123.png
        root/class_y/nsdf3.png
        root/class_y/asd932_.png

    Each sample from this Dataset represents a single task: `examples_per_class` samples
    of `classes_pre_task` classes, and an additional sample of each class for validation.

    Args:
        root (string): Root directory path.
        classes (iterable): which classes to sample; used for meta-validation
        classes_per_task (int)
        examples_per_class (int)
        len (integer, optional): fictive length of the dataset
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, classes, classes_per_task=5, examples_per_class=1, len_=2000, transform=None):
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        samples = make_dataset(root, class_to_idx)

        if len(samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: {}\n".format(root))

        self.root = root
        self.len = len_
        self.classes_per_task = classes_per_task
        self.examples_per_class = examples_per_class

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        imgs = [
            [
                loader(path, self.transform) for path in np.random.choice(class_, self.examples_per_class)
            ] for class_ in (self.samples[x] for x in np.random.choice(len(self.samples), self.classes_per_task, False))
        ]

        x = list(zip(*imgs))

        return x

    def __len__(self):
        return self.len

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def make_split(root, n_classes=5, examples_per_class=1, test_size=0.5):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()

    train_classes, test_classes = train_test_split(classes, test_size=test_size)

    t = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    def collate(x):
        batch = data.dataloader.default_collate(x)

        return [torch.stack(ex)[:, 0] for ex in batch]

    train_data = data.DataLoader(
        TaskDataset(root, train_classes, transform=t, classes_per_task=n_classes, examples_per_class=examples_per_class + 1, len_=1000),
        collate_fn=collate,
    )
    test_data = data.DataLoader(
        TaskDataset(root, test_classes, transform=t, classes_per_task=n_classes, examples_per_class=examples_per_class + 1, len_=100),
        collate_fn=collate
    )

    return train_data, test_data
