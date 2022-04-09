from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.images = images
        self.labels = labels
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, label = self.images[index], self.labels[index]
        if self.transform_x is not None:
            data = self.transform_x(Image.open(data))
        else:
            data = Image.open(data)
        if self.transform_y is not None:
            label = self.transform_y(label)
        return data, label


class TransformDataset(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.data = images
        self.targets = labels
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform_x:
            sample = self.transform_x(sample)
        if self.transform_y:
            target = self.transform_y(target)

        return sample, target
